import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch.nn import MultiheadAttention
import torch.optim as optim
import os
import mediapipe as mp
import numpy as np
import cv2
import base64
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# --- 1. Define your GNN Model Architecture ---
# (This MUST be the *exact* same class definition you used for training)
class SignLanguageGCN(nn.Module):
    def __init__(self, num_keypoints, num_features, layer_dims=[64, 128], num_classes=26, dropout_prob=0.5):
        super(SignLanguageGCN, self).__init__()

        self.spatial_gcn_layers = nn.ModuleList()
        self.spatial_bn_layers = nn.ModuleList()
        self.attention_gcn_layers = nn.ModuleList()
        self.attention_bn_layers = nn.ModuleList()
        self.spatial_proj_layers = nn.ModuleList()
        self.attention_proj_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout_prob)

        input_dim = num_features
        for hidden_dim in layer_dims:
            self.spatial_gcn_layers.append(GCNConv(input_dim, hidden_dim))
            self.spatial_bn_layers.append(nn.BatchNorm1d(hidden_dim))
            self.attention_gcn_layers.append(GCNConv(input_dim, hidden_dim))
            self.attention_bn_layers.append(nn.BatchNorm1d(hidden_dim))

            if input_dim != hidden_dim:
                self.spatial_proj_layers.append(nn.Linear(input_dim, hidden_dim))
                self.attention_proj_layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                self.spatial_proj_layers.append(None)
                self.attention_proj_layers.append(None)
            input_dim = hidden_dim

        self.attention = MultiheadAttention(embed_dim=layer_dims[-1], num_heads=4)
        self.fc1 = nn.Linear(2 * layer_dims[-1], 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x, edge_index, batch):
        residual_spatial = x
        residual_attention = x
        for gcn, bn, proj in zip(self.spatial_gcn_layers, self.spatial_bn_layers, self.spatial_proj_layers):
            x_spatial = F.relu(bn(gcn(residual_spatial, edge_index)))
            x_spatial = self.dropout(x_spatial)
            if proj is not None:
                residual_spatial = proj(residual_spatial)
            x_spatial += residual_spatial
            residual_spatial = x_spatial

        for gcn, bn, proj in zip(self.attention_gcn_layers, self.attention_bn_layers, self.attention_proj_layers):
            x_attention = F.relu(bn(gcn(residual_attention, edge_index)))
            x_attention = self.dropout(x_attention)
            if proj is not None:
                residual_attention = proj(residual_attention)
            x_attention += residual_attention
            residual_attention = x_attention

        x_attention = x_attention.unsqueeze(1).permute(1, 0, 2)
        x_attention, _ = self.attention(x_attention, x_attention, x_attention)
        x_attention = x_attention.squeeze(0)
        x_attention = x_attention * residual_attention

        x = torch.cat([x_spatial, x_attention], dim=1)
        x = global_mean_pool(x, batch)
        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn_fc2(self.fc2(x))))
        # Return raw logits, as CrossEntropyLoss will be used (or log_softmax if NLLLoss)
        # Let's assume you remove log_softmax for use with CrossEntropyLoss
        # If your saved model *has* log_softmax, use nn.NLLLoss in training
        # For this server, we just need the argmax, so it doesn't matter.
        x = self.fc3(x) 
        return F.log_softmax(x, dim=-1) # Keep if your model was trained with it


# --- 2. Define Helper Functions ---

# !! IMPORTANT !!
# This list must match the *exact* order of classes your model was trained on.
# Update this list based on your classification report.
CLASS_NAMES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9','null',
    'a', 'b', 'bye', 'c', 'd', 'e', 'good', 'good morning', 'hello',
    'little bit', 'no', 'pardon', 'please', 'project', 'whats up', 'yes',
]
NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = 'sign_model.pth' # <-- Put your model's filename here

# Define the hand skeleton edges (must be same as training)
hand_edges = [
    [0, 1], [1, 2], [2, 3], [3, 4],    # Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],    # Index finger
    [0, 9], [9, 10], [10, 11], [11, 12], # Middle finger
    [0, 13], [13, 14], [14, 15], [15, 16], # Ring finger
    [0, 17], [17, 18], [18, 19], [19, 20]  # Pinky
]
undirected_hand_edges = hand_edges + [[j, i] for i, j in hand_edges]
edge_index = torch.tensor(undirected_hand_edges, dtype=torch.long).t().contiguous()

# Define the keypoint normalization function (must be same as training)
def normalize_keypoints(kp_set):
    if kp_set.shape[0] == 0:
        return kp_set
    min_vals = kp_set.min(axis=0, keepdims=True)
    max_vals = kp_set.max(axis=0, keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1e-8 # Avoid division by zero
    normalized_kp_set = (kp_set - min_vals) / range_vals
    return normalized_kp_set

# --- 3. Load Model and Initialize MediaPipe ---

# Load your trained GNN model
device = torch.device('cpu') # Run inference on CPU
model = SignLanguageGCN(num_keypoints=21, num_features=3, layer_dims=[64, 128], num_classes=NUM_CLASSES)

# Load the saved weights
# Adjust this line if you saved the whole model vs. just the state_dict
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device)['model_state_dict'])
except:
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure 'sign_model.pth' is in the same directory and the class definition is correct.")
        exit()

model.to(device)
model.eval() # Set model to evaluation mode (CRITICAL!)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, # Process a video stream
    max_num_hands=1,         # We only care about one hand
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONT_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'front'))
# --- 4. Setup Flask Server ---
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

@app.route('/')
def index():
    """Serve the index.html file from the 'front' directory."""
    
    # Check if the file exists to give a better error message if it doesn't
    if not os.path.isfile(os.path.join(FRONT_DIR, 'index.html')):
        return "Error: index.html not found at " + FRONT_DIR, 404
        
    return send_from_directory(FRONT_DIR, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receive a webcam frame, process it, and return a prediction."""
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image data sent.'}), 400
    
    # 1. Decode Base64 image
    img_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Flip the image horizontally (webcam feed is mirrored)
    img = cv2.flip(img, 1)
    
    # 2. Process with MediaPipe
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if not results.multi_hand_landmarks:
        return jsonify({'prediction': '---'})

    # 3. Extract, Normalize, and Format Keypoints
    keypoints = []
    for landmark in results.multi_hand_landmarks[0].landmark:
        keypoints.append([landmark.x, landmark.y, landmark.z])
    
    keypoints_np = np.array(keypoints)
    normalized_keypoints = normalize_keypoints(keypoints_np)
    
    # 4. Create Graph Data Object
    x = torch.tensor(normalized_keypoints, dtype=torch.float).to(device)
    
    # Note: edge_index is static and defined globally
    graph_data = Data(x=x, edge_index=edge_index.to(device))
    
    # 5. Create a Batch (model expects a batch)
    batch = Batch.from_data_list([graph_data]).to(device)
    
    # 6. Make Prediction
    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch.batch)
        pred_idx = out.argmax(dim=1).item()
        prediction = CLASS_NAMES[pred_idx]
        
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    print(f"Model {MODEL_PATH} loaded successfully.")
    print(f"Recognizing {NUM_CLASSES} classes.")
    print("Starting Flask server... Open http://127.0.0.1:5000 in your browser.")
    app.run(debug=True, port=5000)
