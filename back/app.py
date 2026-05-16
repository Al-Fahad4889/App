import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch.nn import MultiheadAttention
import os
import urllib.request

# --- Model Architecture ---
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
        x = self.fc3(x) 
        return F.log_softmax(x, dim=-1)

# --- Configuration & Loading ---
MODEL_PATH = 'sign_model.pth'
HAND_MODEL_PATH = 'hand_landmarker.task'
CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','NULL', 'A', 'B', 'BYE', 'C', 'D', 'E', 'GOOD', 'GOOD MORNING', 'HELLO', 'LITTLE BIT', 'NO', 'PARDON', 'PLEASE', 'PROJECT', 'WHATS UP', 'YES']

# Hand skeleton edges
hand_edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
undirected_hand_edges = hand_edges + [[j, i] for i, j in hand_edges]
edge_index = torch.tensor(undirected_hand_edges, dtype=torch.long).t().contiguous()

def normalize_keypoints(kp_set):
    if kp_set.shape[0] == 0: return kp_set
    min_v = kp_set.min(axis=0, keepdims=True)
    max_v = kp_set.max(axis=0, keepdims=True)
    rng = max_v - min_v
    rng[rng == 0] = 1e-8
    return (kp_set - min_v) / rng

@st.cache_resource
def load_models():
    device = torch.device('cpu')
    model = SignLanguageGCN(num_keypoints=21, num_features=3, layer_dims=[64, 128], num_classes=len(CLASS_NAMES))
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    model.eval()

    if not os.path.exists(HAND_MODEL_PATH):
        urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", HAND_MODEL_PATH)
    
    base_options = python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)
    
    return model, detector, device

# --- Video Processing Class ---
class SignLanguageTransformer(VideoTransformerBase):
    def __init__(self, model, detector, device):
        self.model = model
        self.detector = detector
        self.device = device
        self.latest_prediction = "..."

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Resize for speed
        img_small = cv2.resize(img, (320, 240))
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        detection_result = self.detector.detect(mp_image)
        
        if detection_result.hand_landmarks:
            keypoints = np.array([[lm.x, lm.y, lm.z] for lm in detection_result.hand_landmarks[0]])
            norm_kp = normalize_keypoints(keypoints)
            
            x_tensor = torch.tensor(norm_kp, dtype=torch.float).to(self.device)
            graph_data = Data(x=x_tensor, edge_index=edge_index)
            batch = Batch.from_data_list([graph_data])
            
            with torch.no_grad():
                out = self.model(batch.x, batch.edge_index, batch.batch)
                self.latest_prediction = CLASS_NAMES[out.argmax(dim=1).item()]
        else:
            self.latest_prediction = "..."
            
        return img

# --- Streamlit UI ---
st.set_page_config(page_title="SignFlow Streamlit", layout="centered")
st.title("🤟 Real-Time Sign Language Recognition")
st.markdown("This version uses WebRTC for lower latency streaming.")

model_gnn, hand_detector, dev = load_models()

webrtc_ctx = webrtc_streamer(
    key="sign-lang",
    video_transformer_factory=lambda: SignLanguageTransformer(model_gnn, hand_detector, dev),
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

if webrtc_ctx.video_transformer:
    st.markdown(f"""
        <div style="background-color: #1F2937; border: 4px solid #3B82F6; border-radius: 10px; padding: 20px; text-align: center;">
            <h1 style="color: #93C5FD; font-size: 4rem; margin: 0;">{webrtc_ctx.video_transformer.latest_prediction}</h1>
        </div>
    """, unsafe_content_supported=True)
