import streamlit as st
import os
import numpy as np
import torch
import torch.nn as nn
import cv2
import gymnasium as gym
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Unsupervised Latent Visual RL", page_icon="🤖", layout="wide")

st.title("🤖 Unsupervised Latent Visual RL")
st.write("Interactive model visualizations and Gym Environment controller running entirely in the browser.")

# -----------------------------------------------------
# Define the CAE Architecture
# -----------------------------------------------------
class ConvEncoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, kernel_size=4, stride=2, padding=1),
                                 nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.conv = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 256))
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)
    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        def block(i, o):
            return nn.Sequential(nn.ConvTranspose2d(i, o, kernel_size=4, stride=2, padding=1),
                                 nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.deconv = nn.Sequential(block(256, 128), block(128, 64), block(64, 32),
                                    nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid())
    def forward(self, z):
        return self.deconv(self.fc(z).view(-1, 256, 4, 4))

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder, self.decoder = ConvEncoder(latent_dim), ConvDecoder(latent_dim)
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

# -----------------------------------------------------
# Load Pretrained Autoencoder model weights
# -----------------------------------------------------
@st.cache_resource
def load_model(dim):
    checkpoint_path = Path(f"demo_assets/cae_z{dim}.pt")
    if not checkpoint_path.exists():
        return None, None, None
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ConvAutoencoder(dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint.get("z_mean", np.zeros(dim)), checkpoint.get("z_std", np.ones(dim))

# Sidebar options
st.sidebar.header("Configuration")
latent_dim = st.sidebar.selectbox("Select Bottleneck Size (Latent Dim)", [4, 8, 16, 32, 64], index=2)
model, z_mean, z_std = load_model(latent_dim)

if model is None:
    st.sidebar.error(f"Weights file for z={latent_dim} not found in demo_assets/")

# App Tabs
tab1, tab2, tab3 = st.tabs(["🔍 Latent Lens Visualizer", "🕹️ Interactive CartPole Simulator", "📈 Performance Metrics"])

with tab1:
    st.header("1. Dynamic Image Reconstruction Bottleneck")
    st.write("Upload or draw any image to see how well the model compresses and reconstructs physical features through its narrow latent space.")
    
    uploaded_file = st.file_uploader("Upload a frame", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # Preprocess uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        
        # Run model forward pass
        x_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
        with torch.no_grad():
            recon_tensor, z_vector = model(x_tensor)
        recon_img = (recon_tensor.squeeze().numpy() * 255).astype(np.uint8)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Frame")
            st.image(img, width=300)
        with col2:
            st.subheader("Reconstructed Frame")
            st.image(recon_img, width=300)
            st.write(f"Latent State Vector z: {np.round(z_vector.squeeze().numpy(), 3).tolist()}")
    else:
        st.info("No uploaded image. Showing pre-rendered test set reconstruction outputs.")
        fig_path = Path("results/figures/reconstructions.png")
        if fig_path.exists():
            st.image(str(fig_path), caption="Inputs (Top) vs Autoencoder Reconstructions (Bottom)")

with tab2:
    st.header("2. Live Interactive CartPole Environment")
    st.write("Modify Gym environment state variables in real-time, view rendering, and analyze reconstructed representations:")
    
    # State control sliders
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cart_pos = st.slider("Cart Position", -2.4, 2.4, 0.0)
    with col2:
        cart_vel = st.slider("Cart Velocity", -3.0, 3.0, 0.0)
    with col3:
        pole_ang = st.slider("Pole Angle (Rad)", -0.4, 0.4, 0.0)
    with col4:
        pole_vel = st.slider("Pole Angular Velocity", -3.0, 3.0, 0.0)
        
    # Instantiate gym and set state
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.reset()
    env.unwrapped.state = np.array([cart_pos, cart_vel, pole_ang, pole_vel], dtype=np.float32)
    
    # Obtain render frame and crop
    raw_frame = env.render()
    cropped = raw_frame[150:350]
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    env.close()
    
    # Generate output via neural network
    x_tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    if model is not None:
        with torch.no_grad():
            recon_tensor, z_vector = model(x_tensor)
        recon_img = (recon_tensor.squeeze().numpy() * 255).astype(np.uint8)
    else:
        recon_img = resized
        
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Original Render Frame (64x64)")
        st.image(resized, width=300)
    with colB:
        st.subheader("Autoencoder Reconstruction")
        st.image(recon_img, width=300)

with tab3:
    st.header("3. Quantitative Metrics & Comparisons")
    summary_path = Path("results/tables/summary_by_method.csv")
    if summary_path.exists():
        import pandas as pd
        df_summary = pd.read_csv(summary_path)
        st.dataframe(df_summary)
    else:
        st.warning("Experimental tables and figures were not loaded.")
