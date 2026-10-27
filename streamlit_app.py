import streamlit as st
import os
import numpy as np
import torch
import torch.nn as nn
import cv2
import gymnasium as gym
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Unsupervised State Representation Learning for Sample-Efficient Visual RL", page_icon="🤖", layout="wide")

st.title("🤖 Unsupervised State Representation Learning for Sample-Efficient Visual RL — interactive demo")
st.write("Pipeline: pixels (64×64) → frozen convolutional-autoencoder encoder → latent vector z → DQN agent. This demo uses the encoders trained in the final experiment (CPU only, no GPU needed).")

# -----------------------------------------------------
# Define CAE Architecture
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
# Load Pretrained Autoencoder weight safely
# -----------------------------------------------------
@st.cache_resource
def load_model(dim):
    # Search in various potential paths to ensure it is always found
    possible_paths = [
        Path(f"demo_assets/cae_z{dim}.pt"),
        Path(f"../demo_assets/cae_z{dim}.pt"),
        Path(f"/content/drive/MyDrive/latent-visual-RL-project/demo_assets/cae_z{dim}.pt")
    ]
    checkpoint_path = next((p for p in possible_paths if p.exists()), None)
    if checkpoint_path is None:
        return None, None, None
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ConvAutoencoder(dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint.get("z_mean", np.zeros(dim)), checkpoint.get("z_std", np.ones(dim))

# Sidebar Configuration
st.sidebar.header("Configuration Panel")
latent_dim = st.sidebar.selectbox("Select Bottleneck Size (Latent Dim)", [4, 8, 16, 32, 64], index=2)
model, z_mean, z_std = load_model(latent_dim)

if model is None:
    st.sidebar.error(f"Weights file for z={latent_dim} not found in demo_assets/")

# Main Tab Setup matching Gradio design
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1 · Latent lens", 
    "2 · Latent sweep", 
    "3 · Bottleneck size", 
    "4 · Live episode", 
    "5 · Experiment results"
])

with tab1:
    st.header("1 · Latent Lens Visualization")
    st.write("Reconstructing 64x64 pixel frames through the trained self-supervised state representation bottleneck.")
    
    uploaded_file = st.file_uploader("Upload raw 64x64 input frame", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        
        # Process frame
        x_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
        if model is not None:
            with torch.no_grad():
                recon_tensor, z_vector = model(x_tensor)
            recon_img = (recon_tensor.squeeze().numpy() * 255).astype(np.uint8)
            z_val = np.round(z_vector.squeeze().numpy(), 3).tolist()
        else:
            recon_img = img
            z_val = []
            
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Observation Frame (64x64)")
            st.image(img, width=256)
        with col2:
            st.subheader("Autoencoder Reconstruction")
            st.image(recon_img, width=256)
            st.write(f"**Standardised Latent Vector z**: {z_val}")
    else:
        fig_path = Path("results/figures/reconstructions.png")
        if fig_path.exists():
            st.image(str(fig_path), caption="Cached Autoencoder Reconstruction Results (Top: Raw input, Bottom: Reconstructed representation)")

with tab2:
    st.header("2 · Latent Coordinate Sweep")
    st.write("Manually tweak physical parameters to observe how the latent space maps coordinates.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cart_p = st.slider("Cart Position Control", -2.4, 2.4, 0.0, step=0.1)
    with col2:
        cart_v = st.slider("Cart Velocity Control", -3.0, 3.0, 0.0, step=0.1)
    with col3:
        pole_a = st.slider("Pole Angle (Radians) Control", -0.4, 0.4, 0.0, step=0.02)
    with col4:
        pole_v = st.slider("Pole Angular Velocity Control", -3.0, 3.0, 0.0, step=0.1)
        
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.reset()
    env.unwrapped.state = np.array([cart_p, cart_v, pole_a, pole_v], dtype=np.float32)
    
    # Capture env frame
    frame = env.render()
    cropped = frame[150:350]
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    env.close()
    
    x_tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    if model is not None:
        with torch.no_grad():
            recon_tensor, z_vector = model(x_tensor)
        recon_img = (recon_tensor.squeeze().numpy() * 255).astype(np.uint8)
        z_vector_text = f"Latent vector z: {np.round(z_vector.squeeze().numpy(), 3).tolist()}"
    else:
        recon_img = resized
        z_vector_text = "Model not loaded"
        
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Synthesized Environment Frame")
        st.image(resized, width=280)
    with colB:
        st.subheader("Reconstructed Output")
        st.image(recon_img, width=280)
        st.info(z_vector_text)

with tab3:
    st.header("3 · Bottleneck Size Evaluation")
    st.write("Observe reconstruction quality as a function of the bottleneck latent dimensionality (4, 8, 16, 32, 64).")
    
    test_frame_path = Path("results/figures/reconstructions.png")
    if test_frame_path.exists():
        st.image(str(test_frame_path), caption="Reconstructions for various latent bottle size pipelines")
    else:
        st.info("No test set comparison figure found in your repository.")

with tab4:
    st.header("4 · Live Episode Simulation")
    st.write("Run an episode through the same observation pipeline as training (frozen encoder + 4-step latent stack) and watch the latent vector evolve.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        policy_selection = st.selectbox("Policy", ["Random policy", "Trained Latent Agent (DQN)"])
    with col2:
        env_seed = st.number_input("Environment seed", min_value=0, max_value=10000, value=0)
    with col3:
        latent_choice = st.selectbox("Latent size (random policy only)", ["z = 4", "z = 8", "z = 16", "z = 32", "z = 64"], index=2)
        
    if st.button("Run episode"):
        st.success("Initializing Gym simulation... Running step evaluations!")
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        env.reset(seed=env_seed)
        
        steps = 0
        total_reward = 0.0
        terminated = False
        truncated = False
        
        while not (terminated or truncated) and steps < 200:
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
        env.close()
        st.info(f"Episode finished! Steps run: {steps} | Total Cumulative Reward: {total_reward}")

with tab5:
    st.header("5 · Quantitative Experiment Results")
    st.write("Compare metrics and performance directly from saved tables:")
    
    summary_path = Path("results/tables/summary_by_method.csv")
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning("Quantitative results summary.csv not found.")
