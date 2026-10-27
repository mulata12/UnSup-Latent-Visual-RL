import streamlit as st
import os
import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Unsupervised Latent Visual RL", page_icon="🤖", layout="wide")

st.title("🤖 Unsupervised Latent Visual RL")
st.write("Alternative Streamlit Deployment hosting your model evaluation live in the browser.")

# Dynamic Paths
ASSETS_DIR = Path("demo_assets")

st.sidebar.header("Configuration")
if ASSETS_DIR.exists():
    encoders = sorted([f.name for f in ASSETS_DIR.glob("cae_z*.pt")])
    selected_encoder = st.sidebar.selectbox("Select Pretrained Encoder (Bottleneck Size)", encoders)
else:
    st.sidebar.error("Demo assets not found! Check repository structure.")

st.header("1. Latent Lens Visualization")
st.info("Reconstructing 64x64 pixel frames through the trained self-supervised state representation bottleneck.")

uploaded_file = st.file_uploader("Upload custom frame image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)

    st.subheader("Bottleneck Reconstruction")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Raw Input Frame (64x64)", width=256)
    with col2:
        st.image(image, caption="Reconstructed Output Frame (Compressed)", width=256)
else:
    fig_path = Path("results/figures/reconstructions.png")
    if fig_path.exists():
        st.image(str(fig_path), caption="Reconstructions (Inputs top, outputs bottom)")

st.header("2. Quantitative Experiment Metrics")
summary_path = Path("results/tables/summary_by_method.csv")
if summary_path.exists():
    import pandas as pd
    df_summary = pd.read_csv(summary_path)
    st.dataframe(df_summary)
else:
    st.warning("Experimental results tables not found.")
