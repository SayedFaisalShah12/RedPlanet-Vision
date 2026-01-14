import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path
import torch
# import matplotlib.pyplot as plt
import sys

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import PROCESSED_DATA_DIR, RESULTS_DIR, MODELS_DIR
from src.explainability import run_explainability

st.set_page_config(page_title="RadPlanet-Vision", layout="wide")

st.title("🔴 RadPlanet-Vision: Mars Terrain Analysis")
st.markdown("Autonomous AI analysis of Martian surface imagery from Curiosity & Perseverance.")

# Sidebar
st.sidebar.header("Control Panel")
mode = st.sidebar.radio("Navigation", ["Dataset Explorer", "Model Inference", "Training History"])

if mode == "Dataset Explorer":
    st.header("📂 Dataset Exploration")
    
    # Load stats
    if (RESULTS_DIR / "eda/camera_distribution.png").exists():
        col1, col2 = st.columns(2)
        with col1:
            st.image(str(RESULTS_DIR / "eda/camera_distribution.png"), caption="Camera Distribution")
        with col2:
            st.image(str(RESULTS_DIR / "eda/sol_distribution.png"), caption="Sol Distribution")
    
    # Gallery
    st.subheader("Processed Clusters")
    if PROCESSED_DATA_DIR.exists():
        classes = [d.name for d in PROCESSED_DATA_DIR.glob("*/*") if d.is_dir()]
        selected_class = st.selectbox("Select Terrain Type", classes)
        
        # Show images from that class
        class_path = list(PROCESSED_DATA_DIR.glob(f"*/{selected_class}"))[0]
        images = list(class_path.glob("*.jpg"))[:10]
        
        if images:
            cols = st.columns(5)
            for idx, img_path in enumerate(images):
                with cols[idx % 5]:
                    st.image(str(img_path), caption=img_path.name)
        else:
            st.warning("No images found in this class.")

elif mode == "Model Inference":
    st.header("🧠 Live Model Inference & Explainability")
    
    # Load Model
    model_path = list(MODELS_DIR.glob("*_best.pth"))[0]
    # Cache model to avoid reloading
    @st.cache_resource
    def load_model(path):
        return torch.load(path, weights_only=False).eval()
    
    model = load_model(model_path)
    st.success(f"Loaded Model: {model_path.name}")
    
    # Pick Image
    st.subheader("Select Test Image")
    all_images = list(PROCESSED_DATA_DIR.rglob("*.jpg"))
    if not all_images:
        st.error("No images found.")
    else:
        # User selects index
        img_idx = st.slider("Image Index", 0, len(all_images)-1, 0)
        selected_img = all_images[img_idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(str(selected_img), caption="Original Input", width=400)
            
        with col2:
            if st.button("Run Analysis"):
                with st.spinner("Analyzing terrain textures..."):
                    # Run GradCAM
                    target_layer = model.layer4[-1]
                    save_path = RESULTS_DIR / "temp_gradcam.jpg"
                    run_explainability(model, str(selected_img), target_layer)
                    # Check if saved to specific ID name or generic
                    # run_explainability saves to RESULTS_DIR / f"gradcam_{Path(image_path).stem}.jpg"
                    result_path = RESULTS_DIR / f"gradcam_{selected_img.stem}.jpg"
                    
                    st.image(str(result_path), caption="Grad-CAM Attention Map", width=400)
                    st.info("Red areas indicate regions influencing the classification.")

elif mode == "Training History":
    st.header("📈 Training Performance")
    
    hist_path = MODELS_DIR / "training_history.png"
    if hist_path.exists():
        st.image(str(hist_path), use_column_width=True)
    
    report_path = RESULTS_DIR / "classification_report.txt"
    if report_path.exists():
        with open(report_path) as f:
            st.text(f.read())
