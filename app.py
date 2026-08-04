import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import gdown

@st.cache_resource
def download_and_load_models():
    ground_path = "ground_water_stress.pth"
    if not os.path.exists(ground_path):
        ground_id = "1WDDSMYceMJ9NrzdGAkVnun4kNEtWE1PG"
        gdown.download(id=ground_id, output=ground_path, quiet=False)

    sat_path = "satellite_droughtwatch.pth"
    if not os.path.exists(sat_path):
        sat_id = "17h_ATL2kZrS0VTXIMXpytH6VsFi1jSB8"
        gdown.download(id=sat_id, output=sat_path, quiet=False)
download_and_load_models()
from model import (
    GroundDroughtModel,
    SatelliteDroughtModel,
    calculate_pred,
    generate_gradcam,
    generate_prescriptive_drills,
)

st.set_page_config(
    page_title="Drought Watch AI Platform",
    page_icon="🌾",
    layout="wide",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GROUND_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@st.cache_resource
def load_ground_model():
    model = GroundDroughtModel()
    try:
        model.load_state_dict(torch.load("ground_water_stress.pth", map_location=DEVICE))
        st.sidebar.success(" Loaded Ground Model")
    except Exception as e:
        st.sidebar.warning(" Could not load ground_water_stress.pth (using unweighted model)")
    model.to(DEVICE)
    model.eval()
    return model

@st.cache_resource
def load_satellite_model():
    model = SatelliteDroughtModel(in_channels=10, num_classes=4)
    try:
        model.load_state_dict(torch.load("satellite_droughtwatch.pth", map_location=DEVICE))
        st.sidebar.success(" Loaded Satellite Model")
    except Exception as e:
        st.sidebar.warning(" Could not load satellite_droughtwatch.pth (using unweighted model)")
    model.to(DEVICE)
    model.eval()
    return model

ground_model = load_ground_model()
satellite_model = load_satellite_model()

st.title("🌾 Dual-Scale Drought Intelligence Platform")
st.markdown(f"**Hardware Acceleration status:** `{DEVICE.type.upper()}`")
st.markdown("---")

tab1, tab2 = st.tabs(["🌿 Microscopic Ground Leaf Analysis", "🛰️ Regional Satellite Assessment"])

with tab1:
    st.header("Ground Leaf Stress Diagnostic")
    st.write("Upload a close-up photograph of plant leaves to analyze cellular moisture stress and inspect Grad-CAM focus areas.")

    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        raw_img = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Input")
            st.image(raw_img, use_container_width=True)

        input_tensor = GROUND_TRANSFORM(raw_img).unsqueeze(0).to(DEVICE)

        if st.button("Run Diagnostic", type="primary"):
            with st.spinner("Analyzing cell wall structure & running Grad-CAM..."):
                result = calculate_pred(input_tensor, ground_model)
                risk_score = result["risk_score"]
                tier, drills = generate_prescriptive_drills(risk_score)

                gradcam_img = generate_gradcam(input_tensor, ground_model, raw_img)

            with col2:
                st.subheader("Grad-CAM Explainability Map")
                st.image(gradcam_img, use_container_width=True)

            st.markdown("---")
            st.subheader("Diagnostic Results")

            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric(label="Drought Risk Index", value=f"{risk_score * 100:.1f}%")
                st.progress(risk_score)
            
            with metric_col2:
                if risk_score >= 0.7:
                    st.error(f"Alert Level: {tier}")
                elif risk_score >= 0.4:
                    st.warning(f"Alert Level: {tier}")
                else:
                    st.success(f"Alert Level: {tier}")

            st.subheader("📋 Prescriptive Drill Protocols")
            for step in drills:
                st.markdown(f"* {step}")

with tab2:
    st.header("Regional Landsat 8 Forage Index")
    st.write("Analyze 10-band multi-spectral satellite tensor arrays to determine grazing land capacity.")

    st.info("Simulate satellite array tensor feed or test demo arrays.")

    if st.button("Generate Random 10-Band Satellite Patch", type="primary"):
        with st.spinner("Processing multi-spectral array..."):
            dummy_satellite_tensor = torch.randn(1, 10, 65, 65).to(DEVICE)
            
            res = calculate_pred(dummy_satellite_tensor, satellite_model)
            pred_class = res["predicted_class"]
            probs = res["class_probabilities"]

        st.subheader("Model Prediction")
        
        class_labels = [
            "Class 0: 0% Forage (Barren / Desert)",
            "Class 1: 1-30% Forage (Sparse Vegetation)",
            "Class 2: 31-60% Forage (Moderate Growth)",
            "Class 3: >60% Forage (Dense Pasture)",
        ]

        st.success(f"**Predicted Tier:** {class_labels[pred_class]}")

        st.subheader("Probability Distribution Across Classes")
        for i, (label, prob) in enumerate(zip(class_labels, probs)):
            st.write(f"**{label}**")
            st.progress(prob)