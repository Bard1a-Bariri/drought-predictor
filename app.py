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
    if os.path.exists(ground_path) and os.path.getsize(ground_path) < 1_000_000:
        os.remove(ground_path)
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
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

SATELLITE_TRANSFORM = transforms.Compose([
    transforms.Resize((65, 65)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@st.cache_resource
def load_ground_model():
    model = GroundDroughtModel()
    loaded_successfully = False
    try:
        model.load_state_dict(
            torch.load("ground_water_stress.pth", map_location=DEVICE, weights_only=False),
            strict=False
        )
        loaded_successfully = True
    except Exception as e:
        loaded_successfully = False

    model.to(DEVICE)
    model.eval()
    
    return model, loaded_successfully

ground_model, is_loaded = load_ground_model()

if is_loaded:
    st.sidebar.success("Loaded Ground Model")
else:
    st.sidebar.warning("Could not load ground_water_stress.pth (using unweighted model)")
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

satellite_model = load_satellite_model()

st.title("🌾 TerraSight")
st.markdown("---")

tab1, tab2 = st.tabs(["🌿Ground Assesment", "🛰️ Satellite Assessment"])

with tab1:
    st.header("Ground Drought Calculator")
    st.write("Upload a photo of plant leaves to analyze water stress levels...")

    uploaded_file = st.camera_input("Take a picture...")

    if uploaded_file is not None:
        raw_img = Image.open(uploaded_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Input")
            st.image(raw_img, use_container_width=True)

        input_tensor = GROUND_TRANSFORM(raw_img).unsqueeze(0).to(DEVICE)

        if st.button("Run", type="primary"):
            with st.spinner("Analyzing image & preparing heatmap..."):
                result = calculate_pred(input_tensor, ground_model)
                risk_score = result["risk_score"]
                tier, drills = generate_prescriptive_drills(risk_score)

                gradcam_img = generate_gradcam(input_tensor, ground_model, raw_img)

            with col2:
                st.subheader("Corresponding Heatmap")
                st.image(gradcam_img, use_container_width=True)

            st.markdown("---")
            st.subheader("Results")

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

            st.subheader("📋 What to do next")
            for step in drills:
                st.markdown(f"* {step}")

with tab2:
    st.header("Satellite Landscape Index")
    st.write("Upload a satellite tile image to determine grazing land capacity.")

    sat_file = st.file_uploader("Upload Satellite Tile (JPG/PNG)", type=["jpg", "jpeg", "png"], key="sat_uploader")

    if sat_file is not None:
        raw_sat_img = Image.open(sat_file).convert("RGB")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Uploaded Tile")
            st.image(raw_sat_img, use_container_width=True)

        rgb_tensor = SATELLITE_TRANSFORM(raw_sat_img)

        
        R = rgb_tensor[0:1, :, :]
        G = rgb_tensor[1:2, :, :]
        B = rgb_tensor[2:3, :, :]

        NIR = G * 2.0         
        SWIR1 = G * 0.5       
        SWIR2 = R * 0.3       


        ten_band_tensor = torch.cat([
            B,      
            B,      
            G,     
            R,      
            NIR,    
            SWIR1,  
            SWIR2,  
            G,     
            B,      
            R      
        ], dim=0)

        input_satellite_tensor = ten_band_tensor.unsqueeze(0).to(DEVICE)

        if st.button("Run Satellite Analysis", type="primary"):
            with st.spinner("Processing spectral array..."):
                res = calculate_pred(input_satellite_tensor, satellite_model)
                pred_class = res["predicted_class"]
                probs = res["class_probabilities"]

                drought_risk_score = (probs[0] * 1.00) + (probs[1] * 0.85) + (probs[2] * 0.10) + (probs[3] * 0.00)
                drought_percentage = float(drought_risk_score) * 100        

                if drought_percentage >= 60:
                    status_tier = "CRITICAL DROUGHT RISK"
                    status_color = "error"
                    next_steps = [
                        "🚨 **Emergency Livestock Relocation:** Initiate pasture transfer or supplemental feeding immediately.",
                        "💧 **Water Management:** Enforce immediate agricultural water rationing in high-risk zones.",
                        "🛰️ **High-Frequency Monitoring:** Schedule daily satellite spectral re-scans."
                    ]
                elif drought_percentage >= 30:
                    status_tier = "MODERATE DROUGHT WARNING"
                    status_color = "warning"
                    next_steps = [
                        "🌾 **Rotational Grazing:** Reduce grazing density on sparse vegetation patches.",
                        "🚰 **Irrigation Efficiency:** Audit and adjust drip/sprinkler systems for targeted delivery.",
                        "📊 **Soil Moisture Audits:** Perform ground-level soil testing in vulnerable sections."
                    ]
                else:
                    status_tier = "HEALTHY / MINIMAL DROUGHT RISK"
                    status_color = "success"
                    next_steps = [
                        "✅ **Maintain Standard Rotation:** Forage capacity is sufficient for normal herd density.",
                        "🌱 **Soil Health Monitoring:** Keep standard seasonal monitoring schedule.",
                        "🌧️ **Rainwater Capture:** Prepare infrastructure for upcoming dry cycles."
                    ]

            with col2:
                st.subheader("Model Diagnostics")
                
                st.metric(label="Calculated Drought Risk Index", value=f"{drought_percentage:.1f}%")
                st.progress(float(drought_risk_score))

                class_labels = [
                    "Class 0: Barren / Desert (High Risk)",
                    "Class 1: Sparse Vegetation (Moderate Risk)",
                    "Class 2: Moderate Growth (Low Risk)",
                    "Class 3: Dense Pasture (Minimal Risk)",
                ]

                if status_color == "error":
                    st.error(f"**Status:** {status_tier}")
                elif status_color == "warning":
                    st.warning(f"**Status:** {status_tier}")
                else:
                    st.success(f"**Status:** {status_tier}")

                st.markdown("---")
                st.subheader("Class Probability Distribution")
                for label, prob in zip(class_labels, probs):
                    st.write(f"**{label}:** `{float(prob)*100:.1f}%`")
                    st.progress(float(prob))

            st.markdown("---")
            st.subheader("📋 Recommended Next Steps")
            for step in next_steps:
                st.markdown(f"* {step}")