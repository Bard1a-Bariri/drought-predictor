import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import gdown
from weasyprint import HTML

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
    page_title="TerraSight AI Platform",
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
        st.sidebar.success("Loaded Satellite Model")
    except Exception as e:
        st.sidebar.warning("Could not load satellite_droughtwatch.pth (using unweighted model)")
    model.to(DEVICE)
    model.eval()
    return model

satellite_model = load_satellite_model()

def generate_pdf_report(region_name, risk_tier, risk_percentage, steps):
    steps_html = "".join([f"<li>{step}</li>" for step in steps])
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{ font-family: 'Helvetica', sans-serif; color: #1e293b; padding: 20px; }}
        .header {{ background: #0f172a; color: white; padding: 15px; border-radius: 6px; }}
        .header h1 {{ margin: 0; color: #38bdf8; font-size: 18pt; }}
        .card {{ border: 1px solid #e2e8f0; padding: 15px; margin-top: 15px; border-radius: 6px; }}
        .badge {{ background-color: #fef2f2; color: #dc2626; padding: 4px 8px; font-weight: bold; border-radius: 4px; }}
    </style>
    </head>
    <body>
        <div class="header">
            <h1>TERRASIGHT DIAGNOSTIC REPORT</h1>
            <p>Automated Environmental Assessment & Decision Support</p>
        </div>
        <div class="card">
            <h3>Assessment Parameters</h3>
            <p><strong>Target Region:</strong> {region_name}</p>
            <p><strong>Status Tier:</strong> <span class="badge">{risk_tier}</span></p>
            <p><strong>Calculated Risk Score:</strong> {risk_percentage:.1f}%</p>
        </div>
        <div class="card">
            <h3>Actionable Next Steps</h3>
            <ul>{steps_html}</ul>
        </div>
    </body>
    </html>
    """
    return HTML(string=html_template).write_pdf()

st.title("🌾 TerraSight AI Platform")
st.markdown("---")

tab1, tab2 = st.tabs(["🌿 Ground Assessment", "🛰️ Satellite Assessment"])

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

        if st.button("Run Analysis", type="primary"):
            with st.spinner("Analyzing image & preparing heatmap..."):
                result = calculate_pred(input_tensor, ground_model)
                risk_score = result["risk_score"]
                tier, drills = generate_prescriptive_drills(risk_score)
                gradcam_img = generate_gradcam(input_tensor, ground_model, raw_img)

                st.session_state["ground_results"] = {
                    "risk_score": risk_score,
                    "tier": tier,
                    "drills": drills,
                    "gradcam_img": gradcam_img
                }

        if "ground_results" in st.session_state:
            res = st.session_state["ground_results"]
            risk_score = res["risk_score"]
            tier = res["tier"]
            drills = res["drills"]

            with col2:
                st.subheader("Corresponding Heatmap")
                st.image(res["gradcam_img"], use_container_width=True)

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

            pdf_bytes = generate_pdf_report("Ground Leaf Scan", tier, risk_score * 100, drills)
            st.download_button(
                label="📄 Export PDF Report",
                data=pdf_bytes,
                file_name="ground_drought_report.pdf",
                mime="application/pdf"
            )

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

        # Synthetic 10-band spectral expansion
        R = rgb_tensor[0:1, :, :]
        G = rgb_tensor[1:2, :, :]
        B = rgb_tensor[2:3, :, :]

        NIR = G * 2.0        
        SWIR1 = G * 0.5      
        SWIR2 = R * 0.3      

        ten_band_tensor = torch.cat([B, B, G, R, NIR, SWIR1, SWIR2, G, B, R], dim=0)
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
                        "Emergency Livestock Relocation: Initiate pasture transfer immediately.",
                        "Water Management: Enforce immediate water rationing in high-risk zones.",
                        "High-Frequency Monitoring: Schedule daily satellite spectral re-scans."
                    ]
                elif drought_percentage >= 30:
                    status_tier = "MODERATE DROUGHT WARNING"
                    status_color = "warning"
                    next_steps = [
                        "Rotational Grazing: Reduce grazing density on sparse vegetation patches.",
                        "Irrigation Efficiency: Audit and adjust drip/sprinkler systems.",
                        "Soil Moisture Audits: Perform ground-level soil testing in vulnerable sections."
                    ]
                else:
                    status_tier = "HEALTHY / MINIMAL DROUGHT RISK"
                    status_color = "success"
                    next_steps = [
                        "Maintain Standard Rotation: Forage capacity is sufficient for herd density.",
                        "Soil Health Monitoring: Keep standard seasonal monitoring schedule.",
                        "Rainwater Capture: Prepare infrastructure for upcoming dry cycles."
                    ]

                st.session_state["sat_results"] = {
                    "drought_percentage": drought_percentage,
                    "drought_risk_score": drought_risk_score,
                    "status_tier": status_tier,
                    "status_color": status_color,
                    "probs": probs,
                    "next_steps": next_steps
                }

        if "sat_results" in st.session_state:
            s_res = st.session_state["sat_results"]
            
            with col2:
                st.subheader("Model Diagnostics")
                st.metric(label="Calculated Drought Risk Index", value=f"{s_res['drought_percentage']:.1f}%")
                st.progress(float(s_res['drought_risk_score']))

                class_labels = [
                    "Class 0: Barren / Desert (High Risk)",
                    "Class 1: Sparse Vegetation (Moderate Risk)",
                    "Class 2: Moderate Growth (Low Risk)",
                    "Class 3: Dense Pasture (Minimal Risk)",
                ]

                if s_res['status_color'] == "error":
                    st.error(f"**Status:** {s_res['status_tier']}")
                elif s_res['status_color'] == "warning":
                    st.warning(f"**Status:** {s_res['status_tier']}")
                else:
                    st.success(f"**Status:** {s_res['status_tier']}")

                st.markdown("---")
                st.subheader("Class Probability Distribution")
                for label, prob in zip(class_labels, s_res['probs']):
                    st.write(f"**{label}:** `{float(prob)*100:.1f}%`")
                    st.progress(float(prob))

            st.markdown("---")
            st.subheader("📋 Recommended Next Steps")
            for step in s_res['next_steps']:
                st.markdown(f"* {step}")

            sat_pdf_bytes = generate_pdf_report(
                "Satellite Tile Assessment", 
                s_res['status_tier'], 
                s_res['drought_percentage'], 
                s_res['next_steps']
            )
            st.download_button(
                label="📄 Export Satellite Assessment PDF",
                data=sat_pdf_bytes,
                file_name="satellite_drought_report.pdf",
                mime="application/pdf"
            )