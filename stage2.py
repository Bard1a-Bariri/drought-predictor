from PIL import Image
from torchvision import transforms
import numpy as np
import streamlit as st
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def preprocessimg(uploaded_file):
    raw_image = Image.open(uploaded_file).convert("RGB")
    
    transformed_tensor = transform(raw_image)
    ready_tensor = transformed_tensor.unsqueeze(0)
    
    return ready_tensor, raw_image