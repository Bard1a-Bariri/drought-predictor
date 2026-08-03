from PIL import Image
from torchvisions import transforms
import numpy as np
import streamlit as st
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def preprocessimg(uploaded_file):
    show = Image.open(uploaded_file).convert("RGB")
    st.image(show, caption="Original Upload", width=300)
    transformed_tensor = transform(show)
    return transformed_tensor