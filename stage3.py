import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18
class DroughtModelResNet(nn.Module):
    def __init__(self):
        super(DroughtModelResNet, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)
def calculate_pred(ready_tensor, model):
    model.eval()
    with torch.no_grad():
        prediction = model(ready_tensor)
    return prediction.item()
def generate_gradcam(ready_tensor, model, raw_image):
    """Generates an authentic activation heatmap highlighting pixels driving the model decision."""
    model.eval()

    # Target the last convolutional layer of ResNet18
    target_layer = model.model.layer4[-1]

    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    hook_handle = target_layer.register_forward_hook(forward_hook)

    ready_tensor.requires_grad = True
    output = model(ready_tensor)
    model.zero_grad()

    output.backward()
    hook_handle.remove()

    act = activations[0].detach().cpu().numpy()[0]
    grad = gradients[0].detach().cpu().numpy()[0]

    weights = np.mean(grad, axis=(1, 2))

    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    img_array = np.array(raw_image)
    h, w, _ = img_array.shape
    cam_resized = cv2.resize(cam, (w, h))

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(img_array, 0.6, heatmap_colored, 0.4, 0)

    return Image.fromarray(blended)


def generate_prescriptive_drills(prediction_score):
    risk_percentage = prediction_score * 100
    
    if risk_percentage >= 70.0:
        tier = "CRITICAL THREAT LEVEL"
        drill_steps = [
            "Activate target sub-surface drip-irrigation networks immediately.",
            "Deploy liquid potassium foliar sprays to bolster plant cell wall pressure.",
            "Schedule automated sensor checks to repeat at 6-hour intervals."
        ]
    elif risk_percentage >= 40.0:
        tier = "MODERATE WATCH LEVEL"
        drill_steps = [
            "Increase ground soil-moisture sensor logging to daily intervals.",
            "Inspect field grid sections manually for early microscopic stem sagging.",
            "Prepare secondary irrigation lines in case moisture indexes drop further."
        ]
    else:
        tier = "STABLE BASELINE STATUS"
        drill_steps = [
            "Maintain baseline weekly irrigation configurations.",
            "No microscopic cellular stress anomalies flagged by the computer vision eye."
        ]
        
    return tier, drill_steps