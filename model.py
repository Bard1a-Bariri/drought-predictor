import torch
import cv2
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.models import ResNet18_Weights, resnet18
class GroundDroughtModel(nn.Module):
    def __init__(self):
        super(GroundDroughtModel, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        for param in self.model.layer4.parameters():
            param.requires_grad = True
            
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.model(x)


class SatelliteDroughtModel(nn.Module):
    def __init__(self, in_channels=10, num_classes=4):
        super(SatelliteDroughtModel, self).__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
def calculate_pred(ready_tensor, model):
    model.eval()
    with torch.no_grad():
        logits = model(ready_tensor)
        
        if logits.shape[-1] == 1:
            score = torch.sigmoid(logits).item()
            return {"type": "ground", "risk_score": score}
        
        else:
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            return {
                "type": "satellite",
                "predicted_class": pred_class,
                "class_probabilities": probs.squeeze().tolist()
            }
def generate_gradcam(ready_tensor, model, raw_image, use_gradcam_plus_plus=True):
    model.eval()
    
    target_layer = model.model.layer4[-1]

    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def forward_hook(module, input, output):
        activations.append(output)
        output.register_hook(save_gradient)

    hook_handle = target_layer.register_forward_hook(forward_hook)

    input_tensor = ready_tensor.clone().detach().requires_grad_(True)
    output = model(input_tensor)
    model.zero_grad()

    if output.shape[-1] == 1:
        target_score = output[0, 0]
    else:
        target_class = torch.argmax(output, dim=1).item()
        target_score = output[0, target_class]

    target_score.backward()
    hook_handle.remove()

    act = activations[0].detach().cpu().numpy()[0]   
    grad = gradients[0].detach().cpu().numpy()[0]  

    if use_gradcam_plus_plus:
        grad_sq = grad ** 2
        grad_cube = grad ** 3
        sum_act = np.sum(act, axis=(1, 2), keepdims=True)
        
        alpha_denom = 2 * grad_sq + sum_act * grad_cube
        alpha_denom = np.where(alpha_denom != 0, alpha_denom, 1e-7)
        alpha = grad_sq / alpha_denom
        
        weights = np.sum(alpha * np.maximum(grad, 0), axis=(1, 2))
    else:
        weights = np.mean(grad, axis=(1, 2))

    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()

    raw_rgb = raw_image.convert("RGB")
    img_array = np.array(raw_rgb)
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