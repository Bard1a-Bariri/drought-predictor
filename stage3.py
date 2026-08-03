import torch
import torch.nn as nn
import torch.nn.functional as f
from stage2.py import transformed_tensor
import streamlit as st
class DroughtModel(nn.Module):
    def __init__(self):
        super(DroughtModel(), self).__init__
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channel=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
        model = DroughtModel()
        model.eval()
        with torch.no_grad():
            prediction = model(transformed_tensor)
        return prediction