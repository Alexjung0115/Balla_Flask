import cv2
import torch
import gc
from PIL import Image
from ultralytics import YOLO
import numpy as np
import pandas as pd
import torch.nn as nn
import uuid
import torchvision.transforms as transforms
import torchvision.models as models

class WrinkleRegressionEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.efficientnet_b0(pretrained=True)
        in_features = self.base.classifier[1].in_features
        self.base.classifier[1] = nn.Linear(in_features, 1)
        self.act = nn.Sigmoid()  # [0, 1] → 이후 * 6으로 0~6 범위 조절
        print(self.base.classifier)
    def forward(self, x):
        x = self.base(x)
        x = self.act(x) * 6.0  # 등급 범위에 맞게 조절
        return x
class PigmentationRegressionEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.efficientnet_b0(pretrained=True)
        in_features = self.base.classifier[1].in_features
        self.base.classifier[1] = nn.Linear(in_features, 1)
        self.act = nn.Sigmoid()  # [0, 1] → 이후 * 6으로 0~6 범위 조절
        print(self.base.classifier)
    def forward(self, x):
        x = self.base(x)
        x = self.act(x) * 5.0  # 등급 범위에 맞게 조절
        return x
class PoreRegressionEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.efficientnet_b0(pretrained=True)
        in_features = self.base.classifier[1].in_features
        self.base.classifier[1] = nn.Linear(in_features, 1)
        self.act = nn.Sigmoid()  # [0, 1] → 이후 * 6으로 0~6 범위 조절
        print(self.base.classifier)
    def forward(self, x):
        x = self.base(x)
        x = self.act(x) * 5.0  # 등급 범위에 맞게 조절
        return x
class DrynessRegressionEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.efficientnet_b0(pretrained=True)
        in_features = self.base.classifier[1].in_features
        self.base.classifier[1] = nn.Linear(in_features, 1)
        self.act = nn.Sigmoid()  # [0, 1] → 이후 * 6으로 0~6 범위 조절
        print(self.base.classifier)
    def forward(self, x):
        x = self.base(x)
        x = self.act(x) * 4.0  # 등급 범위에 맞게 조절
        return x
class SaggingRegressionEfficientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = models.efficientnet_b0(pretrained=True)
        in_features = self.base.classifier[1].in_features
        self.base.classifier[1] = nn.Linear(in_features, 1)
        self.act = nn.Sigmoid()  # [0, 1] → 이후 * 6으로 0~6 범위 조절
        print(self.base.classifier)
    def forward(self, x):
        x = self.base(x)
        x = self.act(x) * 6.0  # 등급 범위에 맞게 조절
        return x

class RegressionEfficientNet(nn.Module):
    def __init__(self,label):
        super().__init__()
        self.base = models.efficientnet_b0(pretrained=True)
        in_features = self.base.classifier[1].in_features
        self.base.classifier[1] = nn.Linear(in_features, 1)
        self.act = nn.Sigmoid() 
        self.label = label
        print(self.base.classifier)
    def forward(self, x):
        x = self.base(x)
        if self.label =='Wrinkle':
            x = self.act(x) * 6.0  # 등급 범위에 맞게 조절
        elif self.label =='Pore':
            x = self.act(x) * 5.0  # 등급 범위에 맞게 조절
        elif self.label =='Dry':
            x = self.act(x) * 4.0  # 등급 범위에 맞게 조절
        elif self.label =='Sagging':
            x = self.act(x) * 6.0  # 등급 범위에 맞게 조절
        elif self.label =='Pigmentation':
            x = self.act(x) * 7.0  # 등급 범위에 맞게 조절
        return x
def load_all_models():
    # YOLO 모델
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model1 = YOLO('best_weight_model/newfaceeye.pt').to(device)
    model3 = YOLO("best_weight_model/face_segmentation.pt").to(device)

    wrinkle_model = RegressionEfficientNet('Wrinkle').to(device)
    wrinkle_model.load_state_dict(torch.load("best_weight_model/best_wrinkle_model.pth", map_location=device))
    wrinkle_model.eval()
    Pore_model = RegressionEfficientNet('Pore').to(device)
    Pore_model.load_state_dict(torch.load("best_weight_model/Pore_best_model_2nd.pth", map_location=device))
    Pore_model.eval()
    Pig_model = RegressionEfficientNet('Pigmentation').to(device)
    Pig_model.load_state_dict(torch.load("best_weight_model/Pigmentation_best_model_2nd.pth", map_location=device))
    Pig_model.eval()
    Sagging_model = RegressionEfficientNet('Sagging').to(device)
    Sagging_model.load_state_dict(torch.load("best_weight_model/Sagging_best_model_2nd.pth", map_location=device))
    Sagging_model.eval()
    Dry_model = RegressionEfficientNet('Dry').to(device)
    Dry_model.load_state_dict(torch.load("best_weight_model/Dry_best_model_2nd.pth", map_location=device))
    Dry_model.eval()

    return model1,model3,wrinkle_model,Pore_model,Pig_model,Sagging_model,Dry_model

