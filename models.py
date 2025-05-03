import os
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import JSONResponse
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Dict, Any, Optional
import torchvision.models as models
from torch import nn

from response import *

# Конфігурація
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
MIN_CONFIDENCE = 0.5

# Аугментації для валідації
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# Глобальний словник для зберігання моделей
MODELS: Dict[str, Any] = {}


def build_resnet50() -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, len(CLASS_NAMES))
    )
    return model


def load_model(model_name: str) -> nn.Module:
    """Завантажує модель з урахуванням її архітектури"""
    if model_name == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    elif model_name == "densenet":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(1024, len(CLASS_NAMES))
    elif model_name == "resnet50":
        model = build_resnet50()
    else:
        return error_response(
            f"Unsupported model: {model_name}",
            code=409
        )

    model_path = f"models/{model_name}.pth"
    if not os.path.exists(model_path):
        return error_response(
            f"Model weights not found at {model_path}",
            code=404
        )

    checkpoint = torch.load(model_path, map_location=DEVICE)

    # Завантаження ваг
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Для ResNet потрібно ключі
    if model_name == "resnet50":
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('fc.'):
                new_state_dict[k] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model


def predict_image(image: Image.Image, model_name: str) -> Dict[str, Any]:
    """Виконання прогноз на зображенні"""
    if model_name not in MODELS:
        return error_response(
            f"Model {model_name} not loaded",
            code=404
        )

    model = MODELS.get(model_name)

    # Перетворення зображення
    image_np = np.array(image.convert("RGB"))
    augmented = val_transform(image=image_np)
    image_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

    # Передбачення
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_prob, pred_class = torch.max(probs, 1)

    return {
        "predicted_class": CLASS_NAMES[pred_class.item()],
        "confidence": round(pred_prob.item(), 2),
        "probabilities": {cls: round(prob.item(), 2) for cls, prob in zip(CLASS_NAMES, probs[0])}
    }


