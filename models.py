import base64
import os

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import JSONResponse
import segmentation_models_pytorch as smp
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
MIN_CONFIDENCE = 0.1
SEG_IMG_SIZE = 256

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


def load_model(model_name: str, model_type: str = "classifier") -> nn.Module:
    model = None

    if model_type == "classifier":
        if model_name == "efficientnet":
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))

        elif model_name == "densenet":
            model = models.densenet121(weights=None)
            model.classifier = nn.Linear(1024, len(CLASS_NAMES))

        elif model_name == "resnet50":
            model = models.resnet50(weights=None)
            model.fc = nn.Sequential(
                nn.Linear(model.fc.in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, len(CLASS_NAMES))
            )

        path = f"models/{model_name}.pth"  # Шлях до класифікаторів

    # 2.Сегментації
    elif model_type == "segmenter":
        if model_name == "unet":
            model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
            path = "models/unet_brain_mri.pth"
        elif model_name == "segformer":
            model = smp.Segformer(encoder_name="mit_b0", encoder_weights=None, in_channels=3, classes=1)
            path = "models/segformer_brain_mri.pth"
        else:
            return None

    if model is None:
        return None

    # Завантаження ваг
    if not os.path.exists(path):
        print(f"❌ Error: Model file not found at {path}")
        return None

    try:
        checkpoint = torch.load(path, map_location=DEVICE)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

        model.load_state_dict(state_dict, strict=True)
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        print(f"❌ Error loading {model_name}: {e}")
        return None


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


def predict_segmentation(image: Image.Image, model_name: str) -> str:
    """
    Повертає зображення у форматі Base64 string.
    """
    if model_name not in MODELS:
        return None

    model = MODELS[model_name]

    # 1. обробка для сегментації (OpenCV Resize + Normalize / 255.0)
    image_np = np.array(image.convert("RGB"))
    original_size = image_np.shape[:2]  # Зберігаємо оригінальний розмір, якщо треба

    # Ресайз до 256x256
    img_resized = cv2.resize(image_np, (SEG_IMG_SIZE, SEG_IMG_SIZE))

    # Нормалізація (H, W, C) -> (C, H, W)
    img_tensor = img_resized.transpose(2, 0, 1).astype('float32') / 255.0
    img_tensor = torch.from_numpy(img_tensor).unsqueeze(0).to(DEVICE)

    # 2. Інференс
    with torch.no_grad():
        logits = model(img_tensor)
        prob = torch.sigmoid(logits).cpu().numpy()[0, 0]

    # 3. Створення маски (Overlay)
    mask = (prob > 0.5).astype(np.uint8)

    result_img = img_resized.copy()
    if np.sum(mask) > 0:
        # Створюємо червону маску
        colored_mask = np.zeros_like(img_resized)
        colored_mask[:, :, 0] = 255  # Red

        # Змішуємо
        blended = cv2.addWeighted(img_resized, 0.6, colored_mask, 0.4, 0)
        result_img[mask == 1] = blended[mask == 1]

    # 4. Конвертація результату в Base64 для JSON
    # Спочатку BGR для OpenCV кодування
    is_success, buffer = cv2.imencode(".jpg", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    if not is_success:
        return None

    return base64.b64encode(buffer).decode('utf-8')

