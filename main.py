from fastapi import FastAPI
from models import MODELS, load_model
import os
from controllers.classify_controller import router as classify_router
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Временное решение

import warnings
warnings.filterwarnings("ignore", message="Found libiomp5md.dll")
app = FastAPI(
    title="Brain Tumor Classification API",
    description="API for classifying brain tumors using EfficientNet-B0, DenseNet121, ResNet50",
    version="1.0"
)
app.include_router(classify_router)

@app.on_event("startup")
async def startup_event():
    """Завантаження всіх моделей при старті програми"""
    try:
        model_names = ["efficientnet_b0", "densenet121", "resnet50"]
        for name in model_names:
            MODELS[name] = load_model(name)
        print("✅ Усі моделі успішно завантажені!")
    except Exception as e:
        print(f"❌ Помилка завантаження моделей: {str(e)}")
        raise

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)