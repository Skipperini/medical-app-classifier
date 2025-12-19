from fastapi import FastAPI
from models import MODELS, load_model
import os
from controllers.classify_controller import router as classify_router
from response import error_response

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import warnings
warnings.filterwarnings("ignore", message="Found libiomp5md.dll")
app = FastAPI(
    title="Brain Tumor Classification API",
    description="API for classifying brain tumors using EfficientNet-B0, DenseNet121, ResNet50",
    version="1.0"
)
app.include_router(classify_router)

model_names = ["efficientnet", "densenet", "resnet50"]
segmenters = ["unet", "segformer"]

@app.on_event("startup")
async def startup_event():
    """Завантаження всіх моделей при старті програми"""
    try:
        for name in model_names:
            MODELS[name] = load_model(name)

        for name in segmenters:
            model = load_model(name, model_type="segmenter")
            if model:
                MODELS[name] = model
                print(f"✅ Segmenter loaded: {name}")
            else:
                print(f"⚠️ Failed to load: {name}")

        print("✅ Усі моделі успішно завантажені!")
    except Exception as e:
        return error_response(
            f"Models not loaded: {str(e)}",
            code=500
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)