from fastapi import APIRouter
from models import *

router = APIRouter(prefix="/classify")

@router.post("/predict")
async def predict(
        request: Request,
        file: UploadFile = File(...),
        model_name: str = Form(...)
):

    """
    Класифікує MRI знімок мозку.

    Параметри:
    - file: Зображення у форматі JPEG/PNG
    - model_name: Модель для використання (efficientnet_b0, densenet121, resnet50)

    Повертає:
    - predicted_class: Передбачений клас
    - confidence: Впевненість моделі
    - probabilities: Можливості для всіх класів
    """
    try:
        # Перевірка моделі
        if model_name not in MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} not available. Choose from {list(MODELS.keys())}"

            )

        # Читання та перевірка зображення
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        if not validate_mri_image(image):
            raise HTTPException(
                status_code=400,
                detail="Invalid image: Doesn't look like a brain MRI scan"
            )

        # Передбачення
        result = predict_image(image, model_name)

        # Перевірка confidence
        if result["confidence"] < MIN_CONFIDENCE:
            return JSONResponse(
                status_code=200,
                content={
                    "error": f"Low model confidence ({result['confidence']:.2f}). Image might be invalid or unclear.",
                    "suggestion": "Please upload a clearer brain MRI scan.",
                    **result
                }
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
