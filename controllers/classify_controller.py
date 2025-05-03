from fastapi import APIRouter
from models import *
from response import *
from validation.image import validate_image
import io

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
        if not file.size:
            return error_response(
                "Image not uploaded",
                code=422
            )

        # Перевірка моделі
        if model_name not in MODELS:
            return error_response(
                f"Model {model_name} not available. Choose from {list(MODELS.keys())}",
                code=404
            )
        file = await file.read()
        image = Image.open(io.BytesIO(file))

        # Читання та перевірка зображення
        if not await validate_image(file):
            return error_response(
                "Invalid image: Doesn't look like a brain MRI scan",
                code=409
            )

        # Передбачення
        result = predict_image(image, model_name)

        # Перевірка confidence
        if result["confidence"] < MIN_CONFIDENCE:
            return error_response(
                f"Low model confidence ({result['confidence']:.2f}). Image might be invalid or unclear.",
                code=409
            )
        return success_response(
            data=result
        )

    except HTTPException:
        raise
    except Exception as e:
        return error_response(
            f"Internal server error: {str(e)}",
            code=500
        )
