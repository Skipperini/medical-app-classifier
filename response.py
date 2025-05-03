from fastapi import HTTPException
from fastapi.responses import JSONResponse

def success_response(message = '', data = None, code = 200):
    return JSONResponse(
        status_code=code,
        content={
            "success": True,
            "message": message,
            "data": data,
        }
    )

def error_response(message = '', errors = None, code = 400):
    return JSONResponse(
        status_code=code,
        content={
            "success": False,
            "message": message,
            "errors": errors,
        }
    )