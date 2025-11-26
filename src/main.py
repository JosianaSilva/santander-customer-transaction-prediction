from fastapi import FastAPI
import uvicorn
from src.routes.predictions import router as predictions_router

app = FastAPI()

app.include_router(predictions_router)

@app.get("/")
def read_root():
    return {
        "message": "Customer Transaction Prediction API",
        "description": "API para predição de transações de clientes usando modelos de ML.",
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        },
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)