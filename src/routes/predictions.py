from fastapi import APIRouter, HTTPException, UploadFile, File
import pandas as pd
import io
from src.models.predict import predict_single, load_models

from pydantic import BaseModel
from typing import List

class TransactionData(BaseModel):
    features: List[float]

class BatchTransactionData(BaseModel):
    transactions: List[List[float]]

router = APIRouter()

@router.post("/predict")
def predict_transaction(data: TransactionData):
    """
    Prediz se uma transação será realizada com base nas features fornecidas.
    
    Args:
        data: Dados da transação com 200 features
        
    Returns:
        dict: Predição e probabilidades
    """
    try:
        if len(data.features) != 200:
            raise HTTPException(
                status_code=400, 
                detail=f"Esperado 200 features, recebido {len(data.features)}"
            )
        
        result = predict_single(data.features, models_path="models")
        
        return {
            "success": True,
            "prediction": result["prediction"],
            "probability_no_transaction": result["probability_class_0"],
            "probability_transaction": result["probability_class_1"],
            "confidence": max(result["probability_class_0"], result["probability_class_1"])
        }
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=503, 
            detail="Modelos não encontrados. Verifique se os modelos foram treinados."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@router.get("/models/status")
def check_models_status():
    """
    Verifica se os modelos estão disponíveis.
    
    Returns:
        dict: Status dos modelos
    """
    try:
        load_models("models")
        return {
            "models_loaded": True,
            "status": "ready_for_predictions"
        }
    except FileNotFoundError:
        return {
            "models_loaded": False,
            "status": "models_not_found",
            "message": "Execute o treinamento primeiro"
        }