from fastapi import APIRouter, HTTPException, UploadFile, File
import pandas as pd
import io
from src.scripts.predict import predict_single, predict_batch, load_models

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
    
@router.post("/predict/batch")
def predict_transactions_batch(data: BatchTransactionData):
    """
    Prediz múltiplas transações em lote.
    
    Args:
        data: Lista de transações, cada uma com 200 features
        
    Returns:
        dict: Lista de predições e probabilidades
    """
    try:
        for i, transaction in enumerate(data.transactions):
            if len(transaction) != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Transação {i}: esperado 200 features, recebido {len(transaction)}"
                )
        
        df = pd.DataFrame(data.transactions)
        
        results = predict_batch(df, models_path="models")
        
        predictions = []
        for _, row in results.iterrows():
            predictions.append({
                "prediction": int(row["prediction"]),
                "probability_no_transaction": float(row["probability_class_0"]),
                "probability_transaction": float(row["probability_class_1"]),
                "confidence": float(max(row["probability_class_0"], row["probability_class_1"]))
            })
        
        return {
            "success": True,
            "total_predictions": len(predictions),
            "predictions": predictions
        }
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=503, 
            detail="Modelos não encontrados. Verifique se os modelos foram treinados."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@router.post("/predict/csv")
async def predict_from_csv_file(file: UploadFile = File(...)):
    """
    Prediz transações a partir de um arquivo CSV.
    
    Args:
        file: Arquivo CSV com as features
        
    Returns:
        dict: Predições em formato JSON
    """
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Arquivo deve ser CSV")
        
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        results = predict_batch(df, models_path="models")
        
        predictions = results.to_dict('records')
        
        return {
            "success": True,
            "filename": file.filename,
            "total_predictions": len(predictions),
            "predictions": predictions
        }
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=503, 
            detail="Modelos não encontrados. Verifique se os modelos foram treinados."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar arquivo: {str(e)}")

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