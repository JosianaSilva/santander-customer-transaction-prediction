import pandas as pd
import numpy as np
import joblib
from typing import Union, List
import os
import xgboost as xgb
from dotenv import load_dotenv

load_dotenv()

def load_models(models_path: str = "models"):
    """
    Carrega os modelos treinados (StandardScaler, PCA e classificador XGBoost).
    
    Args:
        models_path (str): Caminho para o diretório contendo os modelos
        
    Returns:
        tuple: (scaler, pca, classifier_model)
    """
    try:
        scaler = joblib.load(os.path.join(models_path, 'scaler.pkl'))
        pca = joblib.load(os.path.join(models_path, 'pca.pkl'))
        classifier_model = joblib.load(os.path.join(models_path, 'xgb_model.pkl'))
        
        device = os.getenv('DEVICE', 'cpu')
        if hasattr(classifier_model, 'set_params'):
            classifier_model.set_params(device=device)
        
        return scaler, pca, classifier_model
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Modelo não encontrado: {e}")

def preprocess_data(data: Union[pd.DataFrame, pd.Series, List, np.ndarray], 
                   scaler, pca) -> np.ndarray:
    """
    Preprocessa os dados aplicando as mesmas transformações do treinamento.
    
    Args:
        data: Dados a serem preprocessados (pode ser DataFrame, Series, lista ou array)
        scaler: StandardScaler treinado
        pca: Modelo PCA treinado
        
    Returns:
        np.ndarray: Dados transformados pelo StandardScaler e PCA
    """
    if isinstance(data, (list, np.ndarray)):
        if isinstance(data, list):
            data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        data = pd.DataFrame(data)
    elif isinstance(data, pd.Series):
        data = data.to_frame().T
    
    if 'ID_code' in data.columns:
        data = data.drop('ID_code', axis=1)

    data_scaled = scaler.transform(data)
    data_transformed = pca.transform(data_scaled)
    
    return data_transformed

def predict_single(instance: Union[pd.Series, List, np.ndarray], 
                  models_path: str = "models") -> dict:
    """
    Faz predição para uma única instância.
    
    Args:
        instance: Uma única instância para predição (pode ser Series, lista ou array)
        models_path: Caminho para os modelos
        
    Returns:
        dict: Dicionário com predição e probabilidade
    """
    scaler, pca, classifier_model = load_models(models_path)
    
    processed_data = preprocess_data(instance, scaler, pca)
    
    prediction = classifier_model.predict(processed_data)[0]
    prediction_proba = classifier_model.predict_proba(processed_data)[0]
    
    return {
        'prediction': int(prediction),
        'probability_class_0': float(prediction_proba[0]),
        'probability_class_1': float(prediction_proba[1])
    }

def predict_batch(df: pd.DataFrame, 
                 models_path: str = "models") -> pd.DataFrame:
    """
    Faz predições em lote para um DataFrame.
    
    Args:
        df: DataFrame com as instâncias para predição
        models_path: Caminho para os modelos
        
    Returns:
        pd.DataFrame: DataFrame com predições e probabilidades
    """
    scaler, pca, classifier_model = load_models(models_path)
    
    id_codes = None
    if 'ID_code' in df.columns:
        id_codes = df['ID_code'].copy()
    
    processed_data = preprocess_data(df, scaler, pca)
    
    predictions = classifier_model.predict(processed_data)
    predictions_proba = classifier_model.predict_proba(processed_data)
    
    result = pd.DataFrame({
        'prediction': predictions.astype(int),
        'probability_class_0': predictions_proba[:, 0],
        'probability_class_1': predictions_proba[:, 1]
    })
    
    if id_codes is not None:
        result.insert(0, 'ID_code', id_codes.values)
    
    return result
