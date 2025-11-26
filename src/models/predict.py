import pandas as pd
import numpy as np
import joblib
from typing import Union, List
import os

def load_models(models_path: str = "models"):
    """
    Carrega os modelos treinados (PCA e classificador).
    
    Args:
        models_path (str): Caminho para o diretório contendo os modelos
        
    Returns:
        tuple: (pca_model, classifier_model)
    """
    try:
        pca_model = joblib.load(os.path.join(models_path, 'pca_model.pkl'))
        classifier_model = joblib.load(os.path.join(models_path, 'gbc_model.pkl'))
        return pca_model, classifier_model
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Modelo não encontrado: {e}")

def preprocess_data(data: Union[pd.DataFrame, pd.Series, List, np.ndarray], 
                   pca_model) -> np.ndarray:
    """
    Preprocessa os dados aplicando as mesmas transformações do treinamento.
    
    Args:
        data: Dados a serem preprocessados (pode ser DataFrame, Series, lista ou array)
        pca_model: Modelo PCA treinado
        
    Returns:
        np.ndarray: Dados transformados pelo PCA
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

    data_transformed = pca_model.transform(data)
    
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
    pca_model, classifier_model = load_models(models_path)
    
    processed_data = preprocess_data(instance, pca_model)
    
    prediction = classifier_model.predict(processed_data)[0]
    prediction_proba = classifier_model.predict_proba(processed_data)[0]
    
    return {
        'prediction': int(prediction),
        'probability_class_0': float(prediction_proba[0]),
        'probability_class_1': float(prediction_proba[1])
    }

