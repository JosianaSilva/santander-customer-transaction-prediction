import json
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from huggingface import HuggingFaceModelManager

load_dotenv()

def calculate_metrics_from_models() -> dict:
    """Calcula métricas carregando os modelos salvos"""
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    MODEL_PATH = PROJECT_ROOT / "models" / "xgb_model.pkl"
    PCA_PATH = PROJECT_ROOT / "models" / "pca.pkl"
    SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"
    TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test_split.csv"
    
    try:
        df_test = pd.read_csv(TEST_DATA_PATH)
        X_test = df_test.drop('target', axis=1)
        y_test = df_test['target']
        
        scaler = joblib.load(SCALER_PATH)
        pca = joblib.load(PCA_PATH)
        model = joblib.load(MODEL_PATH)
        
        X_test_scaled = scaler.transform(X_test)
        X_test_reduced = pca.transform(X_test_scaled)
        
        predictions = model.predict(X_test_reduced)
        
        accuracy = accuracy_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        metrics = {
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        
        # Salvar métricas
        metrics_file = PROJECT_ROOT / "models" / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Métricas calculadas e salvas em: {metrics_file}")
        return metrics
        
    except Exception as e:
        print(f"Erro ao calcular métricas: {e}")
        return {}

def check_metrics_criteria(metrics_file: str) -> bool:
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        accuracy = metrics.get('accuracy', 0) * 100
        roc_auc = metrics.get('roc_auc', 0) * 100
        
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"ROC AUC: {roc_auc:.2f}%")
        
        accuracy_ok = accuracy > 90.0
        roc_auc_ok = roc_auc > 60.0
        
        print(f"Accuracy > 90%: {'✓' if accuracy_ok else '✗'}")
        print(f"ROC AUC > 60%: {'✓' if roc_auc_ok else '✗'}")
        
        return accuracy_ok and roc_auc_ok
        
    except FileNotFoundError:
        print("Métricas não encontradas. Execute 'make train' primeiro.")
        return False
    except Exception as e:
        print(f"Erro ao ler métricas: {e}")
        return False


def deploy_to_huggingface():
    """Executa o deploy para o Hugging Face"""
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    MODEL_PATH = PROJECT_ROOT / "models" / "xgb_model.pkl"
    PCA_PATH = PROJECT_ROOT / "models" / "pca.pkl"
    SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"
    METRICS_FILE = PROJECT_ROOT / "models" / "metrics.json"
    
    if not MODEL_PATH.exists():
        print("Modelo não encontrado. Execute 'make train' primeiro.")
        return False
        
    if not PCA_PATH.exists():
        print("PCA não encontrado. Execute 'make train' primeiro.")
        return False
    
    if not SCALER_PATH.exists():
        print("Scaler não encontrado. Execute 'make train' primeiro.")
        return False
    
    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_USERNAME = os.getenv("HF_USERNAME")
    HF_REPO_NAME = os.getenv("HF_REPO_NAME", "santander-customer-prediction")
    
    if not HF_TOKEN or not HF_USERNAME:
        print("Configure HF_TOKEN e HF_USERNAME no arquivo .env")
        return False
    
    try:
        with open(METRICS_FILE, 'r') as f:
            metrics = json.load(f)
    except Exception as e:
        print(f"Erro ao carregar métricas: {e}")
        return False
    
    try:
        manager = HuggingFaceModelManager(token=HF_TOKEN)
        repo_name = f"{HF_USERNAME}/{HF_REPO_NAME}"
        
        url = manager.upload_model(
            repo_name=repo_name,
            model_path=str(MODEL_PATH),
            pca_path=str(PCA_PATH),
            scaler_path=str(SCALER_PATH),
            model_name="Santander Customer Transaction Prediction",
            metrics=metrics
        )
        
        print(f"Deploy realizado! URL: {url}")
        return True
        
    except Exception as e:
        print(f"Erro durante upload: {e}")
        return False


def main():
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    METRICS_FILE = PROJECT_ROOT / "models" / "metrics.json"
    
    print("Calculando métricas dos modelos")
    metrics = calculate_metrics_from_models()
    
    if not metrics:
        print("Erro ao calcular métricas. Verifique se os modelos e dados existem.")
        return False
    
    if not check_metrics_criteria(str(METRICS_FILE)):
        print("Métricas não atendem aos critérios:")
        print("- Accuracy > 90%")
        print("- ROC AUC > 60%")
        return False
    
    print("Métricas aprovadas. Fazendo deploy")
    success = deploy_to_huggingface()
    
    if success:
        print("Deploy concluído!")
    else:
        print("Deploy falhou.")
    
    return success


if __name__ == "__main__":
    main()