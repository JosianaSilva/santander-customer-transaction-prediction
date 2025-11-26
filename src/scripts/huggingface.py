import joblib
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict
from huggingface_hub import HfApi, create_repo, hf_hub_download
from dotenv import load_dotenv
load_dotenv()


class HuggingFaceModelManager:
    """Gerenciador para modelos no Hugging Face Hub"""
    
    def __init__(self, token: str):
        self.token = token
        self.api = HfApi(token=token)
    
    def _create_readme(self, model_name: str, metrics: Dict = None) -> str:
        """Cria README para o modelo"""
        metrics_text = ""
        if metrics:
            metrics_text = "\n## Metrics\n"
            for k, v in metrics.items():
                if isinstance(v, float):
                    if k in ['accuracy', 'roc_auc']:
                        metrics_text += f"- **{k.replace('_', ' ').title()}**: {v:.4f} ({v*100:.2f}%)\n"
                    else:
                        metrics_text += f"- **{k.replace('_', ' ').title()}**: {v:.4f}\n"
                else:
                    metrics_text += f"- **{k.replace('_', ' ').title()}**: {v}\n"
        
        deployment_info = """
## Deployment Criteria

This model was automatically deployed because it meets the following criteria:
- Accuracy > 90%
- ROC AUC Score > 60%

Only models that meet these performance thresholds are deployed to ensure quality.
"""
        
        return f"""# {model_name}

Modelo para predição de transações de clientes baseado no dataset do Kaggle Santander Customer Transaction Prediction.

Este modelo foi treinado usando Gradient Boosting Classifier com redução de dimensionalidade via PCA.
{metrics_text}{deployment_info}

## Usage
```python
from huggingface_hub import hf_hub_download
import joblib

# Download
model_path = hf_hub_download(repo_id="your-repo", filename="model.pkl")
pca_path = hf_hub_download(repo_id="your-repo", filename="pca.pkl")
scaler_path = hf_hub_download(repo_id="your-repo", filename="scaler.pkl")

# Load
model = joblib.load(model_path)
pca = joblib.load(pca_path)
scaler = joblib.load(scaler_path)

# Predict (assume X is your input data)
X_scaled = scaler.transform(X)
X_transformed = pca.transform(X_scaled)
predictions = model.predict(X_transformed)
probabilities = model.predict_proba(X_transformed)
```

## Model Architecture
- **Algorithm**: Gradient Boosting Classifier
- **Dimensionality Reduction**: PCA (95% variance retained)
- **Preprocessing**: Standard Scaler
- **Features**: 200 anonymous features from Santander dataset

## Training Details
- Train/Test Split: 80/20
- Random State: 42 (for reproducibility)
- Cross-validation: Used during hyperparameter tuning
"""
    
    
    def upload_model(self, repo_name: str, model_path: str, pca_path: str, 
                     scaler_path: str, model_name: str = "Santander Model", metrics: Dict = None):
        """Upload modelo para HF Hub"""
        try:
            create_repo(repo_id=repo_name, token=self.token, exist_ok=True)
            
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                shutil.copy2(model_path, temp_path / "model.pkl")
                shutil.copy2(pca_path, temp_path / "pca.pkl")
                shutil.copy2(scaler_path, temp_path / "scaler.pkl")
                
                readme_content = self._create_readme(model_name, metrics)
                (temp_path / "README.md").write_text(readme_content, encoding='utf-8')
                
                self.api.upload_folder(
                    folder_path=temp_path,
                    repo_id=repo_name,
                    token=self.token
                )
                
            return f"https://huggingface.co/{repo_name}"
        except Exception as e:
            print(f"Erro no upload: {e}")
            raise
    
    def update_model(self, repo_name: str, model_path: str, pca_path: str, scaler_path: str):
        """Atualiza modelo existente"""
        self.api.upload_file(path_or_fileobj=model_path, path_in_repo="model.pkl", 
                           repo_id=repo_name, token=self.token)
        self.api.upload_file(path_or_fileobj=pca_path, path_in_repo="pca.pkl",
                           repo_id=repo_name, token=self.token)
        self.api.upload_file(path_or_fileobj=scaler_path, path_in_repo="scaler.pkl",
                           repo_id=repo_name, token=self.token)


def load_model_from_hf(repo_id: str, token: Optional[str] = None):
    """Carrega modelo do HF Hub"""
    model_path = hf_hub_download(repo_id=repo_id, filename="model.pkl", token=token)
    pca_path = hf_hub_download(repo_id=repo_id, filename="pca.pkl", token=token)
    scaler_path = hf_hub_download(repo_id=repo_id, filename="scaler.pkl", token=token)
    return joblib.load(model_path), joblib.load(pca_path), joblib.load(scaler_path)

def download_model_from_hf(repo_id: str, token: Optional[str] = None):
    """Faz o download dos modelos do HF Hub e salva localmente"""
    model_path = hf_hub_download(repo_id=repo_id, filename="model.pkl", token=token)
    pca_path = hf_hub_download(repo_id=repo_id, filename="pca.pkl", token=token)
    scaler_path = hf_hub_download(repo_id=repo_id, filename="scaler.pkl", token=token)
    
    local_model_path = Path("models") / "xgb_model.pkl"
    local_pca_path = Path("models") / "pca.pkl"
    local_scaler_path = Path("models") / "scaler.pkl"
    
    shutil.copy2(model_path, local_model_path)
    shutil.copy2(pca_path, local_pca_path)
    shutil.copy2(scaler_path, local_scaler_path)
    
    print(f"Modelos baixados para {local_model_path}, {local_pca_path} e {local_scaler_path}")
