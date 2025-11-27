import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.main import app

client = TestClient(app)

def test_health_check():
    """Testa o endpoint de health check"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_root_endpoint():
    """Testa o endpoint raiz"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data
    assert "health" in data["endpoints"]

def test_docs_endpoint():
    """Testa se a documentação está disponível"""
    response = client.get("/docs")
    assert response.status_code == 200

@pytest.mark.integration
def test_predict_endpoint_structure():
    """Testa a estrutura do endpoint de predição (sem fazer predição real)"""
    response = client.get("/predict")
    assert response.status_code == 405  # Method Not Allowed
    
@pytest.mark.integration  
def test_models_status_endpoint():
    """Testa o endpoint de status dos modelos"""
    response = client.get("/models/status")
    assert response.status_code in [200, 404]  # 404 se modelos não existem ainda