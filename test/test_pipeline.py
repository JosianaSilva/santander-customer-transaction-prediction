import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pytest

sys.path.append(str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent


def test_data_integrity():
    raw_train_path = PROJECT_ROOT / "data" / "raw" / "train.csv"
    assert raw_train_path.exists(), "Training data file not found"
    
    df_train = pd.read_csv(raw_train_path)
    
    required_cols = ['ID_code', 'target']
    missing_cols = [col for col in required_cols if col not in df_train.columns]
    assert not missing_cols, f"Missing required columns: {missing_cols}"
    
    target_dist = df_train['target'].value_counts()
    assert len(target_dist) == 2, "Target variable should be binary"
    
    feature_cols = [col for col in df_train.columns if col not in ['ID_code', 'target']]
    non_numeric = df_train[feature_cols].select_dtypes(exclude=[np.number]).columns
    assert len(non_numeric) == 0, f"Non-numeric feature columns found: {list(non_numeric)}"


def test_model_files_exist():
    models_path = PROJECT_ROOT / "models"
    
    required_models = ['scaler.pkl', 'pca.pkl', 'xgb_model.pkl']
    for model_file in required_models:
        assert (models_path / model_file).exists(), f"Model file {model_file} not found"


def test_model_prediction():
    from src.scripts.predict import predict_single
    
    models_path = PROJECT_ROOT / "models"
    test_data = np.random.randn(200)
    
    prediction = predict_single(test_data, str(models_path))
    
    required_keys = ['prediction', 'probability_class_0', 'probability_class_1']
    assert all(key in prediction for key in required_keys), "Missing prediction keys"
    
    assert isinstance(prediction['prediction'], int), "Prediction should be integer"
    assert prediction['prediction'] in [0, 1], "Prediction should be 0 or 1"
    
    prob_sum = prediction['probability_class_0'] + prediction['probability_class_1']
    assert abs(prob_sum - 1.0) < 0.01, "Probabilities should sum to 1"


def test_training_metrics():
    metrics_path = PROJECT_ROOT / "models" / "metrics.json"
    assert metrics_path.exists(), "Metrics file not found"
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    required_metrics = ['accuracy', 'roc_auc', 'precision', 'recall', 'f1_score']
    missing_metrics = [m for m in required_metrics if m not in metrics]
    assert not missing_metrics, f"Missing metrics: {missing_metrics}"
    
    for metric_name, value in metrics.items():
        assert isinstance(value, (int, float)), f"Metric {metric_name} should be numeric"
        assert 0 <= value <= 1, f"Metric {metric_name} should be between 0 and 1"
    
    assert metrics['roc_auc'] >= 0.5, "ROC AUC should be better than random"


def test_prediction_consistency():
    from src.scripts.predict import predict_single
    
    models_path = PROJECT_ROOT / "models"
    test_data = np.random.randn(200)
    
    predictions = []
    for _ in range(3):
        pred = predict_single(test_data.copy(), str(models_path))
        predictions.append(pred)
    
    first_pred = predictions[0]
    for pred in predictions[1:]:
        assert pred['prediction'] == first_pred['prediction'], "Predictions should be deterministic"
        
        for key in ['probability_class_0', 'probability_class_1']:
            assert abs(pred[key] - first_pred[key]) < 1e-10, "Probabilities should be deterministic"


def test_processed_data():
    processed_path = PROJECT_ROOT / "data" / "processed"
    train_path = processed_path / "train_split.csv"
    test_path = processed_path / "test_split.csv"
    
    assert train_path.exists(), "Processed training data not found"
    assert test_path.exists(), "Processed test data not found"
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    assert list(df_train.columns) == list(df_test.columns), "Train and test columns should match"
    assert 'target' in df_train.columns, "Target column missing from training data"
    
    train_missing = df_train.isnull().sum().sum()
    test_missing = df_test.isnull().sum().sum()
    
    assert train_missing == 0, "Training data should not have missing values"
    assert test_missing == 0, "Test data should not have missing values"