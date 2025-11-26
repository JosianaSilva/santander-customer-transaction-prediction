import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json
from dotenv import load_dotenv

load_dotenv()

from pathlib import Path

path = Path(__file__).parent.parent.parent
df = pd.read_csv(f'{path}/data/raw/train.csv')

df.head()

df.shape

df = df.drop('ID_code', axis=1)

all(df.isnull().sum())

X, y = df.drop('target', axis=1), df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

df_train = pd.concat([X_train, y_train], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)

processed_data_path = f'{path}/data/processed'
os.makedirs(processed_data_path, exist_ok=True)

df_train.to_csv(f'{processed_data_path}/train_split.csv', index=False)
df_test.to_csv(f'{processed_data_path}/test_split.csv', index=False)

print(f'train_split.csv saved to {processed_data_path}/train_split.csv')
print(f'test_split.csv saved to {processed_data_path}/test_split.csv')

pca = PCA(n_components=0.95)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

X_reduced = pca.fit_transform(X_scaled)

X_reduced.shape

componentes = pca.components_
componentes.shape

os.makedirs(f'{path}/models', exist_ok=True)

joblib.dump(scaler, f'{path}/models/scaler.pkl')
joblib.dump(pca, f'{path}/models/pca.pkl')

device = os.getenv('DEVICE', 'cpu')

clf = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=20,
    device=device,
    tree_method='hist',
    random_state=0
).fit(X_reduced, y_train)

X_test_scaled = scaler.transform(X_test)
X_test_reduced = pca.transform(X_test_scaled)
predictions = clf.predict(X_test_reduced)


matrix = confusion_matrix(y_test, predictions)
report = classification_report(y_test, predictions)
print('Confusion Matrix:')
print(matrix)
print('\nClassification Report:')
print(report)

accuracy = accuracy_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f'Accuracy: {accuracy:.4f}')
print(f'ROC AUC Score: {roc_auc:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

metrics = {
    'accuracy': float(accuracy),
    'roc_auc': float(roc_auc),
    'precision': float(precision),
    'recall': float(recall),
    'f1_score': float(f1)
}

metrics_path = f'{path}/models/metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f'Métricas salvas em: {metrics_path}')

joblib.dump(clf, f'{path}/models/xgb_model.pkl')
print(f'Modelo salvo em: {path}/models/xgb_model.pkl')
print(f'PCA salvo em: {path}/models/pca.pkl')
print(f'Scaler salvo em: {path}/models/scaler.pkl')