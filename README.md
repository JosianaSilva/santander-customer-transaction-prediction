# Santander Customer Transaction Prediction

API para predição de transações de clientes usando modelos de Machine Learning.

## 🔧 Pré-requisitos
- Python 3.12+
- Git
- Docker (opcional)
- Make 3.81

## 🚀 Começando

### 1. Clonar o repositório
```bash
git clone https://github.com/JosianaSilva/santander-customer-transaction-prediction.git
cd santander-customer-transaction-prediction
```

### 2. Baixar os dados da competição
Baixe os CSVs da competição do Kaggle:
https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

Extraia os arquivos na pasta `data/raw/`:
- `train.csv` (esse é necessário para o treinamento)
- `test.csv`
- `sample_submission.csv`

### 3. Treinar o modelo ML
```bash
python src/models/train.py
```

### 4. Executar a aplicação

#### Opção A: Com Docker
```bash
docker-compose up -d --build
```

#### Opção B: Sem Docker

- Criar e ativar ambiente virtual

```bash
python -m venv env
source env/Scripts/activate  # Windows
source env/bin/activate      # Linux/Mac
```

- Instalar dependências
```bash
pip install -r requirements.txt
```

- Executar aplicação
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

## 📋 Endpoints
- **API**: http://localhost:8000
- **Documentação**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health