.PHONY: help train deploy clean setup test

PYTHON = python3
VENV_PATH = env
VENV_PYTHON = $(VENV_PATH)/Scripts/python.exe
MODELS_DIR = models
SRC_DIR = src
SCRIPTS_DIR = $(SRC_DIR)/scripts
METRICS_FILE = models/metrics.json

help:
	@echo "Comandos disponíveis:"
	@echo "  make setup     - Configura ambiente"
	@echo "  make train     - Treina modelo"
	@echo "  make deploy    - Deploy condicional para HF"
	@echo "  make clean     - Remove arquivos temporários"
	@echo "  make test      - Executa testes"

setup:
	$(PYTHON) -m venv $(VENV_PATH)
	$(VENV_PATH)/Scripts/activate && pip install -r requirements.txt

train:
	@mkdir -p $(MODELS_DIR)
	$(VENV_PYTHON) $(SCRIPTS_DIR)/train.py

deploy:
	$(VENV_PYTHON) $(SCRIPTS_DIR)/deploy.py

clean:
	rm -rf $(MODELS_DIR)/*.pkl $(METRICS_FILE) __pycache__ $(SRC_DIR)/__pycache__

test:
	$(VENV_PYTHON) -m pytest test/ -v

dev-train: train
	$(VENV_PYTHON) -c "import json; metrics = json.load(open('$(METRICS_FILE)')); print(f'Accuracy: {metrics[\"accuracy\"]:.4f}'); print(f'ROC AUC: {metrics[\"roc_auc\"]:.4f}')"

dev-info:
	@echo "Modelos: $(shell ls -1 $(MODELS_DIR)/*.pkl 2>/dev/null | wc -l)"
	@echo "Última atualização: $(shell stat -c %y $(METRICS_FILE) 2>/dev/null || echo 'Nenhum treino')"