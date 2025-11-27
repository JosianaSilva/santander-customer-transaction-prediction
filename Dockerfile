FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir uvicorn[standard]

COPY src/ ./src/

# Create models directory and download from Hugging Face
RUN mkdir -p models

ARG HF_USERNAME  
ARG HF_REPO_NAME
ENV HF_USERNAME=${HF_USERNAME}
ENV HF_REPO_NAME=${HF_REPO_NAME}

# Download models from Hugging Face
RUN python -c "import os; from src.scripts.huggingface import download_model_from_hf; repo_id = f'{os.environ.get(\"HF_USERNAME\")}/{os.environ.get(\"HF_REPO_NAME\")}'; download_model_from_hf(repo_id)"

RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]