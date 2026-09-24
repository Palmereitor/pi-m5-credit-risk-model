# Imagen para el servicio de scoring de riesgo crediticio
FROM python:3.10-slim

WORKDIR /app

# Dependencias del sistema minimas para compilar xgboost/scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ft_engineering.py ./src/ft_engineering.py
COPY api/model_deploy.py ./api/model_deploy.py
COPY models/best_model.joblib ./models/best_model.joblib
COPY models/preprocessor.joblib ./models/preprocessor.joblib
COPY models/model_metadata.json ./models/model_metadata.json

ENV MODEL_PATH=/app/models/best_model.joblib
ENV PREPROCESSOR_PATH=/app/models/preprocessor.joblib
ENV METADATA_PATH=/app/models/model_metadata.json
ENV PYTHONPATH=/app/src

WORKDIR /app/api
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
