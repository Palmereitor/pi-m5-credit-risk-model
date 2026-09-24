"""
model_deploy.py

Expone el modelo de riesgo crediticio seleccionado (best_model.joblib) como
un servicio REST mediante FastAPI, con soporte para prediccion individual y
por lotes (batch).

Endpoints:
    GET  /                  - info general
    GET  /health            - health check
    GET  /model/info        - metadata del modelo en produccion
    POST /predict            - prediccion para un solo cliente
    POST /predict/batch      - prediccion para multiples clientes (batch)

Ejecucion local:
    uvicorn model_deploy:app --reload --port 8000

Autor: Daniel Palmera
"""

import os
import sys
import json
import logging
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Permite importar clean_data / engineer_features desde src/ft_engineering.py
# tanto en ejecucion local (../src) como en el contenedor Docker (PYTHONPATH=/app/src)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_DIR, "..", "src"))
sys.path.insert(0, os.path.join(CURRENT_DIR, "..", "..", "src"))

from ft_engineering import clean_data, engineer_features, NUMERIC_FEATURES, NOMINAL_FEATURES, ORDINAL_FEATURES, BINARY_FEATURES  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(CURRENT_DIR, "..", "models", "best_model.joblib"))
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", os.path.join(CURRENT_DIR, "..", "models", "preprocessor.joblib"))
METADATA_PATH = os.getenv("METADATA_PATH", os.path.join(CURRENT_DIR, "..", "models", "model_metadata.json"))

logger.info(f"Cargando modelo desde {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
with open(METADATA_PATH) as f:
    metadata = json.load(f)
logger.info(f"Modelo cargado: {metadata['model_name']} (ROC-AUC={metadata['metrics']['roc_auc']})")

app = FastAPI(
    title="API - Modelo Predictivo de Riesgo Crediticio",
    description="Servicio de scoring de riesgo crediticio (Proyecto Integrador M5 - Daniel Palmera)",
    version="1.0.0",
)


class CreditoInput(BaseModel):
    tipo_credito: str = Field(..., description="Codigo del tipo de credito, ej. '7'")
    fecha_prestamo: str = Field(..., description="Fecha del prestamo ISO 8601, ej. '2026-01-15T10:00:00'")
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    tipo_laboral: str = Field(..., description="'Empleado' o 'Independiente'")
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    puntaje_datacredito: Optional[float] = None
    cant_creditosvigentes: int
    huella_consulta: int
    saldo_mora: Optional[float] = 0
    saldo_total: Optional[float] = 0
    saldo_principal: Optional[float] = 0
    saldo_mora_codeudor: Optional[float] = 0
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: Optional[float] = None
    tendencia_ingresos: Optional[str] = Field(None, description="'Creciente', 'Estable', 'Decreciente' o null")

    class Config:
        json_schema_extra = {
            "example": {
                "tipo_credito": "7", "fecha_prestamo": "2026-01-15T10:00:00",
                "capital_prestado": 5000000, "plazo_meses": 24, "edad_cliente": 35,
                "tipo_laboral": "Empleado", "salario_cliente": 2500000,
                "total_otros_prestamos": 1000000, "cuota_pactada": 250000,
                "puntaje_datacredito": 720, "cant_creditosvigentes": 3, "huella_consulta": 2,
                "saldo_mora": 0, "saldo_total": 1500000, "saldo_principal": 1400000,
                "saldo_mora_codeudor": 0, "creditos_sectorFinanciero": 2,
                "creditos_sectorCooperativo": 0, "creditos_sectorReal": 1,
                "promedio_ingresos_datacredito": 2400000, "tendencia_ingresos": "Estable",
            }
        }


class BatchInput(BaseModel):
    registros: List[CreditoInput]


class PredictionOutput(BaseModel):
    probabilidad_pago_atiempo: float
    probabilidad_no_pago: float
    prediccion: int
    riesgo: str
    threshold_usado: float


def _clasificar_riesgo(prob_no_pago: float) -> str:
    if prob_no_pago < 0.10:
        return "Bajo"
    elif prob_no_pago < 0.30:
        return "Medio"
    return "Alto"


def _prepare_dataframe(records: List[dict]) -> pd.DataFrame:
    """Aplica el mismo flujo de limpieza + feature engineering que ft_engineering.py."""
    df = pd.DataFrame(records)
    df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"])
    df["Pago_atiempo"] = 0  # placeholder requerido solo por compatibilidad de clean_data, no se usa
    df["tipo_credito"] = df["tipo_credito"].astype(str)

    df = clean_data(df)
    df = engineer_features(df)

    feature_cols = NUMERIC_FEATURES + NOMINAL_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES
    return df[feature_cols]


def _predict_df(X_raw: pd.DataFrame, threshold: float) -> List[PredictionOutput]:
    X_proc = preprocessor.transform(X_raw)
    probas = model.predict_proba(X_proc)[:, 1]  # prob de Pago_atiempo=1
    outputs = []
    for p in probas:
        prob_no_pago = 1 - p
        outputs.append(PredictionOutput(
            probabilidad_pago_atiempo=round(float(p), 4),
            probabilidad_no_pago=round(float(prob_no_pago), 4),
            prediccion=int(p >= threshold),
            riesgo=_clasificar_riesgo(prob_no_pago),
            threshold_usado=threshold,
        ))
    return outputs


@app.get("/")
def root():
    return {
        "servicio": "API Riesgo Crediticio",
        "modelo_en_produccion": metadata["model_name"],
        "endpoints": ["/health", "/model/info", "/predict", "/predict/batch"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "modelo_cargado": metadata["model_name"]}


@app.get("/model/info")
def model_info():
    return metadata


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: CreditoInput, threshold: float = 0.5):
    try:
        X_raw = _prepare_dataframe([payload.model_dump()])
        result = _predict_df(X_raw, threshold)[0]
        return result
    except Exception as e:
        logger.exception("Error en /predict")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch")
def predict_batch(payload: BatchInput, threshold: float = 0.5):
    if not payload.registros:
        raise HTTPException(status_code=400, detail="La lista 'registros' no puede estar vacia")
    try:
        X_raw = _prepare_dataframe([r.model_dump() for r in payload.registros])
        results = _predict_df(X_raw, threshold)
        return {"total_registros": len(results), "predicciones": results}
    except Exception as e:
        logger.exception("Error en /predict/batch")
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("model_deploy:app", host="0.0.0.0", port=8000, reload=True)
