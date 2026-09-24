"""
ft_engineering.py

Componente de ingenieria de caracteristicas del pipeline de riesgo crediticio.

Responsabilidades:
    1. Cargar el dataset intermedio generado por cargar_datos.ipynb.
    2. Limpiar anomalias detectadas en el EDA (comprension_eda.ipynb).
    3. Generar atributos derivados con fundamento de negocio.
    4. Construir un Pipeline de sklearn (imputacion + encoding + escalado)
       ajustado UNICAMENTE sobre el conjunto de entrenamiento.
    5. Retornar/persistir los conjuntos de entrenamiento y evaluacion listos
       para el modelado.

Uso:
    python ft_engineering.py
    o
    from ft_engineering import run_pipeline
    X_train, X_test, y_train, y_test, preprocessor = run_pipeline()

Autor: Daniel Palmera
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PROCESSED = os.path.join(BASE_DIR, "..", "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

TARGET = "Pago_atiempo"

VALID_TENDENCIA = {"Creciente", "Estable", "Decreciente"}

# NOTA: 'puntaje' se excluye deliberadamente. El EDA (comprension_eda.ipynb,
# seccion 4.1) detecto que esta variable separa PERFECTAMENTE la clase objetivo
# (AUC=1.0, sin solapamiento), lo cual es evidencia de Data Leakage: el score
# fue calculado con informacion del desenlace / posterior al otorgamiento del
# credito y no estaria disponible en el momento real de originar un prestamo.
NUMERIC_FEATURES = [
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "total_otros_prestamos", "cuota_pactada", "puntaje_datacredito",
    "cant_creditosvigentes", "huella_consulta", "saldo_mora", "saldo_total",
    "saldo_principal", "saldo_mora_codeudor", "creditos_sectorFinanciero",
    "creditos_sectorCooperativo", "creditos_sectorReal",
    "promedio_ingresos_datacredito",
    # derivadas
    "carga_financiera", "ratio_cuota_ingreso", "saldo_mora_ratio",
    "total_creditos_sector",
]

BINARY_FEATURES = ["tiene_codeudor_en_mora"]

NOMINAL_FEATURES = ["tipo_laboral", "tipo_credito"]
ORDINAL_FEATURES = ["tendencia_ingresos"]


def load_data(path: str = None) -> pd.DataFrame:
    """Carga el dataset intermedio persistido por cargar_datos.ipynb."""
    path = path or os.path.join(DATA_PROCESSED, "dataset_cargado.csv")
    logger.info(f"Cargando dataset desde {path}")
    df = pd.read_csv(path, parse_dates=["fecha_prestamo"])
    logger.info(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica las reglas de validacion y limpieza identificadas en el EDA
    (comprension_eda.ipynb, seccion 5 - Reglas de validacion de datos).
    """
    df = df.copy()

    # Regla 1: tendencia_ingresos solo admite categorias validas -> resto a NaN
    mask_invalida = ~df["tendencia_ingresos"].isin(VALID_TENDENCIA) & df["tendencia_ingresos"].notna()
    n_invalidas = mask_invalida.sum()
    if n_invalidas:
        logger.info(f"Unificando {n_invalidas} valores invalidos de tendencia_ingresos a NaN")
    df.loc[mask_invalida, "tendencia_ingresos"] = np.nan

    # Regla 2: edad_cliente en [18, 90] -> capping
    df["edad_cliente"] = df["edad_cliente"].clip(lower=18, upper=90)

    # Regla 3: puntaje_datacredito rango plausible [150, 950] -> capping
    df["puntaje_datacredito"] = df["puntaje_datacredito"].clip(lower=150, upper=950)

    # Regla 4: salario_cliente sin negativos, winsorizing percentil 99.5
    df["salario_cliente"] = df["salario_cliente"].clip(lower=0)
    p995 = df["salario_cliente"].quantile(0.995)
    df["salario_cliente"] = df["salario_cliente"].clip(upper=p995)

    # tipo_credito como texto para tratarlo como nominal puro
    df["tipo_credito"] = df["tipo_credito"].astype(str)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera los atributos derivados identificados en el EDA
    (comprension_eda.ipynb, seccion 7 - Atributos derivados propuestos).
    """
    df = df.copy()

    df["carga_financiera"] = df["total_otros_prestamos"] / (df["salario_cliente"] + 1)
    df["ratio_cuota_ingreso"] = df["cuota_pactada"] / (df["salario_cliente"] + 1)
    df["saldo_mora_ratio"] = df["saldo_mora"] / (df["saldo_total"] + 1)
    df["total_creditos_sector"] = (
        df["creditos_sectorFinanciero"] + df["creditos_sectorCooperativo"] + df["creditos_sectorReal"]
    )
    df["tiene_codeudor_en_mora"] = (df["saldo_mora_codeudor"].fillna(0) > 0).astype(int)

    # Estacionalidad (mes del prestamo), sin usar la fecha absoluta para evitar fuga temporal
    df["mes_prestamo"] = df["fecha_prestamo"].dt.month
    NUMERIC_FEATURES.append("mes_prestamo") if "mes_prestamo" not in NUMERIC_FEATURES else None

    return df


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    Construye el ColumnTransformer con las transformaciones definidas en el EDA:
    imputacion + escalado para numericas, imputacion + One-Hot para nominales,
    imputacion + Ordinal para tendencia_ingresos.

    El pipeline se ajusta (fit) EXCLUSIVAMENTE sobre el conjunto de entrenamiento
    para prevenir data leakage; el conjunto de prueba solo se transforma.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    nominal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    ordinal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(
            categories=[["Decreciente", "Estable", "Creciente"]],
            handle_unknown="use_encoded_value", unknown_value=-1,
        )),
    ])

    binary_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("nom", nominal_transformer, NOMINAL_FEATURES),
        ("ord", ordinal_transformer, ORDINAL_FEATURES),
        ("bin", binary_transformer, BINARY_FEATURES),
    ])

    return preprocessor


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split estratificado 80/20 (estratificado por la variable objetivo dado el
    fuerte desbalance de clases documentado en el EDA).
    """
    feature_cols = NUMERIC_FEATURES + NOMINAL_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")
    logger.info(f"Distribucion target train: {y_train.value_counts(normalize=True).round(3).to_dict()}")
    logger.info(f"Distribucion target test:  {y_test.value_counts(normalize=True).round(3).to_dict()}")
    return X_train, X_test, y_train, y_test


def run_pipeline(save_artifacts: bool = True):
    """Ejecuta el flujo completo de ingenieria de caracteristicas."""
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)

    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessing_pipeline()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    logger.info(f"Shape final X_train procesado: {X_train_proc.shape}")
    logger.info(f"Shape final X_test procesado:  {X_test_proc.shape}")

    if save_artifacts:
        os.makedirs(DATA_PROCESSED, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)

        X_train.to_csv(os.path.join(DATA_PROCESSED, "X_train_raw.csv"), index=False)
        X_test.to_csv(os.path.join(DATA_PROCESSED, "X_test_raw.csv"), index=False)
        y_train.to_csv(os.path.join(DATA_PROCESSED, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(DATA_PROCESSED, "y_test.csv"), index=False)

        joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.joblib"))
        logger.info("Artefactos guardados: X_train_raw.csv, X_test_raw.csv, y_train.csv, y_test.csv, preprocessor.joblib")

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    run_pipeline()
