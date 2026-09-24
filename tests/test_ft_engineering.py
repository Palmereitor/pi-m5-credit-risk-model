"""
Smoke tests basicos para el pipeline de ingenieria de caracteristicas.

Ejecucion:
    cd src && python -m pytest ../tests/test_ft_engineering.py -v
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from ft_engineering import load_data, clean_data, engineer_features, build_preprocessing_pipeline, split_data, TARGET


def test_load_data_shape():
    df = load_data()
    assert df.shape[0] > 0
    assert TARGET in df.columns


def test_clean_data_unifies_tendencia_ingresos():
    df = load_data()
    df_clean = clean_data(df)
    valores_validos = {"Creciente", "Estable", "Decreciente"}
    valores_presentes = set(df_clean["tendencia_ingresos"].dropna().unique())
    assert valores_presentes.issubset(valores_validos), "tendencia_ingresos contiene valores invalidos tras la limpieza"


def test_clean_data_caps_edad():
    df = load_data()
    df_clean = clean_data(df)
    assert df_clean["edad_cliente"].max() <= 90
    assert df_clean["edad_cliente"].min() >= 18


def test_engineer_features_creates_derived_columns():
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    for col in ["carga_financiera", "ratio_cuota_ingreso", "saldo_mora_ratio", "total_creditos_sector", "tiene_codeudor_en_mora"]:
        assert col in df.columns, f"Falta la columna derivada {col}"


def test_puntaje_excluded_from_features():
    """Verifica que la variable con data leakage ('puntaje') no se use como feature."""
    from ft_engineering import NUMERIC_FEATURES
    assert "puntaje" not in NUMERIC_FEATURES, "'puntaje' presenta data leakage (AUC=1.0) y no debe usarse como feature"


def test_pipeline_output_shapes():
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2)

    preprocessor = build_preprocessing_pipeline()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    assert X_train_proc.shape[0] == len(y_train)
    assert X_test_proc.shape[0] == len(y_test)
    assert X_train_proc.shape[1] == X_test_proc.shape[1]
