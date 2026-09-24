"""
model_monitoring.py

Componente de monitoreo de Data Drift para el modelo de riesgo crediticio.

Compara la distribucion de las variables de entrada (y opcionalmente las
predicciones) de un periodo "actual" contra una distribucion de referencia
(la usada durante el entrenamiento), utilizando:

    - Kolmogorov-Smirnov (KS test)            -> variables numericas
    - Population Stability Index (PSI)         -> variables numericas y categoricas
    - Jensen-Shannon divergence                -> variables numericas
    - Chi-cuadrado                             -> variables categoricas

Genera un reporte JSON por periodo con las metricas de drift y alertas
automaticas cuando se superan los umbrales criticos. El reporte es consumido
por la aplicacion de monitoreo en Streamlit (app/streamlit_app.py).

Uso:
    python model_monitoring.py

Autor: Daniel Palmera
"""

import os
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PROCESSED = os.path.join(BASE_DIR, "..", "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "..", "monitoring_reports")

# Umbrales de alerta (estandar de la industria para PSI; KS y JS calibrados empiricamente)
PSI_THRESHOLD_WARNING = 0.10
PSI_THRESHOLD_CRITICAL = 0.25
KS_PVALUE_THRESHOLD = 0.05
JS_THRESHOLD_WARNING = 0.10
CHI2_PVALUE_THRESHOLD = 0.05

NUMERIC_MONITOR_COLS = [
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "total_otros_prestamos", "cuota_pactada", "puntaje_datacredito",
    "cant_creditosvigentes", "huella_consulta", "saldo_mora", "saldo_total",
]
CATEGORICAL_MONITOR_COLS = ["tipo_laboral", "tipo_credito", "tendencia_ingresos"]


def population_stability_index(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """
    Calcula el PSI entre una distribucion de referencia (expected) y una
    distribucion actual (actual). Funciona tanto para variables numericas
    (usando cuantiles de 'expected' como cortes de bucket) como para
    categoricas (si se le pasan las categorias directamente).

    Interpretacion estandar:
        PSI < 0.10           -> sin cambio significativo
        0.10 <= PSI < 0.25    -> cambio moderado (alerta amarilla)
        PSI >= 0.25           -> cambio significativo (alerta roja, posible re-entrenamiento)
    """
    expected = expected.dropna()
    actual = actual.dropna()

    if pd.api.types.is_numeric_dtype(expected):
        breakpoints = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1)))
        if len(breakpoints) < 3:
            return 0.0
        expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    else:
        categories = sorted(set(expected.unique()) | set(actual.unique()))
        expected_pct = expected.value_counts(normalize=True).reindex(categories, fill_value=0).values
        actual_pct = actual.value_counts(normalize=True).reindex(categories, fill_value=0).values

    # Evitar log(0)
    expected_pct = np.clip(expected_pct, 1e-6, None)
    actual_pct = np.clip(actual_pct, 1e-6, None)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def ks_test_numeric(reference: pd.Series, current: pd.Series) -> dict:
    """Prueba de Kolmogorov-Smirnov para variables numericas continuas."""
    reference, current = reference.dropna(), current.dropna()
    statistic, pvalue = stats.ks_2samp(reference, current)
    return {"statistic": float(statistic), "pvalue": float(pvalue), "drift_detectado": bool(pvalue < KS_PVALUE_THRESHOLD)}


def jensen_shannon_numeric(reference: pd.Series, current: pd.Series, bins: int = 20) -> float:
    """Divergencia de Jensen-Shannon entre dos distribuciones numericas (via histogramas)."""
    reference, current = reference.dropna(), current.dropna()
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    if min_val == max_val:
        return 0.0
    edges = np.linspace(min_val, max_val, bins + 1)
    ref_hist = np.histogram(reference, bins=edges)[0].astype(float)
    cur_hist = np.histogram(current, bins=edges)[0].astype(float)
    ref_hist = ref_hist / ref_hist.sum() if ref_hist.sum() > 0 else ref_hist
    cur_hist = cur_hist / cur_hist.sum() if cur_hist.sum() > 0 else cur_hist
    return float(stats.entropy(ref_hist, (ref_hist + cur_hist) / 2, base=2) / 2 +
                 stats.entropy(cur_hist, (ref_hist + cur_hist) / 2, base=2) / 2) ** 0.5


def chi2_categorical(reference: pd.Series, current: pd.Series) -> dict:
    """Prueba de Chi-cuadrado de independencia para variables categoricas."""
    reference, current = reference.dropna(), current.dropna()
    categories = sorted(set(reference.unique()) | set(current.unique()))
    ref_counts = reference.value_counts().reindex(categories, fill_value=0)
    cur_counts = current.value_counts().reindex(categories, fill_value=0)
    contingency = np.array([ref_counts.values, cur_counts.values])
    try:
        chi2, pvalue, dof, _ = stats.chi2_contingency(contingency)
    except ValueError:
        return {"statistic": None, "pvalue": None, "drift_detectado": False}
    return {"statistic": float(chi2), "pvalue": float(pvalue), "drift_detectado": bool(pvalue < CHI2_PVALUE_THRESHOLD)}


def compute_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame, periodo: str = "actual") -> dict:
    """Calcula el reporte completo de drift (numericas + categoricas) entre dos dataframes."""
    report = {"periodo": periodo, "generado_en": datetime.now().isoformat(), "variables": {}, "alertas": []}

    for col in NUMERIC_MONITOR_COLS:
        if col not in reference_df.columns:
            continue
        psi = population_stability_index(reference_df[col], current_df[col])
        ks = ks_test_numeric(reference_df[col], current_df[col])
        js = jensen_shannon_numeric(reference_df[col], current_df[col])

        nivel = "OK"
        if psi >= PSI_THRESHOLD_CRITICAL:
            nivel = "CRITICO"
        elif psi >= PSI_THRESHOLD_WARNING or ks["drift_detectado"]:
            nivel = "ALERTA"

        report["variables"][col] = {
            "tipo": "numerica", "psi": round(psi, 4), "ks_statistic": round(ks["statistic"], 4),
            "ks_pvalue": round(ks["pvalue"], 4), "js_divergence": round(js, 4), "nivel": nivel,
        }
        if nivel != "OK":
            report["alertas"].append(f"[{nivel}] {col}: PSI={psi:.3f}, KS p-value={ks['pvalue']:.4f}")

    for col in CATEGORICAL_MONITOR_COLS:
        if col not in reference_df.columns:
            continue
        psi = population_stability_index(reference_df[col].astype(str), current_df[col].astype(str))
        chi2 = chi2_categorical(reference_df[col].astype(str), current_df[col].astype(str))

        nivel = "OK"
        if psi >= PSI_THRESHOLD_CRITICAL:
            nivel = "CRITICO"
        elif psi >= PSI_THRESHOLD_WARNING or (chi2["pvalue"] is not None and chi2["pvalue"] < CHI2_PVALUE_THRESHOLD):
            nivel = "ALERTA"

        report["variables"][col] = {
            "tipo": "categorica", "psi": round(psi, 4),
            "chi2_statistic": chi2["statistic"], "chi2_pvalue": chi2["pvalue"], "nivel": nivel,
        }
        if nivel != "OK":
            report["alertas"].append(f"[{nivel}] {col}: PSI={psi:.3f}, Chi2 p-value={chi2['pvalue']}")

    n_criticos = sum(1 for v in report["variables"].values() if v["nivel"] == "CRITICO")
    n_alertas = sum(1 for v in report["variables"].values() if v["nivel"] == "ALERTA")
    if n_criticos > 0:
        report["recomendacion"] = f"ACCION REQUERIDA: {n_criticos} variable(s) con drift critico. Se recomienda re-entrenar el modelo y revisar el proceso de captura de datos."
    elif n_alertas > 0:
        report["recomendacion"] = f"Monitorear de cerca: {n_alertas} variable(s) con drift moderado. Revisar en el proximo ciclo."
    else:
        report["recomendacion"] = "Sin drift significativo. El modelo sigue siendo representativo de la poblacion actual."

    return report


def simulate_periods(n_periods: int = 4):
    """
    Simula 'n_periods' periodos de produccion ordenando el dataset de test
    cronologicamente por fecha_prestamo y particionandolo en bloques. El
    conjunto de entrenamiento (train) se usa como distribucion de referencia.
    """
    X_train_raw = pd.read_csv(os.path.join(DATA_PROCESSED, "X_train_raw.csv"), parse_dates=["fecha_prestamo"] if False else None)
    dataset = pd.read_csv(os.path.join(DATA_PROCESSED, "dataset_cargado.csv"), parse_dates=["fecha_prestamo"])
    dataset = dataset.sort_values("fecha_prestamo").reset_index(drop=True)

    split_idx = int(len(dataset) * 0.8)
    reference_df = dataset.iloc[:split_idx]
    production_df = dataset.iloc[split_idx:].sort_values("fecha_prestamo")

    period_size = max(1, len(production_df) // n_periods)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    reports = []
    for i in range(n_periods):
        start = i * period_size
        end = (i + 1) * period_size if i < n_periods - 1 else len(production_df)
        current_df = production_df.iloc[start:end]
        if len(current_df) == 0:
            continue
        periodo_label = f"periodo_{i+1}"
        report = compute_drift_report(reference_df, current_df, periodo=periodo_label)
        report["fecha_inicio"] = str(current_df["fecha_prestamo"].min())
        report["fecha_fin"] = str(current_df["fecha_prestamo"].max())
        report["n_registros"] = len(current_df)
        reports.append(report)

        out_path = os.path.join(REPORTS_DIR, f"drift_report_{periodo_label}.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Reporte de drift generado: {out_path} | Alertas: {len(report['alertas'])}")

    consolidated_path = os.path.join(REPORTS_DIR, "drift_reports_consolidado.json")
    with open(consolidated_path, "w") as f:
        json.dump(reports, f, indent=2, default=str)
    logger.info(f"Reporte consolidado guardado en {consolidated_path}")

    return reports


if __name__ == "__main__":
    simulate_periods(n_periods=4)
