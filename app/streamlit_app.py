"""
streamlit_app.py

Dashboard de monitoreo de Data Drift para el modelo de riesgo crediticio.

Consume los reportes generados por src/model_monitoring.py y los presenta
con:
    - Indicadores visuales de alerta (semaforo) por variable y periodo.
    - Tabla de metricas de drift (PSI, KS, Jensen-Shannon, Chi-cuadrado).
    - Comparacion de distribuciones (historica vs actual) por variable.
    - Evolucion temporal del drift a lo largo de los periodos simulados.
    - Recomendaciones automaticas (reentrenamiento / revision de variables).

Ejecucion:
    streamlit run streamlit_app.py

Autor: Daniel Palmera
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CURRENT_DIR, "..", "src"))

from model_monitoring import (  # noqa: E402
    NUMERIC_MONITOR_COLS, CATEGORICAL_MONITOR_COLS,
    PSI_THRESHOLD_WARNING, PSI_THRESHOLD_CRITICAL,
)

DATA_PROCESSED = os.path.join(CURRENT_DIR, "..", "data", "processed")
REPORTS_DIR = os.path.join(CURRENT_DIR, "..", "monitoring_reports")

st.set_page_config(page_title="Monitoreo de Data Drift - Riesgo Crediticio", layout="wide")

NAVY = "#1B3A5C"
GOLD = "#E8A020"
RED = "#C0392B"
GREEN = "#1A7A4A"

NIVEL_EMOJI = {"OK": "🟢", "ALERTA": "🟡", "CRITICO": "🔴"}
NIVEL_COLOR = {"OK": GREEN, "ALERTA": GOLD, "CRITICO": RED}


@st.cache_data
def load_reports():
    path = os.path.join(REPORTS_DIR, "drift_reports_consolidado.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_reference_and_dataset():
    dataset = pd.read_csv(os.path.join(DATA_PROCESSED, "dataset_cargado.csv"), parse_dates=["fecha_prestamo"])
    dataset = dataset.sort_values("fecha_prestamo").reset_index(drop=True)
    split_idx = int(len(dataset) * 0.8)
    reference_df = dataset.iloc[:split_idx]
    production_df = dataset.iloc[split_idx:].sort_values("fecha_prestamo")
    return reference_df, production_df


st.title("📊 Monitoreo de Data Drift — Modelo de Riesgo Crediticio")
st.caption("Proyecto Integrador M5 · Daniel Palmera · Comparacion de la distribucion historica (entrenamiento) vs periodos de produccion simulados")

reports = load_reports()

if not reports:
    st.error("No se encontraron reportes de drift. Ejecuta primero: `python src/model_monitoring.py`")
    st.stop()

reference_df, production_df = load_reference_and_dataset()

# ---------- Sidebar: seleccion de periodo ----------
periodos = [r["periodo"] for r in reports]
periodo_sel = st.sidebar.selectbox("Periodo a inspeccionar", periodos, index=len(periodos) - 1)
report_sel = next(r for r in reports if r["periodo"] == periodo_sel)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Registros en el periodo:** {report_sel.get('n_registros','-')}")
st.sidebar.markdown(f"**Desde:** {str(report_sel.get('fecha_inicio',''))[:10]}")
st.sidebar.markdown(f"**Hasta:** {str(report_sel.get('fecha_fin',''))[:10]}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Umbrales PSI**")
st.sidebar.markdown(f"🟢 OK: PSI < {PSI_THRESHOLD_WARNING}")
st.sidebar.markdown(f"🟡 Alerta: {PSI_THRESHOLD_WARNING} ≤ PSI < {PSI_THRESHOLD_CRITICAL}")
st.sidebar.markdown(f"🔴 Critico: PSI ≥ {PSI_THRESHOLD_CRITICAL}")

# ---------- Resumen / semaforo ----------
n_ok = sum(1 for v in report_sel["variables"].values() if v["nivel"] == "OK")
n_alerta = sum(1 for v in report_sel["variables"].values() if v["nivel"] == "ALERTA")
n_critico = sum(1 for v in report_sel["variables"].values() if v["nivel"] == "CRITICO")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Variables monitoreadas", len(report_sel["variables"]))
col2.metric("🟢 Sin drift", n_ok)
col3.metric("🟡 Drift moderado", n_alerta)
col4.metric("🔴 Drift critico", n_critico)

if n_critico > 0:
    st.error(f"🔴 {report_sel['recomendacion']}")
elif n_alerta > 0:
    st.warning(f"🟡 {report_sel['recomendacion']}")
else:
    st.success(f"🟢 {report_sel['recomendacion']}")

st.markdown("---")

# ---------- Tabla de metricas de drift ----------
st.subheader("Tabla de metricas de drift por variable")

rows = []
for var, m in report_sel["variables"].items():
    rows.append({
        "Variable": var,
        "Tipo": m["tipo"],
        "PSI": m["psi"],
        "KS / Chi2 p-value": m.get("ks_pvalue", m.get("chi2_pvalue")),
        "Jensen-Shannon": m.get("js_divergence", "-"),
        "Nivel": f"{NIVEL_EMOJI[m['nivel']]} {m['nivel']}",
    })
drift_table = pd.DataFrame(rows).sort_values("PSI", ascending=False)
st.dataframe(drift_table, use_container_width=True, hide_index=True)

# Barras de riesgo (PSI por variable)
fig, ax = plt.subplots(figsize=(10, max(3, len(drift_table) * 0.35)))
colors = [NIVEL_COLOR[n.split(" ")[1]] for n in drift_table["Nivel"]]
ax.barh(drift_table["Variable"], drift_table["PSI"], color=colors)
ax.axvline(PSI_THRESHOLD_WARNING, color=GOLD, ls="--", lw=1, label=f"Umbral alerta ({PSI_THRESHOLD_WARNING})")
ax.axvline(PSI_THRESHOLD_CRITICAL, color=RED, ls="--", lw=1, label=f"Umbral critico ({PSI_THRESHOLD_CRITICAL})")
ax.set_xlabel("PSI (Population Stability Index)")
ax.set_title(f"PSI por variable — {periodo_sel}")
ax.legend(fontsize=8)
ax.invert_yaxis()
st.pyplot(fig)

st.markdown("---")

# ---------- Comparacion de distribuciones ----------
st.subheader("Comparacion de distribucion: historica (train) vs periodo seleccionado")

var_sel = st.selectbox("Variable a comparar", list(report_sel["variables"].keys()))

current_period_idx = periodos.index(periodo_sel)
n_reports_cum = report_sel.get("n_registros", 0)

# Reconstruir el sub-dataframe del periodo seleccionado sobre production_df
n_periods_total = len(periodos)
period_size = max(1, len(production_df) // n_periods_total)
start = current_period_idx * period_size
end = (current_period_idx + 1) * period_size if current_period_idx < n_periods_total - 1 else len(production_df)
current_df = production_df.iloc[start:end]

fig2, ax2 = plt.subplots(figsize=(9, 4.5))
if var_sel in NUMERIC_MONITOR_COLS:
    ax2.hist(reference_df[var_sel].dropna(), bins=30, alpha=0.5, density=True, label="Historico (train)", color=NAVY)
    ax2.hist(current_df[var_sel].dropna(), bins=30, alpha=0.5, density=True, label=f"Actual ({periodo_sel})", color=RED)
    ax2.set_xlabel(var_sel)
    ax2.set_ylabel("Densidad")
else:
    ref_counts = reference_df[var_sel].astype(str).value_counts(normalize=True)
    cur_counts = current_df[var_sel].astype(str).value_counts(normalize=True)
    all_cats = sorted(set(ref_counts.index) | set(cur_counts.index))
    x = np.arange(len(all_cats))
    width = 0.35
    ax2.bar(x - width/2, [ref_counts.get(c, 0) for c in all_cats], width, label="Historico (train)", color=NAVY)
    ax2.bar(x + width/2, [cur_counts.get(c, 0) for c in all_cats], width, label=f"Actual ({periodo_sel})", color=RED)
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_cats, rotation=30, ha="right")
    ax2.set_ylabel("Proporcion")

ax2.set_title(f"Distribucion de {var_sel}")
ax2.legend()
st.pyplot(fig2)

m_sel = report_sel["variables"][var_sel]
st.info(f"**{var_sel}** — PSI: {m_sel['psi']} | Nivel: {NIVEL_EMOJI[m_sel['nivel']]} {m_sel['nivel']}")

st.markdown("---")

# ---------- Evolucion temporal del drift ----------
st.subheader("Evolucion temporal del drift (PSI por periodo)")

evol_rows = []
for r in reports:
    for var, m in r["variables"].items():
        evol_rows.append({"periodo": r["periodo"], "variable": var, "psi": m["psi"]})
evol_df = pd.DataFrame(evol_rows)
pivot = evol_df.pivot(index="periodo", columns="variable", values="psi")

vars_to_plot = st.multiselect(
    "Variables a graficar en la evolucion temporal",
    list(pivot.columns),
    default=list(drift_table.sort_values("PSI", ascending=False)["Variable"].head(4)),
)

if vars_to_plot:
    fig3, ax3 = plt.subplots(figsize=(10, 4.5))
    for var in vars_to_plot:
        ax3.plot(pivot.index, pivot[var], marker="o", label=var)
    ax3.axhline(PSI_THRESHOLD_WARNING, color=GOLD, ls="--", lw=1)
    ax3.axhline(PSI_THRESHOLD_CRITICAL, color=RED, ls="--", lw=1)
    ax3.set_ylabel("PSI")
    ax3.set_title("Evolucion de PSI a lo largo de los periodos")
    ax3.legend(fontsize=8, loc="upper left")
    ax3.tick_params(axis="x", rotation=15)
    st.pyplot(fig3)

    # Deteccion simple de tendencia (creciente / decreciente / estable)
    st.markdown("**Deteccion de tendencia:**")
    for var in vars_to_plot:
        serie = pivot[var].values
        if len(serie) >= 2:
            pendiente = np.polyfit(range(len(serie)), serie, 1)[0]
            if pendiente > 0.02:
                tendencia = "📈 Creciente (el drift se agrava con el tiempo)"
            elif pendiente < -0.02:
                tendencia = "📉 Decreciente (el drift se atenua)"
            else:
                tendencia = "➡️ Estable"
            st.write(f"- **{var}**: {tendencia} (pendiente={pendiente:.4f})")

st.markdown("---")

# ---------- Recomendaciones consolidadas ----------
st.subheader("Recomendaciones automaticas")
for r in reports:
    icon = "🔴" if any(v["nivel"] == "CRITICO" for v in r["variables"].values()) else (
        "🟡" if any(v["nivel"] == "ALERTA" for v in r["variables"].values()) else "🟢"
    )
    with st.expander(f"{icon} {r['periodo']} — {r['recomendacion']}"):
        if r["alertas"]:
            for a in r["alertas"]:
                st.write(f"- {a}")
        else:
            st.write("Sin alertas en este periodo.")
