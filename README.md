# Modelo Predictivo de Riesgo Crediticio

**Proyecto Integrador M5 — Soy Henry Data Science**
Daniel Palmera · 2026

Sistema de clasificación binaria para anticipar el comportamiento de pago de nuevos clientes de crédito, desplegado como pipeline reproducible con componentes de ingesta, EDA, feature engineering, modelado, despliegue (API) y monitoreo de data drift.

---

## Caso de negocio

Un equipo de Datos y Analítica de una empresa financiera necesita anticipar, **antes de otorgar un crédito**, si un cliente nuevo pagará a tiempo (`Pago_atiempo = 1`) o no (`Pago_atiempo = 0`). El dataset contiene el historial de créditos otorgados junto con variables demográficas, financieras y de comportamiento crediticio (algunas reportadas por una central de riesgo externa, DataCrédito).

El objetivo del proyecto es construir, evaluar y desplegar un modelo de clasificación que sirva de insumo para la decisión de originación de crédito, junto con la infraestructura mínima de MLOps (API de scoring y monitoreo de drift) para operarlo en producción.

---

## Estructura del repositorio

```
pi-m5-credit-risk-model/
├── set_up.bat                  # script de setup de entorno (entregado por la organización)
├── requirements.txt            # dependencias del proyecto (entregado + ampliado)
├── src/
│   ├── config.json             # configuración del proyecto (project_code, etc.)
│   ├── cargar_datos.ipynb      # Avance 1 — carga de datos
│   ├── comprension_eda.ipynb   # Avance 1 — EDA completo
│   ├── ft_engineering.py       # Avance 2 — ingeniería de características (pipeline sklearn)
│   ├── modelamiento.ipynb      # Avance 2 — entrenamiento y comparación de modelos
│   └── model_monitoring.py     # Avance 3 — cálculo de data drift
├── api/
│   ├── model_deploy.py         # Avance 4 — servicio FastAPI (predict / predict batch)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── .dockerignore
├── app/
│   ├── streamlit_app.py        # dashboard de monitoreo de drift
│   └── requirements.txt
├── data/
│   ├── raw/Base_de_datos.xlsx  # dataset original entregado por la organización
│   └── processed/              # datasets intermedios generados por el pipeline
├── models/                     # modelo entrenado, preprocesador y metadata (generados)
├── monitoring_reports/         # reportes de drift generados por model_monitoring.py
└── docs/
    └── diccionario_datos.md    # diccionario de datos entregado por la organización
```

---

## Dataset

- **Fuente:** `data/raw/Base_de_datos.xlsx`, entregado por la organización como dataset de ejemplo no productivo (en un entorno real, la información residiría en el Data Warehouse/Data Lake corporativo).
- **Tamaño:** 10,763 registros, 22 variables predictoras + variable objetivo.
- **Variable objetivo:** `Pago_atiempo` (1 = pagó a tiempo, 0 = no pagó a tiempo).
- **Diccionario de datos:** ver `docs/diccionario_datos.md`.

---

## Pipeline y hallazgos principales

### 1. Carga de datos (`src/cargar_datos.ipynb`)
Carga el archivo fuente, valida su estructura y persiste una copia intermedia en `data/processed/dataset_cargado.csv`.

### 2. EDA (`src/comprension_eda.ipynb`)
Análisis exploratorio univariable, bivariable y multivariable. Hallazgos clave:

- **Desbalance de clases severo:** ~95.2% pagó a tiempo vs ~4.8% no pagó a tiempo (razón ≈ 20:1). Esto descarta Accuracy como métrica principal; se prioriza ROC-AUC, PR-AUC, Recall y F1.
- **Data leakage crítico detectado y corregido:** la variable `puntaje` separa perfectamente ambas clases (AUC individual = 1.0000, sin solapamiento entre distribuciones). Esto indica que el score fue calculado con información posterior al desenlace del crédito y **no estaría disponible en el momento real de originar un préstamo**. La variable se excluyó completamente del modelado (documentado en detalle en el notebook de EDA, sección 4.1, y en `ft_engineering.py`).
- **Calidad de datos:** 58 registros (0.5%) con valores numéricos inválidos en `tendencia_ingresos` (se unificaron a nulo); outliers en `edad_cliente` (máx. 123 años), `salario_cliente` (máx. 22,000 millones), `puntaje_datacredito` (rango fuera de [150,950]).
- **Variables predictoras relevantes:** `puntaje_datacredito`, `huella_consulta`, `tendencia_ingresos` y variables de carga financiera muestran relación visible con la variable objetivo.
- **Multicolinealidad:** `saldo_total` y `saldo_principal` están altamente correlacionadas.

### 3. Ingeniería de características (`src/ft_engineering.py`)
Pipeline de `sklearn` (`ColumnTransformer`) ajustado únicamente sobre el conjunto de entrenamiento:

- Limpieza según las reglas de validación definidas en el EDA (capping de outliers, unificación de nulos).
- Atributos derivados: `carga_financiera`, `ratio_cuota_ingreso`, `saldo_mora_ratio`, `total_creditos_sector`, `tiene_codeudor_en_mora`, `mes_prestamo`.
- Imputación (mediana / moda), One-Hot Encoding (nominales), Ordinal Encoding (`tendencia_ingresos`), escalado (`StandardScaler`).
- Split estratificado 80/20 (dado el desbalance de clases).

### 4. Modelado (`src/modelamiento.ipynb`)
Se compararon 4 modelos supervisados, todos con balanceo de clases (`class_weight='balanced'` / `scale_pos_weight`):

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Gradient Boosting** ⭐ | 0.9522 | 0.9534 | 0.9985 | 0.9755 | **0.6760** | 0.9751 |
| Random Forest | 0.9229 | 0.9573 | 0.9620 | 0.9596 | 0.6523 | 0.9733 |
| Regresión Logística | 0.6479 | 0.9681 | 0.6519 | 0.7791 | 0.6555 | 0.9714 |
| XGBoost | 0.8082 | 0.9638 | 0.8298 | 0.8918 | 0.6403 | 0.9714 |

**Modelo seleccionado:** Gradient Boosting, por mejor ROC-AUC en el conjunto de evaluación. Se reporta también PR-AUC (Average Precision) porque, dado el fuerte desbalance de clases, es más representativa del desempeño real sobre la clase minoritaria que la ROC-AUC. El modelo, el preprocesador y su metadata se persisten en `models/`.

### 5. Despliegue (`api/model_deploy.py`)
API construida con **FastAPI**, dockerizada, que expone:

- `GET /health` — estado del servicio.
- `GET /model/info` — metadata del modelo en producción.
- `POST /predict` — predicción para un solo cliente (probabilidad, clasificación de riesgo Bajo/Medio/Alto, threshold configurable).
- `POST /predict/batch` — predicción por lotes (múltiples registros en una sola solicitud).

Construcción de la imagen (desde la raíz del repositorio):
```bash
docker build -f api/Dockerfile -t riesgo-crediticio-api .
docker run -p 8000:8000 riesgo-crediticio-api
```
Ejecución local sin Docker:
```bash
cd api
pip install -r requirements.txt
uvicorn model_deploy:app --reload --port 8000
```

### 6. Monitoreo de Data Drift (`src/model_monitoring.py` + `app/streamlit_app.py`)
`model_monitoring.py` simula 4 periodos de producción (particionando cronológicamente el conjunto de evaluación) y calcula, para cada variable y periodo, contra la distribución de entrenamiento como referencia:

- **Kolmogorov-Smirnov (KS test)** — variables numéricas.
- **Population Stability Index (PSI)** — numéricas y categóricas.
- **Jensen-Shannon divergence** — variables numéricas.
- **Chi-cuadrado** — variables categóricas.

Genera reportes JSON en `monitoring_reports/` con niveles de alerta (🟢 OK / 🟡 Alerta / 🔴 Crítico) y recomendaciones automáticas. En la ejecución de referencia se detectó drift **crítico en `plazo_meses`** y **alerta en `salario_cliente`**, indicando un cambio en el mix de créditos originados a lo largo del tiempo.

El dashboard en Streamlit visualiza estos reportes de forma interactiva:
```bash
cd app
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Incluye: tabla de métricas de drift, barras de riesgo por variable, comparación de distribuciones histórica vs actual, evolución temporal del PSI con detección de tendencia, y panel de recomendaciones.

---

## Reproducción del pipeline completo

```bash
# 1. Setup de entorno (Windows) o instalación manual (macOS/Linux)
pip install -r requirements.txt

# 2. Ejecutar notebooks en orden (Jupyter)
jupyter nbconvert --to notebook --execute --inplace src/cargar_datos.ipynb
jupyter nbconvert --to notebook --execute --inplace src/comprension_eda.ipynb

# 3. Feature engineering
python src/ft_engineering.py

# 4. Modelado
jupyter nbconvert --to notebook --execute --inplace src/modelamiento.ipynb

# 5. Monitoreo de drift
python src/model_monitoring.py

# 6. API (en otra terminal)
cd api && uvicorn model_deploy:app --reload

# 7. Dashboard de monitoreo (en otra terminal)
cd app && streamlit run streamlit_app.py
```

---

## Principales conclusiones y recomendaciones

1. El desbalance de clases (95/5) exige evaluar el modelo con ROC-AUC, PR-AUC, Recall y F1, no con Accuracy.
2. El hallazgo de data leakage en `puntaje` fue crítico: incluirla habría producido un modelo con métricas perfectas pero inútil en producción. Se recomienda a la organización auditar el proceso de cálculo de esa variable.
3. El modelo final (Gradient Boosting) prioriza Recall de la clase positiva mayoritaria manteniendo buen PR-AUC (0.975) sobre la detección de no pago; el threshold de decisión puede ajustarse en producción según el apetito de riesgo del negocio.
4. Se detectó drift real en variables de plazo y salario entre el periodo de entrenamiento y los periodos simulados de producción, lo que sugiere monitorear de cerca el mix de créditos originados y considerar reentrenamiento periódico.
5. Próximos pasos sugeridos: incorporar SHAP values para explicabilidad por cliente, agregar autenticación a la API, y programar el reentrenamiento automático cuando el monitoreo detecte drift crítico sostenido.

---

## Stack técnico

Python · pandas · scikit-learn · XGBoost · FastAPI · Docker · Streamlit · SciPy (drift statistics) · Jupyter

---

## Autor

**Daniel Palmera** — Soy Henry Data Science, Proyecto Integrador M5, 2026.
