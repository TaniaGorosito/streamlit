from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Plataforma de Adopción e Impacto de IA Empresarial",
    page_icon="📊",
    layout="wide",
)


# 1. Carga robusta y ultraligera del artefacto serializado (.joblib)
@st.cache_resource
def cargar_artefacto():
  nombre = "modelos_tp4.joblib"
  for d in [Path.cwd(), *Path.cwd().parents]:
    encontrados = list(d.glob(f"**/{nombre}"))
    if encontrados:
      return joblib.load(encontrados[0])
  st.error(
      f"No se encontró '{nombre}'. Ejecutá primero la celda exportadora en el"
      " cuaderno Jupyter."
  )
  st.stop()


datos_modelo = cargar_artefacto()
paises = datos_modelo["paises"]
industrias = datos_modelo["industrias"]

# Inicializar registro de monitoreo básico en sesión (Actividad de Extensión)
if "historial_predicciones" not in st.session_state:
  st.session_state["historial_predicciones"] = []

# Encabezado principal (Consigna A.2)
st.title("Sistema Integrado de Evaluación de Adopción e Impacto de IA")
st.markdown(
    "Aplicación interactiva de puesta en producción (**Trabajo Práctico 4**)"
    " entrenada sobre un panel de **150.000 registros (10.000 empresas globales,"
    " 2023–2026)**. Permite estimar la etapa de madurez tecnológica, la tasa"
    " esperada de adopción y el arquetipo de impacto operativo y laboral sin"
    " requerir conocimientos de programación."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Clasificación de Etapa (TP3)",
    "2. Predicción de Tasa de Adopción (TP2)",
    "3. Segmentación por Impacto (K-Means)",
    "4. Carga Masiva CSV y Monitoreo",
])

# ==============================================================================
# PESTAÑA 1: CLASIFICACIÓN DE ETAPA DE ADOPCIÓN (Regresión Logística Balanceada)
# ==============================================================================
with tab1:
  st.subheader("Clasificador de Etapa de Adopción de IA")
  st.caption(
      "Modelo seleccionado: Regresión Logística Multinomial Balanceada (C=10,"
      " F1 Macro = 0,495 en test de 2.000 empresas nuevas)."
  )

  col_in1, col_in2, col_in3 = st.columns(3)
  with col_in1:
    pais_clf = st.selectbox(
        "País de origen",
        paises,
        index=paises.index("USA") if "USA" in paises else 0,
        key="c_pais",
    )
    ind_clf = st.selectbox(
        "Sector industrial",
        industrias,
        index=industrias.index("Technology")
        if "Technology" in industrias
        else 0,
        key="c_ind",
    )
    anio_clf = st.selectbox(
        "Año del relevamiento", [2023, 2024, 2025, 2026], index=3, key="c_anio"
    )
  with col_in2:
    emp_clf = st.number_input(
        "Cantidad de empleados",
        min_value=10,
        max_value=25000,
        value=450,
        step=10,
        key="c_emp",
    )
    rev_clf = st.number_input(
        "Ingresos anuales ( millones USD)",
        min_value=0.5,
        max_value=15000.0,
        value=85.0,
        step=5.0,
        key="c_rev",
    )
    edad_clf = st.slider(
        "Antigüedad de la empresa (años)", 1, 80, 15, key="c_edad"
    )
  with col_in3:
    horas_clf = st.slider(
        "Horas anuales de capacitación en IA",
        0.0,
        500.0,
        120.0,
        step=5.0,
        key="c_horas",
    )
    tools_clf = st.slider(
        "Cantidad de herramientas de IA activas", 0, 15, 4, key="c_tools"
    )
    umbral_full = st.slider(
        "Umbral de exigencia para alerta 'Full'",
        0.20,
        0.95,
        0.50,
        0.05,
        help=(
            "Subir este umbral aumenta la Precisión (menos falsas alarmas) en"
            " la etapa Full."
        ),
    )

  # Validación de coherencia de entradas
  if tools_clf == 0 and horas_clf > 200:
    st.warning(
        "Atención: ingresaste más de 200 horas de capacitación pero 0"
        " herramientas activas. Verificá si el dato operativo es correcto."
    )

  # Inferencia matemática directa con los pesos de la Regresión Logística
  cfg_c = datos_modelo["clasificacion"]
  x_l = (
      np.log1p(np.array([emp_clf, rev_clf, horas_clf], dtype=float))
      - cfg_c["log_mean"]
  ) / cfg_c["log_scale"]
  x_e = (
      np.array([edad_clf, anio_clf, tools_clf], dtype=float)
      - cfg_c["lin_mean"]
  ) / cfg_c["lin_scale"]
  c_vec = np.zeros(len(paises))
  c_vec[paises.index(pais_clf)] = 1.0
  i_vec = np.zeros(len(industrias))
  i_vec[industrias.index(ind_clf)] = 1.0
  x_total_clf = np.concatenate([x_l, x_e, c_vec, i_vec])

  logits = cfg_c["coef"] @ x_total_clf + cfg_c["intercept"]
  logits_shift = logits - np.max(logits)
  probs = np.exp(logits_shift) / np.sum(np.exp(logits_shift))
  clases_modelo = cfg_c["clases"]
  mapa_probs = {clases_modelo[i]: float(probs[i]) for i in range(len(clases_modelo))}

  # Regla de decisión con umbral ajustable en 'full'
  etapa_pred = max(mapa_probs, key=mapa_probs.get)
  if etapa_pred == "full" and mapa_probs["full"] < umbral_full:
    probs_sin_full = {k: v for k, v in mapa_probs.items() if k != "full"}
    etapa_pred = max(probs_sin_full, key=probs_sin_full.get)

  etiquetas_es = {
      "none": "Sin adopción (None · <10%)",
      "pilot": "Piloto inicial (Pilot · 10% a 35%)",
      "partial": "Adopción parcial (Partial · 35% a 70%)",
      "full": "Adopción plena (Full · >70%)",
  }

  st.divider()
  c_res1, c_res2 = st.columns([1, 1.3])
  with c_res1:
    st.metric("Etapa predicha para la organización", etiquetas_es[etapa_pred])
    st.metric(
        "Confianza en la clase ganadora", f"{mapa_probs[etapa_pred]*100:.1f}%"
    )
    st.info(
        "**Cómo leer este resultado:** La importancia por permutación demostró"
        " que las **horas de capacitación** y la **cantidad de herramientas**"
        " concentran más del 90% del poder explicativo del clasificador."
    )
    if st.button("Registrar predicción en el historial", key="btn_log_clf"):
      st.session_state["historial_predicciones"].append({
          "Fecha": datetime.now().strftime("%H:%M:%S"),
          "Modelo": "Clasificación Logística",
          "Entrada clave": f"{pais_clf}, {ind_clf}, {horas_clf}h cap., {tools_clf} tools",
          "Resultado": etiquetas_es[etapa_pred],
      })
      st.success("Guardado en la pestaña de Monitoreo.")

  with c_res2:
    orden_etapas = ["none", "pilot", "partial", "full"]
    df_probs = pd.DataFrame({
        "Etapa": [etiquetas_es[k] for k in orden_etapas],
        "Probabilidad (%)": [round(mapa_probs[k] * 100, 1) for k in orden_etapas],
    })
    fig_probs = px.bar(
        df_probs,
        x="Probabilidad (%)",
        y="Etapa",
        orientation="h",
        text="Probabilidad (%)",
        range_x=[0, 100],
        title="Probabilidad estimada para cada etapa de adopción",
        color="Probabilidad (%)",
        color_continuous_scale="Blues",
    )
    fig_probs.update_layout(height=290, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_probs, use_container_width=True)

  with st.expander("Ver explicabilidad del modelo (Importancia por Permutación)"):
    imp_df = pd.DataFrame({
        "Variable": list(cfg_c["importancia_perm"].keys()),
        "Caída en F1 Macro al desordenar": list(cfg_c["importancia_perm"].values()),
    }).sort_values("Caída en F1 Macro al desordenar", ascending=True)
    fig_imp = px.bar(
        imp_df,
        x="Caída en F1 Macro al desordenar",
        y="Variable",
        orientation="h",
        title="Peso real de cada variable en el clasificador (Test = 30.004 filas)",
    )
    st.plotly_chart(fig_imp, use_container_width=True)

# ==============================================================================
# PESTAÑA 2: PREDICCIÓN DE TASA DE ADOPCIÓN (Regresión Ridge)
# ==============================================================================
with tab2:
  st.subheader("Estimador Continuo de Tasa de Adopción (Regresión Ridge)")
  st.caption(
      "Modelo seleccionado en TP2: Regresión Ridge (alpha=200, RMSE = 13,89 pp,"
      " MAE = 11,13 pp, R² = 0,098 sobre empresas no vistas)."
  )

  r_col1, r_col2 = st.columns(2)
  with r_col1:
    pais_reg = st.selectbox(
        "País",
        paises,
        index=paises.index("USA") if "USA" in paises else 0,
        key="r_pais",
    )
    ind_reg = st.selectbox(
        "Industria",
        industrias,
        index=industrias.index("Technology")
        if "Technology" in industrias
        else 0,
        key="r_ind",
    )
    anio_reg = st.slider("Año de observación", 2023, 2026, 2025, key="r_anio")
  with r_col2:
    emp_reg = st.number_input(
        "Dotación de empleados",
        min_value=10,
        max_value=25000,
        value=1200,
        step=50,
        key="r_emp",
    )
    rev_reg = st.number_input(
        "Facturación anual (millones USD)",
        min_value=0.5,
        max_value=15000.0,
        value=250.0,
        step=10.0,
        key="r_rev",
    )
    edad_reg = st.slider(
        "Antigüedad de la firma (años)", 1, 80, 20, key="r_edad"
    )

  cfg_r = datos_modelo["regresion"]
  xr_l = (
      np.log1p(np.array([emp_reg, rev_reg], dtype=float)) - cfg_r["log_mean"]
  ) / cfg_r["log_scale"]
  xr_e = (
      np.array([edad_reg, anio_reg], dtype=float) - cfg_r["esc_mean"]
  ) / cfg_r["esc_scale"]
  cr_vec = np.zeros(len(paises))
  cr_vec[paises.index(pais_reg)] = 1.0
  ir_vec = np.zeros(len(industrias))
  ir_vec[industrias.index(ind_reg)] = 1.0
  x_total_reg = np.concatenate([xr_l, xr_e, cr_vec, ir_vec])

  tasa_pred = float(
      np.clip(cfg_r["coef"] @ x_total_reg + cfg_r["intercept"], 0.0, 100.0)
  )
  dif_media = tasa_pred - cfg_r["media_historica"]

  st.divider()
  m1, m2, m3 = st.columns(3)
  m1.metric(
      "Tasa de adopción estimada",
      f"{tasa_pred:.1f}%",
      f"{dif_media:+.1f} pp vs. promedio global",
  )
  m2.metric(
      "Banda esperada (MAE ±11,1 pp)",
      f"{max(0.0, tasa_pred-11.1):.1f}% – {min(100.0, tasa_pred+11.1):.1f}%",
  )
  m3.metric("Promedio histórico del panel", f"{cfg_r['media_historica']:.1f}%")

  fig_gauge = go.Figure(
      go.Indicator(
          mode="gauge+number",
          value=tasa_pred,
          title={
              "text": (
                  "Ubicación respecto a la distribución histórica (Cuartiles"
                  " P25 / P50 / P75)"
              )
          },
          gauge={
              "axis": {"range": [0, 100]},
              "bar": {"color": "#1f77b4"},
              "steps": [
                  {"range": [0, cfg_r["p25"]], "color": "#e5e7eb"},
                  {"range": [cfg_r["p25"], cfg_r["p75"]], "color": "#bfdbfe"},
                  {"range": [cfg_r["p75"], 100], "color": "#93c5fd"},
              ],
              "threshold": {
                  "line": {"color": "red", "width": 4},
                  "thickness": 0.75,
                  "value": cfg_r["media_historica"],
              },
          },
      )
  )
  fig_gauge.update_layout(height=280)
  st.plotly_chart(fig_gauge, use_container_width=True)
  st.caption(
      "Nota metodológica: Los atributos de catálogo explican el ~10% de la"
      " variabilidad total (R² = 0,098). El ~90% restante responde a prácticas"
      " internas de capacitación y despliegue de herramientas."
  )

# ==============================================================================
# PESTAÑA 3: SEGMENTACIÓN DE EMPRESAS POR IMPACTO (K-Means, k=3)
# ==============================================================================
with tab3:
  st.subheader("Segmentación No Supervisada según Impacto Operativo y Laboral")
  st.caption(
      "Agrupamiento K-Means (k=3) con StandardScaler sobre las 10.000 empresas"
      " consolidadas, normalizando el impacto laboral cada 1.000 empleados."
  )

  cfg_k = datos_modelo["clustering"]
  k_c1, k_c2 = st.columns(2)
  with k_c1:
    task_auto = st.slider("Tasa de automatización de tareas (%)", 0.0, 60.0, 28.0, 0.5)
    time_saved = st.slider("Horas ahorradas por semana", 0.0, 20.0, 8.0, 0.5)
    productivity = st.slider("Cambio en la productividad (%)", 0.0, 35.0, 14.0, 0.5)
  with k_c2:
    revenue_growth = st.slider("Crecimiento de ingresos (%)", -5.0, 30.0, 7.0, 0.5)
    desplazados = st.slider(
        "Puestos desplazados (por cada 1.000 emp.)", 0.0, 100.0, 22.0, 1.0
    )
    creados = st.slider(
        "Puestos creados (por cada 1.000 emp.)", 0.0, 100.0, 30.0, 1.0
    )

  vec_k = np.array([
      task_auto,
      time_saved,
      productivity,
      revenue_growth,
      desplazados,
      creados,
  ])
  vec_k_esc = (vec_k - cfg_k["medias"]) / cfg_k["escalas"]
  distancias = np.linalg.norm(cfg_k["centroides"] - vec_k_esc, axis=1)
  cluster_ganador = int(np.argmin(distancias))
  nombre_ganador = cfg_k["nombres"][cluster_ganador]

  st.divider()
  kc_res1, kc_res2 = st.columns([1, 1.4])
  with kc_res1:
    st.metric("Arquetipo asignado", nombre_ganador)
    st.write("**Distancia euclídea a cada centroide estandarizado:**")
    for idx_c, dist_c in enumerate(distancias):
      st.write(f"- **{cfg_k['nombres'][idx_c]}:** `{dist_c:.2f}`")
    st.write("**Perfil promedio del segmento asignado:**")
    st.dataframe(cfg_k["perfiles"].loc[[cluster_ganador]], use_container_width=True)

  with kc_res2:
    muestra_df = cfg_k["muestra"]
    fig_scatter = px.scatter(
        muestra_df,
        x="task_automation_rate",
        y="productivity_change_percent",
        color="perfil",
        opacity=0.55,
        title="Ubicación de la empresa ingresada frente a los 3 segmentos",
        labels={
            "task_automation_rate": "Tasa de automatización (%)",
            "productivity_change_percent": "Cambio de productividad (%)",
        },
    )
    fig_scatter.add_trace(
        go.Scatter(
            x=[task_auto],
            y=[productivity],
            mode="markers",
            marker=dict(color="black", size=16, symbol="star", line=dict(width=2, color="white")),
            name="Empresa evaluada",
        )
    )
    fig_scatter.update_layout(height=360)
    st.plotly_chart(fig_scatter, use_container_width=True)

# ==============================================================================
# PESTAÑA 4: CARGA MASIVA EN LOTE (CSV) Y MONITOREO (Actividades de Extensión)
# ==============================================================================
with tab4:
  st.subheader("Evaluación por Lotes (CSV) y Registro de Predicciones")
  st.write(
      "Subí un archivo CSV con múltiples empresas para clasificar su etapa y"
      " estimar su tasa de adopción de forma masiva, o descargá una plantilla de"
      " ejemplo."
  )

  df_ejemplo = pd.DataFrame([
      {
          "country": "USA",
          "industry": "Technology",
          "survey_year": 2026,
          "num_employees": 850,
          "annual_revenue_usd_millions": 140.0,
          "company_age": 12,
          "ai_training_hours": 180.0,
          "num_ai_tools_used": 6,
      },
      {
          "country": "Germany",
          "industry": "Manufacturing",
          "survey_year": 2025,
          "num_employees": 320,
          "annual_revenue_usd_millions": 45.0,
          "company_age": 28,
          "ai_training_hours": 35.0,
          "num_ai_tools_used": 2,
      },
  ])
  st.download_button(
      "Descargar plantilla CSV de ejemplo",
      data=df_ejemplo.to_csv(index=False).encode("utf-8"),
      file_name="plantilla_empresas_ia.csv",
      mime="text/csv",
  )

  archivo_csv = st.file_uploader("Cargar archivo CSV de empresas", type=["csv"])
  if archivo_csv is not None:
    df_lote = pd.read_csv(archivo_csv)
    cols_req = {
        "country",
        "industry",
        "survey_year",
        "num_employees",
        "annual_revenue_usd_millions",
        "company_age",
        "ai_training_hours",
        "num_ai_tools_used",
    }
    if not cols_req.issubset(set(df_lote.columns)):
      st.error(f"El CSV debe incluir las columnas: {sorted(list(cols_req))}")
    else:
      resultados_etapa = []
      resultados_tasa = []
      for _, fila in df_lote.iterrows():
        # Regresión
        xrl = (
            np.log1p([
                float(fila["num_employees"]),
                float(fila["annual_revenue_usd_millions"]),
            ])
            - cfg_r["log_mean"]
        ) / cfg_r["log_scale"]
        xre = (
            np.array([float(fila["company_age"]), float(fila["survey_year"])])
            - cfg_r["esc_mean"]
        ) / cfg_r["esc_scale"]
        cv = np.zeros(len(paises))
        if str(fila["country"]) in paises:
          cv[paises.index(str(fila["country"]))] = 1.0
        iv = np.zeros(len(industrias))
        if str(fila["industry"]) in industrias:
          iv[industrias.index(str(fila["industry"]))] = 1.0
        t_pred = float(
            np.clip(
                cfg_r["coef"] @ np.concatenate([xrl, xre, cv, iv])
                + cfg_r["intercept"],
                0.0,
                100.0,
            )
        )
        resultados_tasa.append(round(t_pred, 1))

        # Clasificación
        xcl = (
            np.log1p([
                float(fila["num_employees"]),
                float(fila["annual_revenue_usd_millions"]),
                float(fila["ai_training_hours"]),
            ])
            - cfg_c["log_mean"]
        ) / cfg_c["log_scale"]
        xce = (
            np.array([
                float(fila["company_age"]),
                float(fila["survey_year"]),
                float(fila["num_ai_tools_used"]),
            ])
            - cfg_c["lin_mean"]
        ) / cfg_c["lin_scale"]
        lg = (
            cfg_c["coef"] @ np.concatenate([xcl, xce, cv, iv])
            + cfg_c["intercept"]
        )
        resultados_etapa.append(cfg_c["clases"][int(np.argmax(lg))])

      df_lote["tasa_adopcion_estimada_%"] = resultados_tasa
      df_lote["etapa_clasificada"] = resultados_etapa
      st.success(f"Procesadas {len(df_lote)} organizaciones correctamente.")
      st.dataframe(df_lote, use_container_width=True)
      st.download_button(
          "Descargar resultados en CSV",
          data=df_lote.to_csv(index=False).encode("utf-8"),
          file_name="resultados_prediccion_ia.csv",
          mime="text/csv",
      )

  st.divider()
  st.subheader("Registro de Monitoreo de Uso de la Sesión")
  if st.session_state["historial_predicciones"]:
    st.dataframe(
        pd.DataFrame(st.session_state["historial_predicciones"]),
        use_container_width=True,
    )
  else:
    st.caption(
        "Aún no registraste predicciones manuales en esta sesión (usá el botón"
        " en la Pestaña 1)."
    )