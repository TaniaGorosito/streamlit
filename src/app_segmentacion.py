from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Segmentación de Empresas - IA", layout="centered"
)
st.title("¿A qué perfil de adopción pertenece la empresa?")
st.caption(
    "Clasificación en tiempo real mediante K-Means (k=3) y StandardScaler"
)


# 1. Búsqueda automática y robusta del archivo .joblib
@st.cache_resource
def cargar_modelo():
  nombre_archivo = "agrupamiento_empresas.joblib"
  ruta_encontrada = None
  directorios = [Path.cwd(), *Path.cwd().parents]

  for d in directorios:
    coincidencias = list(d.glob(f"**/{nombre_archivo}"))
    if coincidencias:
      ruta_encontrada = coincidencias[0]
      break

  if ruta_encontrada is None:
    st.error(
        f"No se encontró el archivo '{nombre_archivo}'. Asegúrate de haber"
        " ejecutado la celda de guardado en el cuaderno."
    )
    st.stop()

  return joblib.load(ruta_encontrada)


guardado = cargar_modelo()
modelo = guardado["modelo"]
nombres = guardado["nombres"]
columnas = guardado["columnas"]
perfiles = guardado["perfiles"]

# 2. Controles de entrada
st.subheader("Ingrese los datos de la empresa")
c1, c2 = st.columns(2)

with c1:
  task_auto = st.slider("Tasa de automatización (%)", 0.0, 60.0, 25.0, 0.5)
  time_saved = st.slider("Horas ahorradas por semana", 0.0, 20.0, 7.0, 0.5)
  productivity = st.slider("Cambio de productividad (%)", 0.0, 35.0, 12.0, 0.5)

with c2:
  revenue_growth = st.slider(
      "Crecimiento de ingresos (%)", -5.0, 30.0, 6.0, 0.5
  )
  desplazados = st.slider(
      "Puestos desplazados (por 1.000 emp.)", 0.0, 100.0, 25.0, 1.0
  )
  creados = st.slider(
      "Puestos creados (por 1.000 emp.)", 0.0, 100.0, 30.0, 1.0
  )

# 3. Armar DataFrame con las columnas del modelo
fila = pd.DataFrame([{
    "task_automation_rate": task_auto,
    "time_saved_per_week": time_saved,
    "productivity_change_percent": productivity,
    "revenue_growth_percent": revenue_growth,
    "desplazados_x1000": desplazados,
    "creados_x1000": creados,
}])[columnas]

# 4. Clasificación
cluster_predicho = int(modelo.predict(fila)[0])

X_escalado = modelo.named_steps["escalador"].transform(fila)
distancias = np.linalg.norm(
    X_escalado - modelo.named_steps["kmeans"].cluster_centers_, axis=1
)[0]

st.subheader("Resultado de la clasificación")
st.metric("Perfil asignado:", nombres[cluster_predicho])

st.write("Distancia relativa a cada centroide:")
st.write(
    " · ".join([
        f"**{nombres[i]}:** {distancias[i]:.2f}" for i in range(len(nombres))
    ])
)

st.subheader("Comportamiento típico esperado para este grupo")
st.dataframe(perfiles.loc[[cluster_predicho]])