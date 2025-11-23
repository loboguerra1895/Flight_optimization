# streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

MODEL_PATH = "model_artifacts/rf_pipeline.joblib"
DISTANCE_SOURCE = "flight_delays_hackathon.csv"
CSV_PATH = "model_artifacts/user_queries.csv"

@st.cache_resource
def load_pipeline(path=MODEL_PATH):
    return joblib.load(path)

pipeline = load_pipeline()

def load_distances():
    df = pd.read_csv(DISTANCE_SOURCE)
    dist_dict = {}
    for _, row in df.iterrows():
        key = (row["origin_airport"], row["dest_airport"])
        dist_dict[key] = row["distance"]
    return dist_dict

distance_dict = load_distances()

st.title("Sistema Inteligente: Optimización de Vuelos")
st.write("Ingrese un vuelo.")

# --- Sidebar: solo entrada manual ---
st.sidebar.header("Entrada manual")

st.sidebar.markdown("**Nuevo vuelo**")
airline = st.sidebar.selectbox("Aerolínea", ["DL", "UA", "AA"])
origin = st.sidebar.selectbox("Aeropuerto origen (código)", ["ATL", "JFK", "LAX", "ORD", "DFW", "MIA"])
dest = st.sidebar.selectbox("Aeropuerto destino (código)", ["ATL", "JFK", "LAX", "ORD", "DFW", "MIA"])
month = st.sidebar.number_input("Mes (1-12)", min_value=1, max_value=12, value=7)
day_of_week = st.sidebar.number_input("Día semana (1-7)", min_value=1, max_value=7, value=1)
hour_of_day = st.sidebar.number_input("Hora del día (0-23)", min_value=0, max_value=23, value=8)
distance = st.sidebar.number_input("Distancia (mi)", min_value=0, value=distance_dict.get((origin, dest)))
weather_condition = st.sidebar.selectbox("Clima", ["Clear", "Rain", "Snow"])
is_holiday = st.sidebar.selectbox("Es feriado?", [0,1])

if st.sidebar.button("Guardar consulta y predecir"):
    
    # Crear fila base
    new_row = {
        "airline": airline,
        "origin_airport": origin,
        "dest_airport": dest,
        "month": month,
        "day_of_week": day_of_week,
        "hour_of_day": hour_of_day,
        "distance": distance,
        "weather_condition": weather_condition,
        "is_holiday": is_holiday
    }
    
    df_new = pd.DataFrame([new_row])

    # ------------------- PREDICCIÓN ----------------------
    preds = pipeline.predict(df_new)
    pred_delay = preds[0]
    df_new["predicted_delay"] = pred_delay

    # ------------------- RECOMENDACIÓN -------------------
    def recommend_action(pred_delay):
        if pred_delay < 5:
            return "Mantener horario (On time)."
        elif pred_delay < 20:
            return "Monitorear y considerar ajustes pequeños."
        else:
            return "Reprogramar / asignar aeronave alternativa."

    recommendation = recommend_action(pred_delay)
    df_new["recommendation"] = recommendation

    # Mostrar resultados
    st.subheader("Predicción")
    st.write(f"**Retraso estimado: {pred_delay:.2f} minutos**")
    st.subheader("Recomendación")
    st.write(f"**{recommendation}**")

    # ------------------- GUARDAR EN CSV -------------------
    df_new["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not os.path.exists(CSV_PATH):
        df_new.to_csv(CSV_PATH, index=False)
    else:
        df_old = pd.read_csv(CSV_PATH)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all.to_csv(CSV_PATH, index=False)

    st.success("Consulta guardada en user_queries.csv")

    # ------------------- JUSTIFICACIÓN SHAP -------------------
    st.subheader("Justificación (SHAP)")

    try:
        model = pipeline.named_steps["model"]
        preproc = pipeline.named_steps["preprocess"]

        X_trans = preproc.transform(df_new)

        # Obtener nombres
        num_cols = preproc.transformers_[0][2]

        try:
            cat = preproc.named_transformers_["cat"]
            cat_feature_names = cat.get_feature_names_out(preproc.transformers_[1][2])
        except:
            cat_feature_names = [f"cat_{i}" for i in range(X_trans.shape[1] - len(num_cols))]

        feature_names = list(num_cols) + list(cat_feature_names)

        # SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_trans)[0]

        shap_abs = np.abs(shap_values)
        top_idx = np.argsort(shap_abs)[::-1][:5]

        st.markdown("**Top 5 factores que más influyeron:**")
        for i in top_idx:
            val = shap_values[i]
            direction = "aumentó" if val > 0 else "disminuyó"
            st.write(f"- **{feature_names[i]}**: {direction} el retraso ({val:.2f} minutos)")

        # Gráfica
        fig, ax = plt.subplots(figsize=(6,3))
        sns.barplot(
            x=shap_abs[top_idx],
            y=[feature_names[i] for i in top_idx],
            ax=ax
        )
        ax.set_xlabel("Importancia absoluta (SHAP)")
        st.pyplot(fig)

    except Exception as e:
        st.error("Error generando SHAP:")
        st.write(e)

else:
    st.info("Complete la información en el sidebar y presione el botón.")

if os.path.exists(CSV_PATH):
    st.markdown("---")
    st.subheader("Consultas guardadas")
    df_queries = pd.read_csv(CSV_PATH)
    st.dataframe(df_queries)