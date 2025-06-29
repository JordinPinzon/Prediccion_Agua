import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.preparar_datos import cargar_y_unir_datos
from src.entrenar_modelo import entrenar_modelo_caudal
from src.predecir_nivel import predecir_nivel

# Configuración de la página
st.set_page_config(page_title="Predicción H44", layout="wide")
st.title("🔵 Predicción del Nivel de Agua – Estación Antisana")

# Cargar y procesar datos
with st.spinner("Entrenando modelo y generando predicción..."):
    df = cargar_y_unir_datos()
    forecast, modelo = entrenar_modelo_caudal(df)
    forecast = predecir_nivel(forecast, df)

    # Limitar forecast completo hasta 2023 incluyendo columnas de confianza
forecast = forecast.loc[forecast["ds"] < pd.to_datetime("2024-01-01")].copy()

# Eliminar columnas fuera del rango también si existen
if "yhat_lower" in forecast.columns:
    forecast["yhat_lower"] = forecast["yhat_lower"].where(forecast["ds"] < pd.to_datetime("2024-01-01"))
if "yhat_upper" in forecast.columns:
    forecast["yhat_upper"] = forecast["yhat_upper"].where(forecast["ds"] < pd.to_datetime("2024-01-01"))


# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Visualización", "🌍 Mapa de la estación", "📥 Descarga de resultados"])

with tab1:
    st.subheader("📊 Resumen de predicción")

    nivel_actual = forecast["nivel_estimado"].iloc[-1]
    nivel_max = forecast["nivel_estimado"].max()
    nivel_min = forecast["nivel_estimado"].min()
    pendiente = forecast["nivel_estimado"].iloc[-1] - forecast["nivel_estimado"].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nivel actual (m)", f"{nivel_actual:.2f}")
    col2.metric("Máximo estimado (m)", f"{nivel_max:.2f}")
    col3.metric("Mínimo estimado (m)", f"{nivel_min:.2f}")
    col4.metric("Pendiente total (m)", f"{pendiente:.2f}")

    rmse = forecast.attrs.get("rmse_nivel", None)
    if rmse is not None:
        st.metric("📉 Error del modelo (RMSE)", f"{rmse:.2f} m")

    st.subheader("📅 Selecciona el rango de fechas")
    fecha_min = pd.to_datetime(forecast["ds"].min()).date()
    fecha_max = pd.to_datetime(forecast["ds"].max()).date()

    rango = st.slider(
        "Rango de predicción",
        min_value=fecha_min,
        max_value=fecha_max,
        value=(fecha_min, fecha_max),
        format="YYYY-MM-DD"
    )

    # ✅ Línea corregida aquí:
    st.info("📅 Última fecha de predicción: 2023-12-31")

    rango_inicio = pd.to_datetime(rango[0])
    rango_fin = pd.to_datetime(rango[1])
    df_filtrado = forecast[(forecast["ds"] >= rango_inicio) & (forecast["ds"] <= rango_fin)]

    st.subheader("📈 Nivel de Agua Estimado")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_filtrado["ds"], df_filtrado["nivel_estimado"], label="Nivel estimado", color="#1f77b4", linewidth=2)
    nivel_critico = 5.0
    ax.axhline(y=nivel_critico, color='red', linestyle='--', label=f'Alerta crítica ({nivel_critico} m)')
    if "yhat_lower" in df_filtrado.columns and "yhat_upper" in df_filtrado.columns:
        ax.fill_between(df_filtrado["ds"], df_filtrado["yhat_lower"], df_filtrado["yhat_upper"],
                        color="#1f77b4", alpha=0.2, label="Intervalo de confianza")

    ax.set_xlabel("Fecha")
    ax.set_ylabel("Nivel estimado (m)")
    ax.set_title("Predicción del Nivel de Agua hasta 2023")

    # 👉 Recorte explícito al 2023
    ax.set_xlim(left=df_filtrado["ds"].min(), right=pd.Timestamp("2023-01-01 23:59:59"))

    ax.grid()
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.subheader("📍 Ubicación y origen de los datos – Sistema hídrico del Antisana")

    st.markdown("""
    Los datos utilizados para el análisis y predicción provienen de un conjunto de estaciones ubicadas en el **Parque Nacional Antisana**, que forman parte del sistema hídrico que abastece a Quito.

    -  Estación – Antisana DJ Diguchi:  
       
    -  Estación  – Antisana Diguchi: 
   
    -  Estación – Río Antisana AC: 
      
    Además, se consideran los principales ríos que alimentan el embalse La Mica, como el **río Diguchi, río Antisana y río Jatunhuaycu**, que recogen agua de deshielos y lluvias en el ecosistema del Antisana.

Este sistema conjunto permite comprender la dinámica hídrica que garantiza el abastecimiento de agua potable a Quito mediante el embalse **La Mica** y la planta **El Troje**.
    """)

    import folium
    import streamlit.components.v1 as components

    mapa = folium.Map(location=[-0.568, -78.229], zoom_start=11)

        # 🟩 Estación principal (ahora con mismo estilo que las otras)
    folium.Marker(
        location=[-0.5683880379564397, -78.2298390801277],
        popup="Estación H44 Antisana DJ Diguchi",
        tooltip="H44 DJ Diguchi",
        icon=folium.Icon(color="green", icon="leaf")
    ).add_to(mapa)

    # 🟢 Otras estaciones
    estaciones_adicionales = [
        {"nombre": "Estación Antisana Diguchi", "lat": -0.6022867145410288, "lon": -78.1986689291808},
        {"nombre": "Estación Río Antisana AC", "lat": -0.5934839659614135, "lon": -78.20825370752031},
    ]


    for est in estaciones_adicionales:
        folium.Marker(
            location=[est["lat"], est["lon"]],
            popup=est["nombre"],
            tooltip=est["nombre"],
            icon=folium.Icon(color="green", icon="leaf")
        ).add_to(mapa)

    # 🔵 Embalse La Mica
    folium.CircleMarker(
        location=[-0.53806, -78.21015],
        radius=12,
        popup="Embalse La Mica",
        color="red",
        fill=True,
        fill_opacity=0.5
    ).add_to(mapa)

    # 💧 Río Diguchi
    folium.Marker(
    location=[-0.5683880379564397, -78.2398390801277],
    popup="Río Diguchi - Estación H44 DJ Diguchi",
    tooltip="Río Diguchi (H44 DJ Diguchi)",
    icon=folium.Icon(color="blue", icon="tint")
    ).add_to(mapa)


    # 💧 Río Antisana
    folium.Marker(
        location=[-0.5783880379564397, -78.2298390801277],
        popup="Río Antisana",
        tooltip="Río Antisana",
        icon=folium.Icon(color="blue", icon="tint")
    ).add_to(mapa)

    # 💧 Río Jatunyacu
    folium.Marker(
        location=[-0.4935, -78.1810],
        popup="Río Jatunyacu",
        tooltip="Río Jatunyacu",
        icon=folium.Icon(color="blue", icon="tint")
    ).add_to(mapa)

        # 🏭 Planta de tratamiento El Troje
    folium.Marker(
        location=[-0.33343, -78.52261],
        popup="Planta de tratamiento El Troje",
        tooltip="El Troje",
        icon=folium.Icon(color="darkred", icon="industry", prefix='fa')
    ).add_to(mapa)


    # Renderizar mapa
    mapa.get_root().html.add_child(folium.Element("""
        <style>
        html, body, #map { width: 100%; height: 100%; margin: 0; padding: 0; }
        </style>
    """))
    mapa_html = mapa.get_root().render()
    components.html(f"""
        <div style="width: 100%; height: 600px;">
            {mapa_html}
        </div>
    """, height=600)

    # Imagen del flujo de agua
    import base64
    st.markdown("### 🗺️ Flujo del agua hacia la planta El Troje")
    with open("images/flujo_antisana_troje.png", "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{encoded}" style="max-width: 50%; border-radius: 8px;" />
        </div>
    """, unsafe_allow_html=True)



with tab3:
    st.subheader("⬇️ Descarga de datos")
    csv = forecast[["ds", "nivel_estimado"]].to_csv(index=False).encode("utf-8")
    st.download_button("📥 Descargar predicción como CSV", csv, "prediccion_nivel.csv", "text/csv")
    st.caption("Los datos corresponden a predicciones generadas con Prophet y regresión multivariable.")


