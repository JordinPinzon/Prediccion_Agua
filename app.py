import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.preparar_datos import cargar_y_unir_datos
from src.entrenar_modelo import entrenar_modelo_caudal
from src.predecir_nivel import predecir_nivel

# Configuración general de la página
st.set_page_config(page_title="Predicción H44", layout="wide")
st.title("🔵 Predicción del Nivel de Agua – Estación H44 Antisana")

# Cargar y procesar datos
with st.spinner("Entrenando modelo y generando predicción..."):
    df = cargar_y_unir_datos()
    forecast, modelo = entrenar_modelo_caudal(df)
    forecast = predecir_nivel(forecast, df)

# Tabulador visualización vs descargas
tab1, tab2, tab3 = st.tabs(["📈 Visualización", "📥 Descarga de resultados", "🌍 Mapa de la estación"])

with tab1:
    st.subheader("📊 Resumen de predicción")
    
    # Calcular métricas clave
    nivel_actual = forecast["nivel_estimado"].iloc[-1]
    nivel_max = forecast["nivel_estimado"].max()
    nivel_min = forecast["nivel_estimado"].min()
    pendiente = forecast["nivel_estimado"].iloc[-1] - forecast["nivel_estimado"].iloc[0]

    # Mostrar métricas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nivel actual (m)", f"{nivel_actual:.2f}")
    col2.metric("Máximo estimado (m)", f"{nivel_max:.2f}")
    col3.metric("Mínimo estimado (m)", f"{nivel_min:.2f}")
    col4.metric("Pendiente total (m)", f"{pendiente:.2f}")

    # Mostrar RMSE del modelo si está disponible
    rmse = forecast.attrs.get("rmse_nivel", None)
    if rmse is not None:
        st.metric("📉 Error del modelo (RMSE)", f"{rmse:.2f} m")

    # Filtro por rango de fechas
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

    # Filtrar dataframe por rango
    rango_inicio = pd.to_datetime(rango[0])
    rango_fin = pd.to_datetime(rango[1])
    df_filtrado = forecast[(forecast["ds"] >= rango_inicio) & (forecast["ds"] <= rango_fin)]

    # Gráfico único – Nivel estimado
    st.subheader("📈 Nivel de Agua Estimado")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_filtrado["ds"], df_filtrado["nivel_estimado"], label="Nivel estimado", color="#1f77b4", linewidth=2)

    # Línea de alerta visual
    nivel_critico = 5.0
    ax.axhline(y=nivel_critico, color='red', linestyle='--', label=f'Alerta crítica ({nivel_critico} m)')

    # Intervalo de confianza si está disponible
    if "yhat_lower" in df_filtrado.columns and "yhat_upper" in df_filtrado.columns:
        ax.fill_between(df_filtrado["ds"], df_filtrado["yhat_lower"], df_filtrado["yhat_upper"],
                        color="#1f77b4", alpha=0.2, label="Intervalo de confianza")

    ax.set_xlabel("Fecha")
    ax.set_ylabel("Nivel estimado (m)")
    ax.set_title(f"Predicción del Nivel de Agua hasta {fecha_max.year}")
    ax.grid()
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.subheader("⬇️ Descarga de datos")

    csv = forecast[["ds", "nivel_estimado"]].to_csv(index=False).encode("utf-8")
    st.download_button("📥 Descargar predicción como CSV", csv, "prediccion_nivel.csv", "text/csv")

    st.caption("Los datos corresponden a predicciones generadas con Prophet y regresión lineal multivariable.")

with tab3:
    st.subheader("📍 Ubicación de la estación H44 Antisana")

    from streamlit_folium import folium_static
    import folium

    # Coordenadas aproximadas de la estación H44 Antisana (ajústalas si tienes las reales)
    lat = -0.457
    lon = -78.205

    # Validación del RMSE para mostrar en popup
    error_text = f"{rmse:.2f} m" if rmse is not None else "No disponible"

    popup_html = f"""
    <b>Estación H44 Antisana</b><br>
    Nivel actual: {nivel_actual:.2f} m<br>
    Nivel máximo: {nivel_max:.2f} m<br>
    Nivel mínimo: {nivel_min:.2f} m<br>
    Error del modelo (RMSE): {error_text}
    """

    # Crear mapa
    mapa = folium.Map(location=[lat, lon], zoom_start=12)
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(popup_html, max_width=300),
        tooltip="H44 Antisana",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(mapa)

    # Mostrar mapa en pantalla completa
    st.markdown(
        """
        <style>
        .element-container:has(.folium-map) {
            width: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    folium_static(mapa, width=1400, height=600)
