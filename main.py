import matplotlib.pyplot as plt
from src.preparar_datos import cargar_y_unir_datos
from src.entrenar_modelo import entrenar_modelo_caudal
from src.predecir_nivel import predecir_nivel
from src.predecir_precipitacion import predecir_precipitacion  # NUEVA IMPORTACIÓN

# Paso 1: Cargar y preparar datos
df = cargar_y_unir_datos()

# Paso 2: Entrenar modelo y predecir caudal hasta 2026
forecast, modelo = entrenar_modelo_caudal(df)

# Paso 3: Predecir precipitación futura y unirla
forecast_precip = predecir_precipitacion(df, dias=len(forecast))  # NUEVA LÍNEA
forecast = forecast.merge(forecast_precip, on="ds", how="left")   # NUEVA LÍNEA

# Paso 4: Predecir nivel del agua usando caudal y precipitación estimados
forecast = predecir_nivel(forecast, df)

# Paso 5: Visualizar predicción de nivel
plt.figure(figsize=(12,6))

# Línea del nivel estimado
plt.plot(forecast["ds"], forecast["nivel_estimado"], label="Nivel estimado", color="royalblue")

# Intervalo de confianza (usa los valores de Prophet antes de predecir nivel, si quieres, o crea unos artificiales alrededor del nivel estimado)
# Aquí generamos uno aproximado de ±10% para visualización
confianza_inferior = forecast["nivel_estimado"] * 0.9
confianza_superior = forecast["nivel_estimado"] * 1.1
plt.fill_between(forecast["ds"], confianza_inferior, confianza_superior, color="skyblue", alpha=0.3, label="Intervalo de confianza")

# Línea de alerta crítica
plt.axhline(y=5.0, color="red", linestyle="--", label="Alerta crítica (5.0 m)")

# Personalización
plt.title(f"Predicción del Nivel de Agua ({forecast['ds'].min().date()} a {forecast['ds'].max().date()})")
plt.xlabel("Fecha")
plt.ylabel("Nivel estimado (m)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

