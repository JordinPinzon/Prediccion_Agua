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
plt.plot(forecast["ds"], forecast["nivel_estimado"], label="Nivel estimado")
plt.title("Predicción del nivel de agua hasta 2026")
plt.xlabel("Fecha")
plt.ylabel("Nivel (m)")
plt.grid()
plt.legend()
plt.show()
