import matplotlib.pyplot as plt
from src.preparar_datos import cargar_y_unir_datos
from src.entrenar_modelo import entrenar_modelo_caudal
from src.predecir_nivel import predecir_nivel

# Paso 1: Cargar y preparar datos
df = cargar_y_unir_datos()

# Paso 2: Entrenar modelo y predecir caudal hasta 2026
forecast, modelo = entrenar_modelo_caudal(df)

# Paso 3: Predecir nivel del agua
forecast = predecir_nivel(forecast, df)

# Paso 4: Visualizar predicción de nivel
plt.figure(figsize=(12,6))
plt.plot(forecast["ds"], forecast["nivel_estimado"], label="Nivel estimado")
plt.title("Predicción del nivel de agua hasta 2026")
plt.xlabel("Fecha")
plt.ylabel("Nivel (m)")
plt.grid()
plt.legend()
plt.show()
