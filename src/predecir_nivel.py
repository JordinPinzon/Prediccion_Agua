from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np


def predecir_nivel(forecast, df_original):
    # Agregar columnas de mes y año a los datos originales
    df_original = df_original.copy()
    df_original["mes"] = df_original["fecha"].dt.month
    df_original["año"] = df_original["fecha"].dt.year

    # Entrenar el modelo multivariable
    reg = LinearRegression()
    reg.fit(df_original[["caudal", "mes", "año"]], df_original["nivel"])

    # Calcular RMSE del modelo con los datos de entrenamiento
    y_true = df_original["nivel"]
    y_pred = reg.predict(df_original[["caudal", "mes", "año"]])
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Preparar el forecast para predecir
    forecast_temp = forecast.copy()
    forecast_temp = forecast_temp.rename(columns={"yhat": "caudal"})
    forecast_temp["mes"] = forecast_temp["ds"].dt.month
    forecast_temp["año"] = forecast_temp["ds"].dt.year

    # Predecir el nivel estimado
    forecast["nivel_estimado"] = reg.predict(forecast_temp[["caudal", "mes", "año"]])

    # Guardar RMSE dentro del objeto forecast para usarlo en app.py
    forecast.attrs = {"rmse_nivel": rmse}

    return forecast
