from prophet import Prophet
import pandas as pd
from datetime import datetime

def entrenar_modelo_caudal(df):
    df_prophet = df[["fecha", "caudal"]].copy()
    df_prophet.columns = ["ds", "y"]

    # Crear modelo Prophet con estacionalidad y suavizado
    modelo = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.1
    )

    modelo.fit(df_prophet)

    # Proyectar hasta 31 de diciembre de 2023
    fecha_final = datetime(2023, 12, 31)
    dias_extra = (fecha_final - df_prophet["ds"].max()).days

    # Evitar errores si los datos ya llegan a esa fecha
    dias_extra = max(0, dias_extra)

    future = modelo.make_future_dataframe(periods=dias_extra)
    forecast = modelo.predict(future)

    # Limitar explícitamente a 2023
    forecast = forecast[forecast["ds"] <= "2023-12-31"]


    return forecast, modelo
