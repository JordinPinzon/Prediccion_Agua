from prophet import Prophet
import pandas as pd
from datetime import datetime

def entrenar_modelo_caudal(df):
    # Usar caudal y precipitación
    df_prophet = df[["fecha", "caudal", "precipitacion"]].copy()
    df_prophet.columns = ["ds", "y", "precipitacion"]

    # ➕ Agregar columna de lag (precipitacion del día anterior)
    df_prophet["precipitacion_lag1"] = df_prophet["precipitacion"].shift(1)

    # Eliminar filas con valores nulos por el shift
    df_prophet = df_prophet.dropna()

    # Crear modelo Prophet con regresores
    modelo = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.1
    )
    modelo.add_regressor("precipitacion")
    modelo.add_regressor("precipitacion_lag1")

    # Entrenar modelo
    modelo.fit(df_prophet)

    # Definir fecha final de predicción
    fecha_final = datetime(2025, 12, 31)
    dias_extra = (fecha_final - df_prophet["ds"].max()).days
    dias_extra = max(0, dias_extra)

    # Generar fechas futuras
    future = modelo.make_future_dataframe(periods=dias_extra)

    # Unir datos de precipitación para las fechas futuras
    df_precip = df[["fecha", "precipitacion"]].copy()
    df_precip.columns = ["ds", "precipitacion"]

    future = future.merge(df_precip, on="ds", how="left")

    # Rellenar precipitación faltante con promedio
    future["precipitacion"].fillna(df["precipitacion"].mean(), inplace=True)

    # ➕ Agregar lag de precipitación en fechas futuras
    future["precipitacion_lag1"] = future["precipitacion"].shift(1)
    future["precipitacion_lag1"].fillna(df["precipitacion"].mean(), inplace=True)

    # Predecir caudal futuro
    forecast = modelo.predict(future)

    # Limitar explícitamente hasta 2025-12-31
    forecast = forecast[forecast["ds"] <= "2025-12-31"]

    return forecast, modelo
