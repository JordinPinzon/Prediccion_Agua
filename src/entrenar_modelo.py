from prophet import Prophet
import pandas as pd

def entrenar_modelo_caudal(df):
    df_prophet = df[["fecha", "caudal"]].copy()
    df_prophet.columns = ["ds", "y"]

    # Crear modelo con ajustes
    modelo = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.1  # menor = más suave ante cambios bruscos
    )

    modelo.fit(df_prophet)

    # Proyectar solo 365 días hacia el futuro
    future = modelo.make_future_dataframe(periods=365)
    forecast = modelo.predict(future)
    
    return forecast, modelo
