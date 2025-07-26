from prophet import Prophet
import pandas as pd

def predecir_precipitacion(df, dias=30):
    df_prep = df[["fecha", "precipitacion"]].copy()
    df_prep.columns = ["ds", "y"]

    modelo = Prophet()
    modelo.fit(df_prep)

    future = modelo.make_future_dataframe(periods=dias)
    forecast = modelo.predict(future)

    # Solo devolver fecha y valor estimado
    forecast = forecast[["ds", "yhat"]]
    forecast = forecast.rename(columns={"yhat": "precipitacion_estimada"})

    return forecast
