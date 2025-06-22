from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def predecir_nivel(forecast, df_original):
    df = df_original.copy()
    
    # Extraer variables estacionales
    df["mes"] = df["fecha"].dt.month
    df["semana"] = df["fecha"].dt.isocalendar().week
    df["año"] = df["fecha"].dt.year
    df["dias_desde_inicio"] = (df["fecha"] - df["fecha"].min()).dt.days

    # Lags (valores anteriores)
    df["caudal_lag1"] = df["caudal"].shift(1)
    df["precipitacion"] = df.get("precipitacion", 0)
    df["precipitacion_lag1"] = df["precipitacion"].shift(1)

    # Interacciones
    df["caudal_x_mes"] = df["caudal"] * df["mes"]
    df["precipitacion_x_semana"] = df["precipitacion"] * df["semana"]

    # Eliminar filas nulas por lags
    df = df.dropna()

    # Entrenamiento
    features = ["caudal", "precipitacion", "mes", "semana", "año", "dias_desde_inicio",
                "caudal_lag1", "precipitacion_lag1", "caudal_x_mes", "precipitacion_x_semana"]
    X = df[features]
    y = df["nivel"]

    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X, y)

    # Forecast
    forecast_temp = forecast.copy()
    forecast_temp = forecast_temp.rename(columns={"yhat": "caudal"})

    forecast_temp["mes"] = forecast_temp["ds"].dt.month
    forecast_temp["semana"] = forecast_temp["ds"].dt.isocalendar().week
    forecast_temp["año"] = forecast_temp["ds"].dt.year
    forecast_temp["dias_desde_inicio"] = (forecast_temp["ds"] - df["fecha"].min()).dt.days

    forecast_temp["caudal_lag1"] = forecast_temp["caudal"].shift(1)
    forecast_temp["precipitacion"] = 0
    forecast_temp["precipitacion_lag1"] = 0

    forecast_temp["caudal_x_mes"] = forecast_temp["caudal"] * forecast_temp["mes"]
    forecast_temp["precipitacion_x_semana"] = forecast_temp["precipitacion"] * forecast_temp["semana"]

    forecast_temp = forecast_temp.fillna(0)

    forecast["nivel_estimado"] = modelo.predict(forecast_temp[features])

    return forecast
