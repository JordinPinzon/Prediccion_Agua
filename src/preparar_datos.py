import pandas as pd

def cargar_y_unir_datos():
    # Cargar archivos
    df_caudal = pd.read_csv("datos/caudal.csv")
    df_nivel = pd.read_csv("datos/nivel.csv")
    df_precipitacion = pd.read_csv("datos/precipitacion.csv")

    # Limpiar nombres de columnas
    for df in [df_caudal, df_nivel, df_precipitacion]:
        df.columns = df.columns.str.strip().str.lower()
        df["fecha"] = pd.to_datetime(df["fecha"], errors='coerce')
        df.drop_duplicates(subset="fecha", inplace=True)
        df.sort_values("fecha", inplace=True)

    # Renombrar columnas
    df_caudal = df_caudal[["fecha", "valor"]].rename(columns={"valor": "caudal"})
    df_nivel = df_nivel[["fecha", "valor"]].rename(columns={"valor": "nivel"})
    df_precipitacion = df_precipitacion[["fecha", "valor"]].rename(columns={"valor": "precipitacion"})

    # Merge completo con outer join
    df = pd.merge(df_caudal, df_nivel, on="fecha", how="outer")
    df = pd.merge(df, df_precipitacion, on="fecha", how="outer")

    # Ordenar por fecha
    df = df.sort_values("fecha").reset_index(drop=True)
    df.set_index("fecha", inplace=True)

    # Imputar datos de forma segura
    df.interpolate(method="time", inplace=True)  # Interpola suavemente
    df.fillna(method="ffill", limit=3, inplace=True)  # Solo rellena huecos pequeños hacia adelante
    df.fillna(method="bfill", limit=3, inplace=True)  # Lo mismo hacia atrás

    # Eliminar fechas que aún tengan NaN (grandes huecos que no deben interpolarse)
    df = df.dropna()

    # Resetear índice
    df = df.reset_index()

    return df
