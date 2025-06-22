import pandas as pd

def cargar_y_unir_datos():
    # Cargar archivos
    df_caudal = pd.read_csv("datos/caudal.csv")
    df_nivel = pd.read_csv("datos/nivel.csv")
    df_precipitacion = pd.read_csv("datos/precipitacion.csv")

    # Limpiar nombres de columnas
    df_caudal.columns = df_caudal.columns.str.strip().str.lower()
    df_nivel.columns = df_nivel.columns.str.strip().str.lower()
    df_precipitacion.columns = df_precipitacion.columns.str.strip().str.lower()

    # Convertir fechas
    df_caudal["fecha"] = pd.to_datetime(df_caudal["fecha"], errors='coerce')
    df_nivel["fecha"] = pd.to_datetime(df_nivel["fecha"], errors='coerce')
    df_precipitacion["fecha"] = pd.to_datetime(df_precipitacion["fecha"], errors='coerce')

    # Eliminar nulos y duplicados
    df_caudal = df_caudal.dropna(subset=["valor"]).drop_duplicates(subset="fecha")
    df_nivel = df_nivel.dropna(subset=["valor"]).drop_duplicates(subset="fecha")
    df_precipitacion = df_precipitacion.dropna(subset=["valor"]).drop_duplicates(subset="fecha")

    # Renombrar columnas
    df_caudal = df_caudal[["fecha", "valor"]].rename(columns={"valor": "caudal"})
    df_nivel = df_nivel[["fecha", "valor"]].rename(columns={"valor": "nivel"})
    df_precipitacion = df_precipitacion[["fecha", "valor"]].rename(columns={"valor": "precipitacion"})

    # Unir por fecha (inner join)
    df = pd.merge(df_caudal, df_nivel, on="fecha", how="inner")
    df = pd.merge(df, df_precipitacion, on="fecha", how="inner")

    # Ordenar cronológicamente
    df = df.sort_values("fecha").reset_index(drop=True)

    return df
