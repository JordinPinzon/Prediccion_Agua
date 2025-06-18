import pandas as pd

def cargar_y_unir_datos():
    # Cargar archivos
    df_caudal = pd.read_csv("datos/caudal.csv")
    df_nivel = pd.read_csv("datos/nivel.csv")

    # Limpiar
    df_caudal.columns = df_caudal.columns.str.strip()
    df_nivel.columns = df_nivel.columns.str.strip()
    df_caudal["fecha"] = pd.to_datetime(df_caudal["fecha"], errors='coerce')
    df_nivel["fecha"] = pd.to_datetime(df_nivel["fecha"], errors='coerce')

    df_caudal = df_caudal.dropna(subset=["valor"])
    df_nivel = df_nivel.dropna(subset=["valor"])

    df_caudal = df_caudal[["fecha", "valor"]].rename(columns={"valor": "caudal"})
    df_nivel = df_nivel[["fecha", "valor"]].rename(columns={"valor": "nivel"})

    # Unir por fecha
    df = pd.merge(df_caudal, df_nivel, on="fecha")
    return df
