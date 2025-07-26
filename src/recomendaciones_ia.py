import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# Cargar las variables del archivo .env
load_dotenv()

# Configurar la API key de Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

modelo = genai.GenerativeModel("models/gemini-2.0-flash")

def generar_recomendaciones_operativas(df: pd.DataFrame, estacion="H44 Antisana") -> str:
    columnas = ["ds", "nivel_estimado"]
    if "precipitacion" in df.columns:
        columnas.append("precipitacion")

    resumen = df[columnas].tail(30).to_string(index=False)

    prompt = f"""
Eres un experto en gestión hídrica y control operativo de embalses.

A continuación tienes predicciones para la estación {estacion}, incluyendo nivel de agua estimado (en metros){' y precipitación esperada (en mm)' if 'precipitacion' in df.columns else ''} para los próximos 30 días:

{resumen}

Con base en estos datos, proporciona **3 a 5 recomendaciones operativas concretas y en lenguaje técnico natural**. Pueden incluir:

- apertura o cierre de compuertas
- monitoreo especial en fechas críticas
- advertencias por precipitaciones
- ajustes en extracción de agua
- recomendaciones a operadores humanos

Sé claro, específico y profesional.
"""

    respuesta = modelo.generate_content(prompt)
    return respuesta.text.strip()
