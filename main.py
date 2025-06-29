from fastapi import FastAPI, Query
from typing import Literal
import pandas as pd
from joblib import load
import os
import numpy as np

app = FastAPI(
    title="🎯 API de Predicción Climática",
    description="🌦️ Predice precipitación, temperatura, viento... ¡y explora los próximos meses!",
    version="2.0"
)

@app.get("/")
def root():
    return {"mensaje": "✅ Bienvenido a la API de predicción climática avanzada"}

@app.get("/predecir/")
def predecir(
    variable: Literal["precipitacion", "temperatura", "velocidad_viento", "direccion_viento"] = Query(...),
    meses: int = Query(1, ge=1, le=12)
):
    df = pd.read_csv("salidas/datos_unificados.csv").sort_values("fecha")
    ultima_fila = df.drop(columns=["fecha"]).iloc[-1:]
    resultados = []

    for mes in range(1, meses + 1):
        modelo_path = f"salidas/modelos/{variable}_{mes}meses.pkl"
        if not os.path.exists(modelo_path):
            resultados.append({
                "mes": mes,
                "estado": "❌ Modelo no encontrado",
                "modelo": modelo_path
            })
            continue

        modelo = load(modelo_path)

        # Obtener predicción + nivel de confianza
        predicciones_arboles = [arbol.predict(ultima_fila)[0] for arbol in modelo.estimators_]
        media = np.mean(predicciones_arboles)
        std_dev = np.std(predicciones_arboles)

        # Crear mensaje interactivo según valor
        if variable == "precipitacion":
            if media > 150:
                mensaje = "☔ ¡Mes muy lluvioso! Lleva paraguas y revisa el sistema de drenaje."
            elif media < 50:
                mensaje = "🌵 Probablemente seco. Ideal para obras o senderismo."
            else:
                mensaje = "🌤️ Lluvias moderadas, clima estable."
        elif variable == "temperatura":
            if media > 25:
                mensaje = "🔥 ¡Calor intenso! Hidratación y sombra."
            elif media < 10:
                mensaje = "❄️ Frío notable. Abrígate bien."
            else:
                mensaje = "🌡️ Temperatura moderada."
        else:
            mensaje = "📊 Predicción realizada con éxito."

        resultados.append({
            "mes": mes,
            "prediccion": round(media, 2),
            "confianza (±)": round(std_dev, 2),
            "mensaje": mensaje,
            "modelo": modelo_path
        })

    return {
        "variable": variable,
        "meses_solicitados": meses,
        "resultados": resultados
    }
