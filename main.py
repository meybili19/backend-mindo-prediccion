from fastapi import FastAPI, Query
import joblib
from fastapi.middleware.cors import CORSMiddleware

from datetime import datetime
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VARIABLES_INFO = {
    "precipitacion": {
        "modelo": "modelos/modelo_precipitacion.pkl",
        "unidad": "mm"
    },
    "temperatura": {
        "modelo": "modelos/modelo_temperatura.pkl",
        "unidad": "°C"
    },
    "viento": {
        "modelo": "modelos/modelo_viento.pkl",
        "unidad": "m/s"
    },
    "direccion": {
        "modelo": "modelos/modelo_direccion.pkl",
        "unidad": "°"
    }
}

@app.get("/")
def root():
    return {"message": "Backend de predicciones de Mindo"}

@app.get("/api/predicciones")
def predecir(variable: str = Query(...), meses: int = Query(...)):
    if variable not in VARIABLES_INFO:
        return {"error": f"Variable no soportada. Usa una de: {list(VARIABLES_INFO.keys())}"}

    modelo_path = VARIABLES_INFO[variable]["modelo"]
    unidad = VARIABLES_INFO[variable]["unidad"]

    try:
        modelo = joblib.load(modelo_path)
    except FileNotFoundError:
        return {"error": f"No se encontró el modelo para {variable}"}

    # Valor inicial simulado
    ultimo_valor = 100.0

    predicciones = []
    fecha_base = datetime.today()

    for i in range(meses):
        # Predicción de todos los árboles
        todas_las_predicciones = [tree.predict([[ultimo_valor]])[0] for tree in modelo.estimators_]

        valor_predicho = sum(todas_las_predicciones) / len(todas_las_predicciones)
        desviacion = pd.Series(todas_las_predicciones).std()

        # Calculamos confianza aproximada (entre 0 y 100%)
        confianza = max(0, min(100, 100 - desviacion))

        fecha = pd.date_range(start=fecha_base, periods=meses, freq='ME')[i].strftime("%Y-%m")

        predicciones.append({
            "mes": fecha,
            "valor": round(valor_predicho, 2),
            "unidad": unidad,
            "confianza": round(confianza, 1)  # porcentaje estimado
        })

        ultimo_valor = valor_predicho

    return {
        "variable": variable,
        "unidad": unidad,
        "predicciones": predicciones
    }