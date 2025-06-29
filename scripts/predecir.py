import pandas as pd
from joblib import load
import argparse
import os

def predecir(variable_objetivo, meses_adelante):
    # Ruta del modelo
    modelo_path = f"salidas/modelos/{variable_objetivo}_{meses_adelante}meses.pkl"
    if not os.path.exists(modelo_path):
        raise FileNotFoundError(f"No existe el modelo: {modelo_path}")

    # Cargar modelo entrenado
    modelo = load(modelo_path)
    print(f"📦 Modelo cargado: {modelo_path}")

    # Cargar últimos datos
    df = pd.read_csv("salidas/datos_unificados.csv")
    df = df.sort_values("fecha")
    ultima_fila = df.drop(columns=["fecha"]).iloc[-1:]

    # Realizar predicción
    prediccion = modelo.predict(ultima_fila)[0]
    print(f"🔮 Predicción para '{variable_objetivo}' dentro de {meses_adelante} meses: {prediccion:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realizar predicción con modelo entrenado")
    parser.add_argument("variable", help="Variable objetivo (ej: precipitacion, temperatura, velocidad_viento)")
    parser.add_argument("meses", type=int, help="Meses hacia el futuro (1 a 12)")
    args = parser.parse_args()
    predecir(args.variable, args.meses)
