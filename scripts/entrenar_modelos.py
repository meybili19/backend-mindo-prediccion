import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import dump
import os
import argparse

def entrenar_modelo(variable_objetivo, meses_adelante):
    # Cargar datos
    df = pd.read_csv("salidas/datos_unificados.csv")
    df = df.sort_values("fecha")

    # Validar variable
    if variable_objetivo not in df.columns:
        raise Exception(f"La variable '{variable_objetivo}' no existe en los datos.")

    # Crear columna a predecir desplazada hacia el futuro
    columna_futuro = f"{variable_objetivo}_futuro"
    df[columna_futuro] = df[variable_objetivo].shift(-meses_adelante)
    df = df.dropna()

    # Separar datos
    X = df.drop(columns=["fecha", columna_futuro])
    y = df[columna_futuro]

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar modelo
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    # Guardar modelo
    modelo_path = f"salidas/modelos/{variable_objetivo}_{meses_adelante}meses.pkl"
    os.makedirs(os.path.dirname(modelo_path), exist_ok=True)
    dump(modelo, modelo_path)

    print(f"✅ Modelo entrenado y guardado en: {modelo_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de modelo climático")
    parser.add_argument("variable", help="Variable a predecir (ej: precipitacion, temperatura, velocidad_viento, direccion_viento)")
    parser.add_argument("meses", type=int, help="Meses a futuro (1 a 12)")

    args = parser.parse_args()
    entrenar_modelo(args.variable, args.meses)
