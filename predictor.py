import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
from utils import preparar_datos_para_prediccion

def entrenar_y_guardar_modelo(nombre_variable, archivo_csv, output_modelo):
    df = pd.read_csv(archivo_csv)
    data = preparar_datos_para_prediccion(df)

    X = data[['valor']]
    y = data['target']

    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X, y)

    score = modelo.score(X, y)
    print(f"✅ Modelo '{nombre_variable}' entrenado. Precisión (R²): {score:.4f}")

    joblib.dump(modelo, output_modelo)
    print(f"✅ Modelo guardado: {output_modelo}")
