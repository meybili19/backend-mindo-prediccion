from predictor import entrenar_y_guardar_modelo

variables = {
    "precipitacion": "datos/DATOS_ETL_PRECIPITACION.csv",
    "temperatura": "datos/DATOS_ETL_TEMPERATURA.csv",
    "viento": "datos/DATOS_ETL_VELOCIDAD_DEL_VIENTO.csv",
    "direccion": "datos/DATOS_ETL_DIRECCION_VIENTO.csv"
}

for variable, ruta in variables.items():
    modelo_file = f"modelos/modelo_{variable}.pkl"
    entrenar_y_guardar_modelo(variable, ruta, modelo_file)
