import pandas as pd
import os

def preparar_datos():
    archivos = {
        "humedad_relativa": "DATOS_ETL_HUMEDAD RELATIVA.csv",
        "presion_atmosferica": "DATOS_ETL_PRESION ATMOSFERICA.csv",
        "temperatura": "DATOS_ETL_TEMPERATURA AMBIENTE.csv",
        "radiacion_solar": "DATOS_ETL_RADIACION SOLAR.csv",
        "precipitacion": "DATOS_ETL_PRECIPITACION.csv",
        "velocidad_viento": "DATOS_ETL_VELOCIDAD DEL VIENTO.csv",
        "direccion_viento": "DATOS_ETL_DIRECCION VIENTO.csv"
    }

    dfs = {}
    for clave, nombre_archivo in archivos.items():
        ruta = os.path.join("datos", nombre_archivo)
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No se encontró el archivo: {ruta}")
        
        df = pd.read_csv(ruta)
        if "direccion" in clave:
            df = df[["fecha", "direccion"]].rename(columns={"direccion": clave})
        else:
            df = df[["fecha", "valor"]].rename(columns={"valor": clave})
        dfs[clave] = df
        print(f"✅ {clave} cargado desde: {nombre_archivo}")

    # Unificación
    df_final = dfs["humedad_relativa"]
    for clave in archivos:
        if clave != "humedad_relativa":
            df_final = pd.merge(df_final, dfs[clave], on="fecha", how="inner")

    df_final["fecha"] = pd.to_datetime(df_final["fecha"], format="%Y/%m")
    df_final = df_final.sort_values("fecha")

    # Guardar resultado
    os.makedirs("salidas", exist_ok=True)
    salida = "salidas/datos_unificados.csv"
    df_final.to_csv(salida, index=False)
    print(f"\n📁 Datos unificados guardados en: {salida}")

if __name__ == "__main__":
    preparar_datos()
