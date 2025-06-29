import pandas as pd

def preparar_datos_para_prediccion(df, meses_predecir=1):
    df['fecha'] = pd.to_datetime(df['fecha'], format="%Y/%m")
    df = df.sort_values('fecha')
    df = df[['fecha', 'valor']].copy()
    df['target'] = df['valor'].shift(-meses_predecir)
    df.dropna(inplace=True)
    return df
