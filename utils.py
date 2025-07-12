import pandas as pd

def preparar_datos_para_prediccion(df, columna_valor='valor', meses_predecir=1):
    df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
    df = df.dropna(subset=['fecha'])
    df = df.sort_values('fecha')
    df = df[['fecha', columna_valor]].copy()
    df.rename(columns={columna_valor: 'valor'}, inplace=True)  # para que sea consistente
    df['target'] = df['valor'].shift(-meses_predecir)
    df.dropna(inplace=True)
    return df