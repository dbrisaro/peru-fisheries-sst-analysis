import pandas as pd
import numpy as np
import os

output_dir = 'data/imarpe/processed'
os.makedirs(output_dir, exist_ok=True)

print("Cargando datos de producción...")
df = pd.read_excel('../data/imarpe/processed/BD_PA_2013-2023.xlsx', skiprows=3)
df.columns = ['REGION', 'AÑO', 'MES', 'UTILIZACION', 'REGION_2', 'LUGAR_DESEMBARQUE', 'AMBITO', 'ESPECIE', 'DESEMBARQUE_TM']
df = df.iloc[1:].reset_index(drop=True)

print("\nConvirtiendo columnas numéricas...")
df['DESEMBARQUE_TM'] = pd.to_numeric(df['DESEMBARQUE_TM'], errors='coerce')
df['AÑO'] = pd.to_numeric(df['AÑO'], errors='coerce')
df['MES'] = pd.to_numeric(df['MES'], errors='coerce')

print("\nFiltrando datos válidos...")
df = df.dropna(subset=['AÑO', 'MES', 'DESEMBARQUE_TM', 'LUGAR_DESEMBARQUE', 'ESPECIE'])

print("Filtrando puertos marítimos...")
df = df[df['AMBITO'] == 'MARITIMO'].copy()

df = df[~df['LUGAR_DESEMBARQUE'].isin(['OTROS', 'OTROS PUERTOS'])]

df['FECHA'] = pd.to_datetime(df['AÑO'].astype(int).astype(str) + '-' + 
                           df['MES'].astype(int).astype(str) + '-01')

print("\nResumen de datos:")
print(f"Total de registros: {len(df)}")
print("\nRegistros por especie principales:")
print(df['ESPECIE'].value_counts().head(15))
print("\nRegistros por puerto principales:")
print(df['LUGAR_DESEMBARQUE'].value_counts().head(10))
print("\nRango de fechas:")
print(f"Desde: {df['FECHA'].min()}")
print(f"Hasta: {df['FECHA'].max()}")

print("\nCalculando estadísticas mensuales...")
monthly_stats = df.groupby(['FECHA', 'ESPECIE', 'LUGAR_DESEMBARQUE'])['DESEMBARQUE_TM'].agg([
    'sum', 'count', 'mean', 'std'
]).reset_index()

# print("\nGuardando datos limpios...")
# output_file = f'{output_dir}/produccion_mensual_limpia.csv'
# monthly_stats.to_csv(output_file, index=False)
# print(f"Datos guardados en: {output_file}")

print("\nEjemplo de datos procesados:")
print(monthly_stats.head()) 