import pandas as pd
import os

# Crear directorio de salida
output_dir = 'results/pa_analysis'
os.makedirs(output_dir, exist_ok=True)

# Leer el archivo Excel
df = pd.read_excel('data/imarpe/processed/BD_PA_2013-2023.xlsx', skiprows=3)

# Obtener la primera fila que contiene los nombres de las columnas
column_names = df.iloc[0]
df = df.iloc[1:].reset_index(drop=True)

# Renombrar columnas usando la primera fila
df.columns = ['REGION', 'AÑO', 'MES', 'UTILIZACION', 'REGION_2', 'LUGAR_DESEMBARQUE', 'AMBITO', 'ESPECIE', 'DESEMBARQUE_TM']

# Filtrar solo puertos marítimos y mostrar valores únicos
puertos_maritimos = df[df['AMBITO'] == 'MARITIMO']['LUGAR_DESEMBARQUE'].unique()
puertos_maritimos = sorted([p for p in puertos_maritimos if pd.notna(p) and p != ' -'])

# Leer coordenadas
coordenadas = pd.read_csv('data/imarpe/processed/coordenadas_puertos.csv')

# Crear DataFrame con puertos
df_puertos = pd.DataFrame(puertos_maritimos, columns=['PUERTO'])

# Unir con coordenadas
df_puertos = df_puertos.merge(coordenadas, on='PUERTO', how='left')

print("\nPuertos marítimos únicos con coordenadas:")
print(df_puertos)

print(f"\nTotal de puertos marítimos únicos: {len(puertos_maritimos)}")

# Guardar en CSV
df_puertos.to_csv(f'{output_dir}/puertos_maritimos_con_coordenadas.csv', index=False)
print(f"\nLista guardada en: {output_dir}/puertos_maritimos_con_coordenadas.csv") 