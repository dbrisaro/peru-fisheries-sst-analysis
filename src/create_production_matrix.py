import pandas as pd
import numpy as np
import os

# Crear directorio de salida
output_dir = 'data/imarpe/processed'
os.makedirs(output_dir, exist_ok=True)

# Cargar datos limpios
print("Cargando datos de producción limpios...")
df = pd.read_csv('data/imarpe/processed/produccion_mensual_limpia.csv')
df['FECHA'] = pd.to_datetime(df['FECHA'])

# Definir las 4 especies más importantes
top_species = ['BONITO', 'CABALLA', 'POTA', 'LISA', 'PERICO', 'JUREL', 'MERLUZA']

# Filtrar solo las especies de interés
df = df[df['ESPECIE'].isin(top_species)]

# Crear matriz pivote con fechas como índice, puertos como columnas y suma de desembarques como valores
print("\nCreando matriz de producción...")
production_matrix = df.pivot_table(
    index='FECHA',
    columns=['LUGAR_DESEMBARQUE', 'ESPECIE'],
    values='sum',
    fill_value=0
)

# Reorganizar columnas para tener puertos como columnas principales y especies como subcolumnas
production_matrix.columns = [f"{port}_{species}" for port, species in production_matrix.columns]

# Guardar matriz
output_file = f'{output_dir}/produccion_mensual_matrix.csv'
production_matrix.to_csv(output_file)
print(f"Matriz guardada en: {output_file}")

# Mostrar resumen
print("\nResumen de la matriz:")
print(f"Rango de fechas: {production_matrix.index.min()} a {production_matrix.index.max()}")
print(f"Número de puertos: {len(df['LUGAR_DESEMBARQUE'].unique())}")
print(f"Total de columnas: {len(production_matrix.columns)}")
print("\nPrimeras filas de la matriz:")
print(production_matrix.head())

# Calcular elasticidades promedio
print("\nCalculando elasticidades promedio...")
elasticities_df = pd.read_csv('results/port_sst_analysis_monthly/data/max_port_regression_results.csv')

# Filtrar por significancia y número mínimo de observaciones
elasticities_sig = elasticities_df[(elasticities_df['N_OBS'] >= 30) & (elasticities_df['P_VALUE'] < 0.05)]

# Calcular promedios ponderados por N_OBS
weighted_means = elasticities_sig.groupby('ESPECIE').apply(lambda x: np.average(x['ELASTICIDAD'], weights=x['N_OBS']))
std_dev = elasticities_sig.groupby('ESPECIE')['ELASTICIDAD'].std()
n_ports_sig = elasticities_sig.groupby('ESPECIE').size()
n_ports_total = elasticities_df[elasticities_df['N_OBS'] >= 30].groupby('ESPECIE').size()

# Crear DataFrame con resultados
elasticity_summary = pd.DataFrame({
    'Elasticidad_Promedio': weighted_means,
    'Desviacion_Estandar': std_dev,
    'Puertos_Significativos': n_ports_sig,
    'Total_Puertos': n_ports_total
})

# Guardar resultados
elasticity_file = f'{output_dir}/elasticidades_promedio.csv'
elasticity_summary.round(3).to_csv(elasticity_file)
print(f"\nElasticidades promedio guardadas en: {elasticity_file}")
print("\nResumen de elasticidades:")
print(elasticity_summary.round(3).to_string()) 