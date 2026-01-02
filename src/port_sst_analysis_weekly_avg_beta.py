import os
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats
from port_sst_analysis_functions import (
    extract_port_sst_series,
    calculate_percentiles,
    create_anomaly_matrix,
    sample_with_replacement,
    create_port_location_map,
    calculate_losses_matrix_from_daily_wages,
    calculate_losses_matrix_from_regression,
    calculate_annual_total_losses,
    plot_annual_losses, 
    load_port_landing_data
)

def calculate_average_beta(df_prod, port_data_sst):
    """
    Calculate average regression coefficient (beta) across all ports.
    
    Parameters
    ----------
    df_prod : pandas.DataFrame
        DataFrame with landing data for each port
    port_data_sst : dict or pandas.DataFrame
        Dictionary with SST data for each port or DataFrame with SST data
        
    Returns
    -------
    float
        Average beta coefficient across all ports
    """
    all_betas = []
    
    if isinstance(port_data_sst, pd.DataFrame):
        for port_name in port_data_sst.columns:
            sst_series = port_data_sst[port_name]
            data_prod = pd.Series(df_prod[port_name], name='production')

            start_date = max(data_prod.index.min(), sst_series.index.min())
            end_date = min(data_prod.index.max(), sst_series.index.max())

            data_prod = data_prod.loc[start_date:end_date]
            sst_series = sst_series.loc[start_date:end_date]

            df_combined = pd.DataFrame({
                'production': data_prod,
                'sst_anomaly': sst_series
            })

            data = df_combined[['production', 'sst_anomaly']].dropna()
            data = data[data['production'] != 0]

            slope, _, _, _, _ = stats.linregress(
                data['sst_anomaly'], data['production'])
            
            all_betas.append(slope)
    else:
        for port_name, port_info in port_data_sst.items():
            sst_series = port_info['sst_series']
            data_prod = pd.Series(df_prod[port_name], name='production')

            start_date = max(data_prod.index.min(), sst_series.index.min())
            end_date = min(data_prod.index.max(), sst_series.index.max())

            data_prod = data_prod.loc[start_date:end_date]
            sst_series = sst_series.loc[start_date:end_date]

            df_combined = pd.DataFrame({
                'production': data_prod,
                'sst_anomaly': sst_series
            })

            data = df_combined[['production', 'sst_anomaly']].dropna()
            data = data[data['production'] != 0]

            slope, _, _, _, _ = stats.linregress(
                data['sst_anomaly'], data['production'])
            
            all_betas.append(slope)
    
    return np.mean(all_betas)

def resample_to_weekly(sst_data):
    """
    Resample los datos de SST a frecuencia semanal.
    
    Parameters
    ----------
    sst_data : dict
        Diccionario con series temporales de anomalías de SST para cada puerto
        
    Returns
    -------
    pandas.DataFrame
        DataFrame con anomalías de SST semanales
    """
    # Obtener el índice temporal de la primera serie
    first_port = list(sst_data.keys())[0]
    first_series = sst_data[first_port]['sst_series']
    
    # Crear DataFrame con el índice temporal correcto
    df = pd.DataFrame(index=first_series.index)
    for port, data in sst_data.items():
        df[port] = data['sst_series'].values
    
    # Resample a frecuencia semanal (promedio)
    weekly_data = df.resample('W').mean()
    return weekly_data

def resample_landings_to_weekly(landings_data):
    """
    Resample los datos de desembarques a frecuencia semanal.
    
    Parameters
    ----------
    landings_data : pandas.DataFrame
        DataFrame con datos de desembarques diarios
        
    Returns
    -------
    pandas.DataFrame
        DataFrame con datos de desembarques semanales
    """
    # Resample a frecuencia semanal (suma)
    weekly_data = landings_data.resample('W').sum()
    return weekly_data

puertos_coords_subset = {
    "Paita": (-5.0892, -81.1144),
    "Callao": (-12.0432, -77.1469),
    "Parachique": (-5.5745, -80.9006),
    "Chicama": (-7.8432, -79.4000),
    "Samanco": (-9.1845, -78.4942),
    "Casma": (-9.4549, -78.3854),
    "Huarmey": (-10.0965, -78.1716),
    "Supe": (-10.7979, -77.7094),
    "Vegueta": (-11.0281, -77.6489),
    "Huacho": (-11.1067, -77.6056),
    "Tambo de Mora": (-13.4712, -76.1932),
    "Atico": (-16.2101, -73.6111),
    "Mollendo": (-17.0231, -72.0145),
    "Ilo": (-17.6394, -71.3374)
}

port_losses = {
    "Paita": 74396.4,
    "Callao": 33031.6,
    "Parachique": 58683.8,
    "Chicama": 33031.6,
    "Samanco": 33031.6,
    "Casma": 22263.0,
    "Huarmey": 33031.6,
    "Supe": 33031.6,
    "Vegueta": 33031.6,
    "Huacho": 49896.0,
    "Tambo de Mora": 55489.5,
    "Atico": 1720.0,
    "Mollendo": 33031.6,
    "Ilo": 8004.0
}

output_dir = 'results/port_sst_analysis_weekly_avg_beta'
os.makedirs(f'{output_dir}/plots', exist_ok=True)
os.makedirs(f'{output_dir}/data', exist_ok=True)

# Cargar y procesar datos de SST
sst_file = 'data/MODIS/processed/sst_anomaly_daily_2002_2025.nc'
print(f"Cargando datos de SST desde {sst_file}...")
ds = xr.open_dataset(sst_file)

print("Extrayendo series temporales de anomalías de SST para cada puerto...")
port_data_sst = extract_port_sst_series(ds, puertos_coords_subset)

print("Resampleando datos de SST a frecuencia semanal...")
port_data_sst_weekly = resample_to_weekly(port_data_sst)

print("Calculando percentiles para cada puerto (datos semanales)...")
port_sst_percentiles = calculate_percentiles(port_data_sst_weekly)
pd.DataFrame(port_sst_percentiles).to_csv(f'{output_dir}/data/port_sst_percentiles_weekly.csv')

# Cargar y procesar datos de producción
print("Cargando datos de producción...")
port_file = 'data/imarpe/processed/df_produccion_combined_2002_2024_clean.csv'
df_prod = load_port_landing_data(port_file)

print("Resampleando datos de producción a frecuencia semanal...")
df_prod_weekly = resample_landings_to_weekly(df_prod)

percentile_key = '95th'

# Calcular beta promedio
print("Calculando beta promedio para todos los puertos...")
avg_beta = calculate_average_beta(df_prod_weekly, port_data_sst_weekly)
print(f"Beta promedio: {avg_beta:.4f}")

# Guardar el beta promedio
pd.DataFrame({'avg_beta': [avg_beta]}).to_csv(f'{output_dir}/data/avg_beta.csv')

print("Creando matriz de anomalías semanales (1=cierre, 0=abierto)...")
anomaly_matrix_binary, anomaly_matrix_ssta = create_anomaly_matrix(port_data_sst_weekly, port_sst_percentiles, percentile_key=percentile_key)
anomaly_matrix_binary.to_csv(f'{output_dir}/data/anomaly_matrix_binary_weekly_{percentile_key}.csv')
anomaly_matrix_ssta.to_csv(f'{output_dir}/data/anomaly_matrix_ssta_weekly.csv')

num_samples = 100
print(f"Creando {num_samples} muestras de la matriz de anomalías...")
anomaly_matrix_samples_binary, anomaly_matrix_samples_ssta = sample_with_replacement(anomaly_matrix_binary, anomaly_matrix_ssta, num_samples=num_samples, window_size=3, random_seed=42)
anomaly_matrix_samples_binary.to_csv(f'{output_dir}/data/anomaly_matrix_samples_binary_weekly_{percentile_key}.csv')
anomaly_matrix_samples_ssta.to_csv(f'{output_dir}/data/anomaly_matrix_samples_ssta_weekly.csv')

print("Calculando matriz de pérdidas por puerto...")
losses_matrix_daily_wages = calculate_losses_matrix_from_daily_wages(anomaly_matrix_samples_binary, port_losses)
losses_matrix_daily_wages.to_csv(f'{output_dir}/data/port_losses_matrix_daily_wages_weekly_{percentile_key}.csv')

# Crear diccionario con el beta promedio para todos los puertos
avg_beta_dict = {port: {'slope': avg_beta, 'intercept': 0} for port in puertos_coords_subset.keys()}
losses_matrix_regression = calculate_losses_matrix_from_regression(anomaly_matrix_samples_ssta, avg_beta_dict, daily_price_per_ton=100)
losses_matrix_regression.to_csv(f'{output_dir}/data/port_losses_matrix_regression_weekly.csv')

print("Calculando pérdidas totales anuales...")
annual_losses_daily_wages = calculate_annual_total_losses(losses_matrix_daily_wages)
annual_losses_daily_wages.to_csv(f'{output_dir}/data/annual_losses_daily_wages_weekly_{percentile_key}.csv')

annual_losses_regression = calculate_annual_total_losses(losses_matrix_regression)
annual_losses_regression.to_csv(f'{output_dir}/data/annual_losses_regression_weekly.csv')

print("Calculando semanas totales de calentamiento anuales")
annual_weeks_warming = calculate_annual_total_losses(anomaly_matrix_binary)
annual_weeks_warming.to_csv(f'{output_dir}/data/annual_weeks_warming_weekly_{percentile_key}.csv')

print("Creando gráficos de excedencia...")
fig = plot_annual_losses(annual_losses_daily_wages)
fig.savefig(f'{output_dir}/plots/annual_losses_exceedance_daily_wages_weekly_{percentile_key}.png', dpi=300)
plt.close(fig)

fig = plot_annual_losses(annual_losses_regression)
fig.savefig(f'{output_dir}/plots/annual_losses_exceedance_regression_weekly.png', dpi=300)
plt.close(fig)

fig = plot_annual_losses(annual_weeks_warming)
fig.savefig(f'{output_dir}/plots/annual_weeks_warming_weekly.png', dpi=300)
plt.close(fig)

print(f"Análisis completo! Resultados guardados en {output_dir}/")
print(f"Archivos generados:")
print(f"- Beta promedio: {output_dir}/data/avg_beta.csv")
print(f"- Matriz de anomalías binaria semanal: {output_dir}/data/anomaly_matrix_binary_weekly_{percentile_key}.csv")
print(f"- Matriz de anomalías SST semanal: {output_dir}/data/anomaly_matrix_ssta_weekly.csv")
print(f"- Matriz de anomalías SST muestras semanales: {output_dir}/data/anomaly_matrix_ssta_samples_weekly.csv")
print(f"- Matriz de pérdidas por puerto semanal (daily wages): {output_dir}/data/port_losses_matrix_daily_wages_weekly_{percentile_key}.csv")
print(f"- Matriz de pérdidas por puerto semanal (regression): {output_dir}/data/port_losses_matrix_regression_weekly.csv")
print(f"- Pérdidas totales anuales semanales (daily wages): {output_dir}/data/annual_losses_daily_wages_weekly_{percentile_key}.csv")
print(f"- Pérdidas totales anuales semanales (regression): {output_dir}/data/annual_losses_regression_weekly.csv")
print(f"- Semanas totales de calentamiento anuales: {output_dir}/data/annual_weeks_warming_weekly_{percentile_key}.csv") 