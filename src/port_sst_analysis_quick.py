import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
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
    load_port_landing_data, 
    calculate_landing_sst_anomaly_regression
)

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

output_dir = 'results/port_sst_analysis_quick'
os.makedirs(f'{output_dir}/plots', exist_ok=True)
os.makedirs(f'{output_dir}/data', exist_ok=True)

sst_file = 'data/MODIS/processed/sst_anomaly_daily_2002_2025.nc'
print(f"Cargando datos de SST desde {sst_file}...")
ds = xr.open_dataset(sst_file)

print("Extrayendo series temporales de anomalías de SST para cada puerto...")
port_data_sst = extract_port_sst_series(ds, puertos_coords_subset)

print("Calculando percentiles para cada puerto...")
port_sst_percentiles = calculate_percentiles(port_data_sst)
pd.DataFrame(port_sst_percentiles).to_csv(f'{output_dir}/data/port_sst_percentiles.csv')

print("Cargando datos de producción...")
port_file = 'data/imarpe/processed/df_produccion_combined_2002_2024_clean.csv'
df_prod = load_port_landing_data(port_file)

percentile_key = '95th'

results_regression = calculate_landing_sst_anomaly_regression(df_prod, port_data_sst)
results_regression_df = pd.DataFrame(results_regression)
results_regression_df.to_csv(f'{output_dir}/data/results_regression.csv')

print("Creando matriz de anomalías (1=cierre, 0=abierto)...")

anomaly_matrix_binary, anomaly_matrix_ssta = create_anomaly_matrix(port_data_sst, port_sst_percentiles, percentile_key=percentile_key)
anomaly_matrix_binary.to_csv(f'{output_dir}/data/anomaly_matrix_binary_{percentile_key}.csv')
anomaly_matrix_ssta.to_csv(f'{output_dir}/data/anomaly_matrix_ssta.csv')

num_samples = 100
print(f"Creando {num_samples} muestras de la matriz de anomalías...")
anomaly_matrix_samples_binary, anomaly_matrix_samples_ssta = sample_with_replacement(anomaly_matrix_binary, anomaly_matrix_ssta, num_samples=num_samples, window_size=3, random_seed=42)
anomaly_matrix_samples_binary.to_csv(f'{output_dir}/data/anomaly_matrix_samples_binary_{percentile_key}.csv')
anomaly_matrix_samples_ssta.to_csv(f'{output_dir}/data/anomaly_matrix_samples_ssta.csv')

print("Calculando matriz de pérdidas por puerto...")
losses_matrix_daily_wages = calculate_losses_matrix_from_daily_wages(anomaly_matrix_samples_binary, port_losses)
losses_matrix_daily_wages.to_csv(f'{output_dir}/data/port_losses_matrix_daily_wages_{percentile_key}.csv')

losses_matrix_regression = calculate_losses_matrix_from_regression(anomaly_matrix_samples_ssta, results_regression, daily_price_per_ton=100)
losses_matrix_regression.to_csv(f'{output_dir}/data/port_losses_matrix_regression.csv')

print("Calculando pérdidas totales anuales...")
annual_losses_daily_wages = calculate_annual_total_losses(losses_matrix_daily_wages)
annual_losses_daily_wages.to_csv(f'{output_dir}/data/annual_losses_daily_wages_{percentile_key}.csv')

annual_losses_regression = calculate_annual_total_losses(losses_matrix_regression)
annual_losses_regression.to_csv(f'{output_dir}/data/annual_losses_regression.csv')

print("Calculando dias totales de calentamiento anuales")
annual_days_warming = calculate_annual_total_losses(anomaly_matrix_binary)
annual_days_warming.to_csv(f'{output_dir}/data/annual_days_warming_{percentile_key}.csv')


print("Creando gráfico de excedencia de pérdidas anuales...")
fig = plot_annual_losses(annual_losses_daily_wages)
fig.savefig(f'{output_dir}/plots/annual_losses_exceedance_daily_wages_{percentile_key}.png', dpi=300)
plt.close(fig)

fig = plot_annual_losses(annual_losses_regression)
fig.savefig(f'{output_dir}/plots/annual_losses_exceedance_regression.png', dpi=300)
plt.close(fig)

fig = plot_annual_losses(annual_days_warming)
fig.savefig(f'{output_dir}/plots/annual_days_warming.png', dpi=300)
plt.close(fig)

print(f"Análisis completo! Resultados guardados en {output_dir}/")
print(f"Archivos generados:")
print(f"- Matriz de anomalías binaria: {output_dir}/data/anomaly_matrix_binary_{percentile_key}.csv")
print(f"- Matriz de anomalías SST: {output_dir}/data/anomaly_matrix_ssta.csv")
print(f"- Matriz de anomalías SST muestras: {output_dir}/data/anomaly_matrix_ssta_samples.csv")
print(f"- Matriz de pérdidas por puerto (daily wages): {output_dir}/data/port_losses_matrix_daily_wages_{percentile_key}.csv")
print(f"- Matriz de pérdidas por puerto (regression): {output_dir}/data/port_losses_matrix_regression.csv")
print(f"- Pérdidas totales anuales (daily wages): {output_dir}/data/annual_losses_daily_wages_{percentile_key}.csv")
print(f"- Pérdidas totales anuales (regression): {output_dir}/data/annual_losses_regression.csv")
print(f"- Dias totales de calentamiento anuales: {output_dir}/data/annual_days_warming_{percentile_key}.csv")
