import os
import pandas as pd
import xarray as xr
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from port_sst_analysis_functions import (
    find_nearest_point,
    extract_port_sst_series,
    calculate_percentiles
)
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

plt.style.use('default')
sns.set_theme(style="whitegrid")

output_dir = 'results/port_sst_analysis_monthly'
os.makedirs(f'{output_dir}/plots', exist_ok=True)
os.makedirs(f'{output_dir}/data', exist_ok=True)

print("Cargando datos de puertos...")
puertos_df = pd.read_csv('results/pa_analysis/puertos_maritimos_con_coordenadas.csv')
puertos_df = puertos_df[(puertos_df['LATITUD'] != 0) & (puertos_df['LONGITUD'] != 0)]
puertos_coords = dict(zip(puertos_df['PUERTO'], zip(puertos_df['LATITUD'], puertos_df['LONGITUD'])))

print("Cargando datos de SST...")
sst_max_file = 'data/MODIS/processed/sst_anomaly_monthly_sum_2002_2025.nc'
sst_mean_file = 'data/MODIS/processed/sst_anomaly_monthly_2002_2025.nc'

ds_max = xr.open_dataset(sst_max_file)
ds_mean = xr.open_dataset(sst_mean_file)

print("\nVariables en archivo max:")
print(ds_max.variables)
print("\nVariables en archivo mean:")
print(ds_mean.variables)

print("\nCargando datos de producción...")
df_prod = pd.read_csv('data/imarpe/processed/produccion_mensual_matrix.csv', index_col=0, parse_dates=True)

df_prod.index = pd.to_datetime(df_prod.index.strftime('%Y-%m-01'))

print("\nRango de fechas en datos de producción:")
print(f"Desde: {df_prod.index.min()}")
print(f"Hasta: {df_prod.index.max()}")
print(f"Número total de columnas: {len(df_prod.columns)}")

print("\nExtrayendo series de SST...")
port_data_max = extract_port_sst_series(ds_max, puertos_coords)
port_data_mean = extract_port_sst_series(ds_mean, puertos_coords)

sst_max_series = pd.DataFrame({port: data['sst_series'] for port, data in port_data_max.items()})
sst_mean_series = pd.DataFrame({port: data['sst_series'] for port, data in port_data_mean.items()})

sst_max_series.index = pd.to_datetime(sst_max_series.index.strftime('%Y-%m-01'))
sst_mean_series.index = pd.to_datetime(sst_mean_series.index.strftime('%Y-%m-01'))

print("\nRango de fechas en datos de SST:")
print(f"Desde: {sst_max_series.index.min()}")
print(f"Hasta: {sst_max_series.index.max()}")
print(f"Número de puertos con datos SST: {len(sst_max_series.columns)}")

print("\nVerificando mapeo de puertos...")
puertos_prod = set([col.split('_')[0] for col in df_prod.columns])
puertos_sst = set(sst_max_series.columns)
print(f"Puertos en datos de producción: {len(puertos_prod)}")
print(f"Puertos en datos de SST: {len(puertos_sst)}")
print(f"Puertos en producción sin datos SST: {len(puertos_prod - puertos_sst)}")
if len(puertos_prod - puertos_sst) > 0:
    print("Puertos sin datos SST:")
    for puerto in sorted(puertos_prod - puertos_sst):
        print(f"  - {puerto}")

print("\nCalculando percentiles...")
sst_max_percentiles = calculate_percentiles(sst_max_series)
sst_mean_percentiles = calculate_percentiles(sst_mean_series)

pd.DataFrame(sst_max_percentiles).to_csv(f'{output_dir}/data/sst_max_percentiles.csv')
pd.DataFrame(sst_mean_percentiles).to_csv(f'{output_dir}/data/sst_mean_percentiles.csv')

def perform_port_regressions(sst_series, df_prod, output_prefix):
    results = []
    
    species = ['BONITO', 'CABALLA', 'POTA', 'LISA', 'PERICO', 'JUREL', 'MERLUZA']
    
    for especie in species:
        print(f"\nAnalizando especie: {especie}")
        
        especie_cols = [col for col in df_prod.columns if especie in col]
        if not especie_cols:
            print(f"  No hay datos para {especie}")
            continue
            
        puertos_especie = [col.split('_')[0] for col in especie_cols]
        puertos_con_sst = [p for p in puertos_especie if p in sst_series.columns]
        
        if not puertos_con_sst:
            print(f"  No hay datos de SST para puertos de {especie}")
            continue
            
        print(f"  Número de puertos con datos: {len(puertos_con_sst)}")
        
        plt.figure(figsize=(15, 10))
        colors = plt.cm.viridis(np.linspace(0, 1, len(puertos_con_sst)))
        
        for puerto, color in zip(puertos_con_sst, colors):
            sst_puerto = sst_series[puerto]
            
            prod_col = f"{puerto}_{especie}"
            prod_puerto = df_prod[prod_col]
            
            common_dates = prod_puerto.index.intersection(sst_puerto.index)
            
            valid_mask = ~(np.isnan(sst_puerto[common_dates]) | 
                         np.isnan(prod_puerto[common_dates]) |
                         (prod_puerto[common_dates] == 0))
            
            X = sst_puerto[common_dates][valid_mask].values
            y = prod_puerto[common_dates][valid_mask].values
            
            if len(X) < 30:  # Requerir al menos 30 puntos de datos
                continue
                
            X = np.abs(X) + 0.01  # Agregamos 0.01 para evitar ceros
            y = y + 0.01  # Agregamos 0.01 para evitar ceros
            
            X_log = np.log(X).reshape(-1, 1)
            y_log = np.log(y)
            
            reg = LinearRegression()
            reg.fit(X_log, y_log)
            
            r2 = reg.score(X_log, y_log)
            slope = reg.coef_[0]
            intercept = reg.intercept_
            
            n = len(X_log)
            p = 1  # número de predictores
            dof = n - p - 1
            mse = np.sum((y_log - reg.predict(X_log)) ** 2) / dof
            var_b = mse / np.sum((X_log - np.mean(X_log)) ** 2)
            sd_b = np.sqrt(var_b)
            t_stat = slope / sd_b
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), dof))
            
            results.append({
                'ESPECIE': especie,
                'PUERTO': puerto,
                'ELASTICIDAD': slope,
                'INTERCEPT': intercept,
                'R2': r2,
                'P_VALUE': p_value,
                'N_OBS': n
            })
            
            plt.scatter(X_log, y_log, alpha=0.5, color=color, label=puerto)
            
            x_range = np.linspace(min(X_log), max(X_log), 100)
            plt.plot(x_range, reg.predict(x_range), color=color, linewidth=1, alpha=0.7)
            
            equation = f'{puerto}: y = {slope:.3f}x + {intercept:.3f} (R²={r2:.3f})'
            plt.text(0.05, 0.95 - (0.05 * list(puertos_con_sst).index(puerto)), 
                    equation, transform=plt.gca().transAxes, 
                    fontsize=8, verticalalignment='top', color=color)
        
        plt.xlabel('Log(Anomalía SST) (°C)', fontsize=12)
        plt.ylabel('Log(Desembarque)', fontsize=12)
        plt.title(f'{especie}\nRelación SST-Desembarque por Puerto', 
                 fontsize=14, pad=20)
        
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/plots/{output_prefix}_{especie}_all_ports_scatter.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'{output_dir}/data/{output_prefix}_port_regression_results.csv', index=False)
        return results_df
    return None

def create_elasticity_maps(results_df, puertos_df, output_dir):
    """Create maps showing elasticity values by port for each species."""
    maps_dir = os.path.join(output_dir, 'maps')
    os.makedirs(maps_dir, exist_ok=True)
    
    species_list = results_df['ESPECIE'].unique()
    
    results_df = results_df[results_df['N_OBS'] >= 30]
    
    vmin = -.5
    vmax = .5
    
    lat_min, lat_max = -20, -3  # Approximate limits for Peru
    lon_min, lon_max = -85, -67  # Approximate limits for Peru's coast
    
    for species in species_list:
        species_results = results_df[results_df['ESPECIE'] == species]
        
        fig, ax = plt.subplots(figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
        
        ax.add_feature(cfeature.LAND)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        lons = []
        lats = []
        elasticities = []
        
        for _, row in species_results.iterrows():
            port_info = puertos_df[puertos_df['PUERTO'] == row['PUERTO']].iloc[0]
            lons.append(port_info['LONGITUD'])
            lats.append(port_info['LATITUD'])
            elasticities.append(row['ELASTICIDAD'])
        
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.RdBu_r
        
        scatter = ax.scatter(lons, lats, c=elasticities, cmap=cmap, norm=norm,
                           s=100, transform=ccrs.PlateCarree())
        
        cbar = plt.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.05)
        cbar.set_label('Elasticidad')
        
        plt.title(f'Elasticidad SST Acumulada - {species}')
        
        plt.savefig(os.path.join(maps_dir, f'elasticity_map_{species}_sum.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

print("\nRealizando regresiones por puerto...")
max_port_results = perform_port_regressions(sst_max_series, df_prod, 'max')
mean_port_results = perform_port_regressions(sst_mean_series, df_prod, 'mean')

print("\nCreando mapas de elasticidad...")
create_elasticity_maps(max_port_results, puertos_df, output_dir)

print("\nAnálisis completado. Resultados guardados en:", output_dir)

if max_port_results is not None and mean_port_results is not None:
    print("\nResultados con SST acumulada (por puerto):")
    print(max_port_results.to_string(index=False))
    print("\nResultados con SST media (por puerto):")
    print(mean_port_results.to_string(index=False))

def main():
    port_data = load_port_data()
    sst_max, sst_mean = load_sst_data()
    production_data = load_production_data()
    
    sst_series_max, sst_series_mean = extract_sst_series(port_data, sst_max, sst_mean)
    
    percentiles_max, percentiles_mean = calculate_percentiles(sst_series_max, sst_series_mean)
    
    results_dict = perform_port_regressions(port_data, production_data, sst_series_max, sst_series_mean)
    
    create_elasticity_maps(results_dict, puertos_df, 'results/port_sst_analysis_monthly')
    
    save_results(results_dict, 'results/port_sst_analysis_monthly') 