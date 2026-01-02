import pandas as pd
import numpy as np
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os

# Crear directorio de salida
output_dir = 'results/global_elasticities'
os.makedirs(output_dir, exist_ok=True)

def load_production_data():
    """Cargar y agregar datos de producción por especie y fecha"""
    print("Cargando datos de producción...")
    df = pd.read_csv('data/imarpe/processed/produccion_mensual_matrix.csv')
    df['FECHA'] = pd.to_datetime(df['FECHA'])
    df.set_index('FECHA', inplace=True)
    
    # Obtener especies únicas
    species = ['BONITO', 'CABALLA', 'POTA', 'LISA', 'PERICO', 'JUREL', 'MERLUZA']
    
    # Agregar producción por especie
    production_by_species = pd.DataFrame()
    for sp in species:
        cols = [col for col in df.columns if sp in col]
        production_by_species[sp] = df[cols].sum(axis=1)
    
    return production_by_species

def load_sst_data():
    """Cargar datos de SST (mean, max, sum) y calcular promedio espacial"""
    print("\nCargando datos de SST...")
    
    # Cargar los tres tipos de anomalías
    ds_mean = xr.open_dataset('data/MODIS/processed/sst_anomaly_monthly_2002_2025.nc')
    ds_max = xr.open_dataset('data/MODIS/processed/sst_anomaly_monthly_max_2002_2025.nc')
    ds_sum = xr.open_dataset('data/MODIS/processed/sst_anomaly_monthly_sum_2002_2025.nc')
    
    # Calcular promedio espacial para cada tipo
    sst_mean = ds_mean.sst.mean(dim=['lat', 'lon']).to_dataframe()
    sst_max = ds_max.sst.mean(dim=['lat', 'lon']).to_dataframe()
    sst_sum = ds_sum.sst.mean(dim=['lat', 'lon']).to_dataframe()
    
    # Normalizar fechas a inicio de mes
    sst_mean.index = pd.to_datetime(sst_mean.index.strftime('%Y-%m-01'))
    sst_max.index = pd.to_datetime(sst_max.index.strftime('%Y-%m-01'))
    sst_sum.index = pd.to_datetime(sst_sum.index.strftime('%Y-%m-01'))
    
    return sst_mean, sst_max, sst_sum

def calculate_elasticity(X, y):
    """Calcular elasticidad y estadísticas usando regresión log-log"""
    # Asegurarnos de que los valores sean positivos
    X = np.abs(X) + 0.01
    y = y + 0.01
    
    # Transformación log
    X_log = np.log(X)
    y_log = np.log(y)
    
    # Realizar regresión
    reg = LinearRegression()
    reg.fit(X_log.reshape(-1, 1), y_log)
    
    # Calcular estadísticas
    r2 = reg.score(X_log.reshape(-1, 1), y_log)
    slope = reg.coef_[0]
    intercept = reg.intercept_
    
    # Calcular p-valor
    n = len(X_log)
    p = 1  # número de predictores
    dof = n - p - 1
    mse = np.sum((y_log - reg.predict(X_log.reshape(-1, 1))) ** 2) / dof
    var_b = mse / np.sum((X_log - np.mean(X_log)) ** 2)
    sd_b = np.sqrt(var_b)
    t_stat = slope / sd_b
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), dof))
    
    return {
        'elasticidad': slope,
        'intercepto': intercept,
        'r2': r2,
        'p_value': p_value,
        'n_obs': n
    }

def create_scatter_plots(prod_data, sst_data, sst_type, output_dir):
    """Crear gráficos de dispersión para cada especie"""
    for species in prod_data.columns:
        plt.figure(figsize=(10, 6))
        
        # Obtener datos comunes
        common_dates = prod_data.index.intersection(sst_data.index)
        X = sst_data.loc[common_dates, 'sst'].values
        y = prod_data.loc[common_dates, species].values
        
        # Calcular elasticidad
        stats = calculate_elasticity(X, y)
        
        # Crear scatter plot
        plt.scatter(np.log(np.abs(X) + 0.01), np.log(y + 0.01), alpha=0.5)
        
        # Añadir línea de regresión
        X_range = np.linspace(min(np.log(np.abs(X) + 0.01)), max(np.log(np.abs(X) + 0.01)), 100)
        plt.plot(X_range, stats['elasticidad'] * X_range + stats['intercepto'], 'r-')
        
        plt.title(f'{species} - SST {sst_type}\nElasticidad: {stats["elasticidad"]:.3f} (p={stats["p_value"]:.3e})')
        plt.xlabel('Log(SST Anomaly)')
        plt.ylabel('Log(Landings)')
        
        plt.savefig(f'{output_dir}/{species}_{sst_type}_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Cargar datos
    prod_data = load_production_data()
    sst_mean, sst_max, sst_sum = load_sst_data()
    
    # Calcular elasticidades para cada tipo de SST
    results = []
    sst_types = {
        'mean': sst_mean,
        'max': sst_max,
        'sum': sst_sum
    }
    
    print("\nCalculando elasticidades...")
    for sst_type, sst_data in sst_types.items():
        for species in prod_data.columns:
            # Obtener datos comunes
            common_dates = prod_data.index.intersection(sst_data.index)
            X = sst_data.loc[common_dates, 'sst'].values
            y = prod_data.loc[common_dates, species].values
            
            # Calcular elasticidad
            stats = calculate_elasticity(X, y)
            
            results.append({
                'especie': species,
                'tipo_sst': sst_type,
                'elasticidad': stats['elasticidad'],
                'r2': stats['r2'],
                'p_value': stats['p_value'],
                'n_obs': stats['n_obs']
            })
        
        # Crear gráficos de dispersión
        create_scatter_plots(prod_data, sst_data, sst_type, output_dir)
    
    # Guardar resultados
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{output_dir}/elasticidades_globales.csv', index=False)
    
    print("\nResultados de elasticidades globales:")
    print(results_df.round(3).to_string(index=False))
    
    # Crear gráfico comparativo
    plt.figure(figsize=(12, 6))
    sns.barplot(data=results_df, x='especie', y='elasticidad', hue='tipo_sst')
    plt.title('Elasticidades por Especie y Tipo de SST')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparacion_elasticidades.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main() 