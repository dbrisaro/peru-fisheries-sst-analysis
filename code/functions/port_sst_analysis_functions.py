import os
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
import random
from matplotlib.ticker import MaxNLocator
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
import itertools

def find_nearest_point(ds, lat, lon):
    """
    Find the nearest grid point to a given lat/lon in a dataset.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing lat and lon coordinates
    lat : float
        Target latitude
    lon : float
        Target longitude
    
    Returns:
    --------
    dict
        Dictionary containing nearest lat, lon values and their indices
    """
    abs_lat = np.abs(ds.lat.values - lat)
    abs_lon = np.abs(ds.lon.values - lon)
    
    lat_idx = abs_lat.argmin()
    lon_idx = abs_lon.argmin()
    
    return {
        'lat': ds.lat.values[lat_idx],
        'lon': ds.lon.values[lon_idx],
        'lat_idx': lat_idx,
        'lon_idx': lon_idx
    }

def extract_port_sst_series(ds, port_coords):
    """
    Extract SST anomaly time series for each port.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing SST anomaly data
    port_coords : dict
        Dictionary of port names and their coordinates (lat, lon)
    
    Returns:
    --------
    dict
        Dictionary containing port data including SST time series and coordinates
    """
    port_data_sst = {}
    
    for port_name, (lat, lon) in port_coords.items():
        print(f"Processing {port_name}...")
        
        nearest = find_nearest_point(ds, lat, lon)
        
        sst_series = ds.sst.sel(
            lat=nearest['lat'],
            lon=nearest['lon'],
            method='nearest'
        )
        
        time_index = pd.to_datetime(ds.time.values)
        sst_df = pd.Series(sst_series.values, index=time_index, name='sst_anomaly')
        
        port_data_sst[port_name] = {
            'sst_series': sst_df,
            'nearest_lat': nearest['lat'],
            'nearest_lon': nearest['lon'],
            'original_lat': lat,
            'original_lon': lon,
            'lat_idx': nearest['lat_idx'],
            'lon_idx': nearest['lon_idx']
        }
    
    return port_data_sst

def calculate_percentiles(port_data_sst):
    """
    Calculate percentiles for each port's SST anomaly time series.
    
    Parameters:
    -----------
    port_data_sst : dict or pandas.DataFrame
        Dictionary containing port data including SST time series or DataFrame with SST data
    
    Returns:
    --------
    dict
        Dictionary containing percentiles for each port
    """
    port_percentiles = {}
    
    if isinstance(port_data_sst, pd.DataFrame):
        # Si es DataFrame, calcular percentiles para cada columna
        for port_name in port_data_sst.columns:
            sst_values = port_data_sst[port_name].values
            percentiles = {
                '80th': np.nanpercentile(sst_values, 80),
                '90th': np.nanpercentile(sst_values, 90),
                '95th': np.nanpercentile(sst_values, 95)
            }
            port_percentiles[port_name] = percentiles
    else:
        # Si es diccionario, mantener el comportamiento original
        for port_name, data in port_data_sst.items():
            sst_df = data['sst_series']
            percentiles = {
                '80th': np.nanpercentile(sst_df.values, 80),
                '90th': np.nanpercentile(sst_df.values, 90),
                '95th': np.nanpercentile(sst_df.values, 95)
            }
            port_percentiles[port_name] = percentiles
    
    return port_percentiles

def create_distribution_plot(port_name, sst_series, percentiles, output_dir):
    """
    Create a distribution plot for a port's SST anomaly time series.
    
    Parameters:
    -----------
    port_name : str
        Name of the port
    sst_series : pandas.Series
        SST anomaly time series
    percentiles : dict
        Dictionary containing percentile values
    output_dir : str
        Directory to save the plot
    """
    fig = plt.figure(figsize=(10, 6))
    
    ax = plt.axes([0.05, 0.05, 0.9, 0.9])
    sns.histplot(sst_series.dropna(), kde=True, ax=ax)
    
    ax.axvline(percentiles['80th'], color='orange', linestyle='--', 
                label=f'80th percentile: {percentiles["80th"]:.2f}°C')
    ax.axvline(percentiles['90th'], color='red', linestyle='--', 
                label=f'90th percentile: {percentiles["90th"]:.2f}°C')
    ax.axvline(percentiles['95th'], color='darkred', linestyle='--', 
                label=f'95th percentile: {percentiles["95th"]:.2f}°C')
    
    ax.set_title(f'SST Anomaly Distribution for {port_name}', loc='left', fontsize=10)
    ax.set_xlabel('SST Anomaly (°C)', fontsize=8)
    ax.set_ylabel('Frequency', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-4, 4)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.savefig(f'{output_dir}/plots/{port_name}_sst_distribution.png', dpi=300)
    plt.close()

def create_port_location_map(port_data_sst, output_dir):
    """
    Create a map showing the locations of all ports and the exact points where SST data is extracted.
    
    Parameters:
    -----------
    port_data_sst : dict
        Dictionary containing port data including coordinates
    output_dir : str
        Directory to save the map
    """
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.COASTLINE, linewidth=1.2)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    
    ax.add_feature(cfeature.RIVERS, linewidth=0.5, edgecolor='blue', alpha=0.5)
    
    min_lon = min([data['original_lon'] for data in port_data_sst.values()]) - 3
    max_lon = max([data['original_lon'] for data in port_data_sst.values()]) + 3
    min_lat = min([data['original_lat'] for data in port_data_sst.values()]) - 3
    max_lat = max([data['original_lat'] for data in port_data_sst.values()]) + 3
    
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())
    
    for port_name, data in port_data_sst.items():
        ax.plot(data['original_lon'], data['original_lat'], '.', color='black', markersize=2, 
                transform=ccrs.PlateCarree(), zorder=5)
        
        ax.plot(data['nearest_lon'], data['nearest_lat'], '.', color='red', markersize=2, 
                transform=ccrs.PlateCarree(), zorder=5)
        
        ax.plot([data['original_lon'], data['nearest_lon']], 
                [data['original_lat'], data['nearest_lat']], 
                'k-', linewidth=1.0, alpha=0.7, transform=ccrs.PlateCarree(), zorder=4)
        
        ax.text(data['original_lon'] + 0.2, data['original_lat'], port_name, 
                transform=ccrs.PlateCarree(), fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.3'),
                zorder=6)
        
        ax.text(data['nearest_lon'] + 0.1, data['nearest_lat'] - 0.1, 'SST', 
                transform=ccrs.PlateCarree(), fontsize=7, color='darkred',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1),
                zorder=6)
    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Port location'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=8, label='SST extraction point')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9, framealpha=0.9)
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}
    
    scale_bar(ax, (0.8, 0.05), 100, linewidth=2)
    
    x, y, arrow_length = 0.05, 0.95, 0.07
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='black', width=5, headwidth=15),
                ha='center', va='center', fontsize=12, fontweight='bold',
                xycoords=ax.transAxes, textcoords=ax.transAxes)
    
    plt.title('Port Locations and SST Extraction Points', loc='left', fontsize=12, fontweight='bold')
    
    fig.savefig(f'{output_dir}/plots/port_locations_map.png', dpi=300)
    plt.close()

def scale_bar(ax, location, length_km, linewidth=3):
    """
    Add a scale bar to a map.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to draw the scale bar on
    location : tuple
        The location of the scale bar in axis coordinates (x, y)
    length_km : int
        The length of the scale bar in kilometers
    linewidth : int
        The line width of the scale bar
    """
    llcrnrlon, urcrnrlon = ax.get_xlim()
    llcrnrlat, urcrnrlat = ax.get_ylim()
    
    center_lat = (llcrnrlat + urcrnrlat) / 2

    length_degrees = length_km / (111.32 * np.cos(np.radians(center_lat)))
    
    x, y = location
    x_coord = x * (urcrnrlon - llcrnrlon) + llcrnrlon
    y_coord = y * (urcrnrlat - llcrnrlat) + llcrnrlat
    
    ax.plot([x_coord, x_coord + length_degrees], [y_coord, y_coord], 
            color='black', linewidth=linewidth, transform=ccrs.PlateCarree())
    
    ax.text(x_coord + length_degrees/2, y_coord + 0.1, f'{length_km} km', 
            horizontalalignment='center', verticalalignment='bottom',
            transform=ccrs.PlateCarree(), fontsize=8, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

def get_day_of_year_window(date, window_size=3):
    """
    Get a window of days centered around a specific day of the year.
    
    Parameters:
    -----------
    date : datetime
        Target date
    window_size : int
        Half-width of the window (days on each side)
    
    Returns:
    --------
    list
        List of (month, day) tuples representing the window
    """
    day_window = []
    
    base_date = datetime(2001, date.month, date.day)
    
    for i in range(window_size, 0, -1):
        prev_date = base_date - timedelta(days=i)
        day_window.append((prev_date.month, prev_date.day))
    
    day_window.append((base_date.month, base_date.day))
    
    for i in range(1, window_size + 1):
        next_date = base_date + timedelta(days=i)
        day_window.append((next_date.month, next_date.day))
    
    return day_window

def simulate_daily_sst(ds, port_data_sst, port_name, num_simulations=1):
    """
    Simulate daily SST anomalies for one year and include observed values.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing SST anomaly data
    port_data_sst : dict
        Dictionary containing port data
    port_name : str
        Name of the port to simulate
    num_simulations : int
        Number of simulations to run
    
    Returns:
    --------
    tuple
        (simulations DataFrame, observed_climatology Series, observed_timeseries DataFrame)
    """
    port_info = port_data_sst[port_name]
    lat_idx = port_info['lat_idx']
    lon_idx = port_info['lon_idx']
    
    days = pd.date_range(start='2001-01-01', end='2001-12-31')
    
    all_simulations = pd.DataFrame(index=days)
    
    observed_climatology = pd.Series(index=days, dtype=float)
    
    sst_series = port_info['sst_series']
    
    years = sorted(set(sst_series.index.year))
    observed_timeseries = pd.DataFrame(index=days)
    
    for year in years:
        year_data = sst_series[sst_series.index.year == year]
        year_series = pd.Series(index=days, dtype=float)
        
        for day in days:
            target_date = pd.Timestamp(year=year, month=day.month, day=day.day)
            if target_date in year_data.index:
                year_series[day] = year_data[target_date]
            else:
                year_series[day] = np.nan
        
        observed_timeseries[f'observed_{year}'] = year_series
    
    sst_df = sst_series.reset_index()
    sst_df.columns = ['date', 'sst_anomaly']
    sst_df['dayofyear'] = sst_df['date'].dt.dayofyear
    
    daily_means = sst_df.groupby('dayofyear')['sst_anomaly'].mean()
    
    for day in days:
        day_of_year = day.dayofyear
        if day_of_year in daily_means.index:
            observed_climatology[day] = daily_means[day_of_year]
        else:
            observed_climatology[day] = np.nan
    
    observed_climatology = observed_climatology.interpolate(method='linear')
        
    day_pools = {}
    time_index = pd.to_datetime(ds.time.values)
    
    for day in days:
        day_window = get_day_of_year_window(day, window_size=3) 
        historical_values = []
        
        for month, day_of_month in day_window:
            matching_dates = [date for date in time_index if date.month == month and date.day == day_of_month]
            
            if matching_dates:
                for date in matching_dates:
                    date_idx = np.where(time_index == date)[0]
                    if len(date_idx) > 0:
                        date_idx = date_idx[0]
                        for i in range(max(0, lat_idx-1), min(ds.lat.size, lat_idx+2)):
                            for j in range(max(0, lon_idx-1), min(ds.lon.size, lon_idx+2)):
                                value = ds.sst.values[date_idx, i, j]
                                if not np.isnan(value):
                                    historical_values.append(value)
        
        day_pools[day.dayofyear] = historical_values if historical_values else [np.nan]
    
    for sim in range(num_simulations):
        simulated_values = []
        
        for day in days:
            pool = day_pools[day.dayofyear]
            if pool and not all(np.isnan(pool)):
                simulated_values.append(random.choice(pool))
            else:
                simulated_values.append(np.nan)
        
        sim_series = pd.Series(simulated_values, index=days)        
        sim_series = sim_series.interpolate(method='linear')
        all_simulations[f'sim_{sim}'] = sim_series
    
    return all_simulations, observed_climatology, observed_timeseries

def create_spaghetti_plot(simulations, observed_climatology=None, port_name=None, percentiles=None, output_dir=None):
    """
    Create a spaghetti plot of simulated daily SST anomalies with observed values.
    
    Parameters:
    -----------
    simulations : pandas.DataFrame
        DataFrame containing simulated daily SST anomalies
    observed_climatology : pandas.Series, optional
        Series containing observed climatological SST anomalies. If None, observed climatology won't be plotted.
    port_name : str
        Name of the port
    percentiles : dict
        Dictionary containing percentile values
    output_dir : str
        Directory to save the plot
    """

    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes([0.05, 0.05, 0.9, 0.9])

    ax.set_ylim(-10, 10)
    
    for col in simulations.columns:
        ax.plot(simulations.index.dayofyear, simulations[col], 
                 linewidth=0.5, alpha=0.3, color='gray')
    
    mean_series = simulations.mean(axis=1)
    ax.plot(simulations.index.dayofyear, mean_series, 
             linewidth=2, color='blue', label='Simulated Mean')
    
    # Plot observed climatology only if provided
    if observed_climatology is not None:
        ax.plot(observed_climatology.index.dayofyear, observed_climatology, 
                linewidth=2.5, color='green', label='Observed Climatology')
    
    if percentiles:
        ax.axhline(y=percentiles['80th'], color='orange', linestyle='--', 
                    label=f'80th percentile: {percentiles["80th"]:.2f}°C')
        ax.axhline(y=percentiles['90th'], color='red', linestyle='--', 
                    label=f'90th percentile: {percentiles["90th"]:.2f}°C')
        ax.axhline(y=percentiles['95th'], color='darkred', linestyle='--', 
                    label=f'95th percentile: {percentiles["95th"]:.2f}°C')
    
    if port_name:
        ax.set_title(f'Simulated daily SSTa for {port_name}', loc='left', fontsize=10)
    else:
        ax.set_title('Simulated daily SSTa', loc='left', fontsize=10)
        
    ax.set_xlabel('Day of year', fontsize=8)
    ax.set_ylabel('SSTa (°C)', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if output_dir and port_name:
        fig.savefig(f'{output_dir}/plots/{port_name}_spaghetti_plot.png', dpi=300)
        plt.close()
    else:
        return fig, ax

def create_observed_closure_matrix(ds, port_data_sst, port_percentiles, percentile_key='95th'):
    """
    Create a matrix of days x ports with 1s and 0s based on whether the observed
    SST anomaly exceeds the specified percentile.
    
    Parameters:
    -----------
    ds : xarray.Dataset
        Dataset containing SST anomaly data
    port_data_sst : dict
        Dictionary with port data including coordinates and SST series
    port_percentiles : dict
        Dictionary with percentile values for each port
    percentile_key : str
        Key for the percentile to use (default: '95th')
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with days as rows and ports as columns, containing 1s and 0s
    """
    dates = pd.to_datetime(ds.time.values)
    closure_matrix = pd.DataFrame(index=dates)
    
    for port_name, port_info in port_data_sst.items():
        sst_series = port_info['sst_series']
        
        percentile_value = port_percentiles[port_name][percentile_key]
        
        closure_status = (sst_series > percentile_value).astype(int)
        
        closure_matrix[port_name] = closure_status
    
    return closure_matrix


def create_anomaly_matrix(port_data_sst, port_percentiles, percentile_key='95th'):
    """
    Create a matrix of days x ports with 1s and 0s based on whether the observed
    SST anomaly exceeds the specified percentile.
    
    Parameters
    ----------
    port_data_sst : dict or pandas.DataFrame
        Dictionary containing port data with SST time series or DataFrame with SST data
    port_percentiles : dict
        Dictionary containing percentile values for each port.
    percentile_key : str, optional
        Key for the percentile to use (default is '95th').
        
    Returns
    -------
    tuple
        Tuple containing two pandas.DataFrame objects:
        - anomaly_matrix_binary: Matrix with days as rows and ports as columns, containing 1s and 0s.
        - anomaly_matrix_ssta: Matrix with days as rows and ports as columns, containing SST anomalies.
    """
    if isinstance(port_data_sst, pd.DataFrame):
        # Si es DataFrame, usar directamente
        anomaly_matrix_binary = pd.DataFrame(index=port_data_sst.index)
        anomaly_matrix_ssta = port_data_sst.copy()
        
        for port_name in port_data_sst.columns:
            sst_series = port_data_sst[port_name]
            percentile_value = port_percentiles[port_name][percentile_key]
            anomaly_matrix_binary[port_name] = (sst_series > percentile_value).astype(int)
    else:
        # Si es diccionario, mantener el comportamiento original
        dates = port_data_sst[list(port_data_sst.keys())[0]]['sst_series'].index
        anomaly_matrix_binary = pd.DataFrame(index=dates)
        anomaly_matrix_ssta = pd.DataFrame(index=dates)
        
        for port_name, port_info in port_data_sst.items():
            sst_series = port_info['sst_series']
            percentile_value = port_percentiles[port_name][percentile_key]
            
            anomaly_matrix_binary[port_name] = (sst_series > percentile_value).astype(int)
            anomaly_matrix_ssta[port_name] = sst_series

    return anomaly_matrix_binary, anomaly_matrix_ssta


def sample_with_replacement(anomaly_matrix_binary, anomaly_matrix_ssta, num_samples=1000, window_size=3, random_seed=42):
    """
    Create 1000 annual samples by sampling from a window around each calendar day.
    
    Parameters
    ----------
    anomaly_matrix_binary : pandas.DataFrame
        Matrix with days as rows and ports as columns, with datetime index
    anomaly_matrix_ssta : pandas.DataFrame
        Matrix with days as rows and ports as columns, with datetime index
    num_samples : int, optional
        Number of annual samples to create (default is 1000)
    window_size : int, optional
        Number of days before and after each date to include in sampling window (default is 3)
    random_seed : int, optional
        Random seed for reproducibility (default is 42)
        
    Returns
    -------
    pandas.DataFrame
        Matrix containing num_samples annual samples, with datetime index
    """
    np.random.seed(random_seed)
    
    reference_year = 2023
    dates_in_year = pd.date_range(f'{reference_year}-01-01', f'{reference_year}-12-31')
    
    sampled_data_binary = []
    sampled_data_ssta = []
    
    for sample_num in range(num_samples):
        annual_sample_binary = []
        annual_sample_ssta = []
        
        for target_date in dates_in_year:
            month, day = target_date.month, target_date.day
            
            window_dates = []
            for year in anomaly_matrix_binary.index.year.unique():
                try:
                    base_date = pd.Timestamp(year=year, month=month, day=day)
                    date_window = pd.date_range(base_date - pd.Timedelta(days=window_size),
                                              base_date + pd.Timedelta(days=window_size))
                    window_dates.extend(date_window)
                except ValueError:
                    continue  
            
            valid_dates = [d for d in window_dates if d in anomaly_matrix_binary.index]
            
            if valid_dates:
                sampled_date = np.random.choice(valid_dates)
                sample_binary = anomaly_matrix_binary.loc[sampled_date].copy()
                sample_ssta = anomaly_matrix_ssta.loc[sampled_date].copy()
                annual_sample_binary.append(sample_binary)
                annual_sample_ssta.append(sample_ssta)
        
        annual_df_binary = pd.DataFrame(annual_sample_binary, index=dates_in_year, columns=anomaly_matrix_binary.columns)
        annual_df_ssta = pd.DataFrame(annual_sample_ssta, index=dates_in_year, columns=anomaly_matrix_ssta.columns)
        sampled_data_binary.append(annual_df_binary)
        sampled_data_ssta.append(annual_df_ssta)
    
    return pd.concat(sampled_data_binary), pd.concat(sampled_data_ssta)


# --> un anio en las filas y en las columnas los puertos. 


# tengo mil bases de un anio, 

# daily loss por puerto 

# mil anios 

# tengo mil anios de loses, 

# ordeno todos los loses todos de mayor a menor ploteos, percentiles y alv
def calculate_losses_matrix_from_daily_wages(anomaly_matrix_samples, port_losses):
    """
    Multiplica la matriz de anomalías muestreada por los valores de pérdidas correspondientes a cada puerto.
    
    Parameters
    ----------
    anomaly_matrix_samples : pandas.DataFrame
        DataFrame con muestras de la matriz de anomalías (1=cierre, 0=abierto)
        Las columnas tienen el formato "puerto_sample_N" donde N es el número de muestra
    port_losses : dict
        Diccionario con valores de pérdida diaria para cada puerto cuando está cerrado
        
    Returns
    -------
    pandas.DataFrame
        DataFrame con pérdidas diarias por puerto para cada muestra
    """
    losses_matrix = pd.DataFrame(index=anomaly_matrix_samples.index)
    
    for col in anomaly_matrix_samples.columns:
        port_name = col.split('_sample_')[0]
        
        loss_value = port_losses.get(port_name, 0)
        
        losses_matrix[col] = anomaly_matrix_samples[col] * loss_value
    
    return losses_matrix


def calculate_losses_matrix_from_regression(anomaly_matrix_samples_ssta, results_regression, daily_price_per_ton):
    """
    Multiplica la matriz de anomalías muestreada por los valores de pérdidas correspondientes a cada puerto.
    
    Parameters
    ----------
    anomaly_matrix_samples_ssta : pandas.DataFrame
        DataFrame con muestras de la matriz de anomalías SST
    results_regression : dict
        Diccionario con resultados de la regresión para cada puerto
    daily_price_per_ton : float
        Precio por tonelada al día
    Returns
    -------
    pandas.DataFrame
        DataFrame con pérdidas diarias por puerto para cada muestra
    """
    losses_matrix = pd.DataFrame(index=anomaly_matrix_samples_ssta.index)
    
    for port_name, regression_results in results_regression.items():
        sst_series = anomaly_matrix_samples_ssta[port_name]
        delta_production = sst_series.where(sst_series >= 0, 0) * regression_results['slope']*(-1)
        losses_matrix[port_name] = delta_production * daily_price_per_ton

    return losses_matrix


def calculate_annual_total_losses(losses_matrix):
    """
    Calcula las pérdidas totales anuales sumando a través de los puertos,
    para cada una de las simulaciones.
    
    Parameters
    ----------
    losses_matrix : pandas.DataFrame
        DataFrame con pérdidas diarias por puerto para cada muestra
        
    Returns
    -------
    pandas.Series
        Serie con las pérdidas totales anuales para cada simulación
    """

    n_days = 365
    n_chunks = len(losses_matrix) // n_days
    
    annual_losses = []
    
    for i in range(n_chunks):
        start_idx = i * n_days
        end_idx = (i + 1) * n_days
        chunk = losses_matrix.iloc[start_idx:end_idx]
        
        annual_loss = chunk.sum().sum()
        annual_losses.append(annual_loss)
    
    return pd.Series(annual_losses, name='annual_losses')

def sort_annual_losses(annual_losses):
    """
    Ordena el vector de pérdidas anuales de menor a mayor.
    
    Parameters
    ----------
    annual_losses : pandas.Series
        Serie con las pérdidas totales anuales para cada simulación
        
    Returns
    -------
    pandas.Series
        Serie ordenada con las pérdidas totales anuales
    """
    return annual_losses.sort_values(ascending=False)

def plot_annual_losses(annual_losses):
    """
    Crea un gráfico de excedencia (curva de frecuencia acumulada) para las pérdidas anuales.
    
    Parameters
    ----------
    annual_losses : pandas.Series
        Serie con las pérdidas totales anuales para cada simulación
        
    Returns
    -------
    matplotlib.figure.Figure
        Figura con la curva de excedencia de pérdidas anuales
    """
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes([0.1, 0.1, 0.8, 0.8])
    
    sorted_losses = sort_annual_losses(annual_losses)
    
    n = len(sorted_losses)
    exceedance_prob = np.arange(1, n + 1) / n
    
    ax.plot(sorted_losses.values, exceedance_prob, linewidth=2, color='blue')
    
    ax.set_title('Curva de excedencia de pérdidas anuales', fontsize=14)
    ax.set_xlabel('Pérdidas Anuales ($)', fontsize=12)
    ax.set_ylabel('Probabilidad de Excedencia', fontsize=12)
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    ax.tick_params(axis='x', rotation=0)
    
    ax.grid(True, alpha=0.3)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
        
    return fig


def load_port_landing_data(port_file):

    df_prod = pd.read_csv(port_file)
    df_prod['date'] = pd.to_datetime(df_prod['date'])
    df_prod = df_prod.set_index('date')

    return df_prod

def calculate_landing_sst_anomaly_regression(df_prod, port_data_sst):
    """
    Calculate regression between landings and SST anomalies for each port.
    
    Parameters
    ----------
    df_prod : pandas.DataFrame
        DataFrame with landing data for each port
    port_data_sst : dict or pandas.DataFrame
        Dictionary with SST data for each port or DataFrame with SST data
        
    Returns
    -------
    dict
        Dictionary with regression results for each port
    """
    results = {}

    if isinstance(port_data_sst, pd.DataFrame):
        # Si es DataFrame, usar directamente las columnas
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

            slope, intercept, r_value, p_value, std_err = stats.linregress(
                data['sst_anomaly'], data['production'])

            results[port_name] = {
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'p_value': p_value,
                'std_err': std_err
            }
    else:
        # Si es diccionario, mantener el comportamiento original
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

            slope, intercept, r_value, p_value, std_err = stats.linregress(
                data['sst_anomaly'], data['production'])

            results[port_name] = {
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'p_value': p_value,
                'std_err': std_err
            }

    return results
