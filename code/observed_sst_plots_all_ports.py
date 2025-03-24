import os
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from port_sst_analysis_functions import extract_port_sst_series

puertos_coords = {
    "Paita": (-5.0892, -81.1144),
    "Parachique": (-5.5745, -80.9006),
    "Chicama": (-7.8432, -79.4000),
    "Chimbote": (-9.0746, -78.5936),
    "Samanco": (-9.1845, -78.4942),
    "Casma": (-9.4549, -78.3854),
    "Huarmey": (-10.0965, -78.1716),
    "Supe": (-10.7979, -77.7094),
    "Vegueta": (-11.0281, -77.6489),
    "Huacho": (-11.1067, -77.6056),
    "Chancay": (-11.5624, -77.2705),
    "Callao": (-12.0432, -77.1469),
    "Tambo de Mora": (-13.4712, -76.1932),
    "Atico": (-16.2101, -73.6111),
    "Planchada": (-16.4061, -73.2186),
    "Mollendo": (-17.0231, -72.0145),
    "Ilo": (-17.6394, -71.3374)
}

output_dir = 'results/modis_observed_sst_plots'
os.makedirs(output_dir, exist_ok=True)

print(f"Loading MODIS SST anomaly dataset...")
sst_file = 'data/MODIS/processed/sst_anomaly_daily_2002_2025.nc'
ds = xr.open_dataset(sst_file)

print(f"Extracting SST anomaly time series for all ports...")
port_data = extract_port_sst_series(ds, puertos_coords)

for port_name, data in port_data.items():
    print(f"Processing {port_name}...")
    
    sst_series = data['sst_series']
    
    percentiles = {
        '80th': np.nanpercentile(sst_series.values, 80),
        '90th': np.nanpercentile(sst_series.values, 90),
        '95th': np.nanpercentile(sst_series.values, 95)
    }
    
    years = sorted(set(sst_series.index.year))
    days = pd.date_range(start='2001-01-01', end='2001-12-31')
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
    
    print(f"Creating observed time series plot for {port_name}...")


    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes([0.05, 0.05, 0.85, 0.85])

    for col in observed_timeseries.columns:
        year = int(col.split('_')[1])
        if year % 10 == 0:
            lw = 0.5
            color = 'black'
        else:
            lw = 0.2
            color = 'grey'
        ax.plot(observed_timeseries.index.dayofyear, observed_timeseries[col], 
                 linewidth=lw, alpha=1, label=f'Observed {year}', color=color)
        
    ax.set_ylim([-5, 5])

    ax.axhline(y=percentiles['80th'], color='orange', linestyle='--', lw=1, 
                label=f'80th percentile: {percentiles["80th"]:.2f}°C')
    ax.axhline(y=percentiles['90th'], color='red', linestyle='--', lw=1, 
                label=f'90th percentile: {percentiles["90th"]:.2f}°C')
    ax.axhline(y=percentiles['95th'], color='darkred', linestyle='--', lw=1, 
                label=f'95th percentile: {percentiles["95th"]:.2f}°C')
    
    ax.set_title(f'MODIS SSTa for {port_name}', loc='left', fontsize=10)
    ax.set_xlabel('Day of year', fontsize=8)
    ax.set_ylabel('SST anomaly (°C)', fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=8)

    handles, labels = ax.get_legend_handles_labels()
    year_indices = [i for i, label in enumerate(labels) if 'Observed' in label and int(label.split(' ')[1]) % 10 == 0]
    percentile_indices = [i for i, label in enumerate(labels) if 'percentile' in label]
    selected_indices = sorted(year_indices + percentile_indices)
    ax.legend([handles[i] for i in selected_indices], [labels[i] for i in selected_indices], 
               fontsize=8, loc='lower right', frameon=False)
    ax.axhline(y=0, color='black', linestyle='-', lw=0.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.savefig(f'{output_dir}/{port_name}_modis_observed_timeseries.png', dpi=300)
    plt.close()

print(f"All MODIS plots saved to {output_dir}/") 