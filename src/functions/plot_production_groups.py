import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import matplotlib.patches as patches

def create_port_location_map(output_dir='../docs/figs'):
    """
    Create a map showing the location of all ports.
    
    Parameters:
    -----------
    output_dir : str, optional
        Directory where the map will be saved. Defaults to '../docs/figs'
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define port coordinates
    ports = {
        'Paita': (-81.1, -5.1),
        'Parachique': (-81.1, -5.1),
        'Chicama': (-79.0, -7.7),
        'Chimbote': (-78.6, -9.1),
        'Samanco': (-78.4, -9.3),
        'Tambo de Mora': (-75.7, -13.5),
        'Huarmey': (-78.2, -10.1),
        'Casma': (-78.3, -9.5),
        'Supe': (-77.7, -10.8),
        'Vegueta': (-77.6, -11.0),
        'Huacho': (-77.6, -11.1),
        'Chancay': (-77.3, -11.6),
        'Callao': (-77.2, -12.1),
        'Atico': (-73.6, -16.2),
        'Planchada': (-73.5, -16.3),
        'Mollendo': (-72.0, -17.0),
        'Ilo': (-71.3, -17.6)
    }
    
    # Define port groups
    port_groups = {
        'Group_1': ['Paita', 'Parachique'],
        'Group_2': ['Chicama', 'Chimbote', 'Samanco', 'Tambo de Mora'],
        'Group_3': ['Huarmey', 'Casma'],
        'Group_4': ['Supe', 'Vegueta', 'Huacho', 'Chancay', 'Callao'],
        'Group_5': ['Atico', 'Planchada', 'Mollendo', 'Ilo']
    }
    
    # Create figure and axis with projection
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add coastlines and country borders
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    
    # Set map extent (Peru)
    ax.set_extent([-82, -70, -18, 0], crs=ccrs.PlateCarree())
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    
    # Plot ports with different colors for each group
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    markers = ['o', 's', '^', 'D', 'v']
    
    for (group, group_ports), color, marker in zip(port_groups.items(), colors, markers):
        for port in group_ports:
            lon, lat = ports[port]
            ax.plot(lon, lat, color=color, marker=marker, markersize=8, 
                   transform=ccrs.PlateCarree(), label=f'{group} - {port}')
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Add title
    plt.title('Port Locations by Group', pad=20)
    
    # Save the map
    plt.savefig(f'{output_dir}/port_locations.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_production_by_group(data, start_date, end_date, output_dir='../docs/figs'):
    """
    Create production plots for each group of ports.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with production data indexed by date
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    output_dir : str, optional
        Directory where figures will be saved. Defaults to '../docs/figs'
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    years = pd.date_range(start=start_date, end=end_date, freq='YE')
    data = data.resample('ME').sum()
    
    # Get the maximum y value across all groups for consistent y-axis
    y_max = data.max().max()
    
    # Create a plot for each group
    for group in data.columns:
        fig = plt.figure(figsize=(8, 4))
        ax = plt.axes([0.1, 0.1, 0.8, 0.8])
        
        x_data = data.index
        y_data = data[group]
        
        ax.plot(x_data, y_data, color='red', linewidth=1)
        ax.set_ylabel('Production (tons)', fontsize=8)
        ax.set_xlabel('Date', fontsize=8)
        ax.set_title(f'Production by {group}', fontsize=8, loc='left', pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=8)
        ax.set_xlim([pd.to_datetime(start_date), pd.to_datetime(end_date)])
        ax.set_xticks(pd.date_range(start=start_date, end=end_date, freq='YE'))
        ax.set_ylim([0, y_max])
        ax.set_xticklabels([d.strftime('%Y') for d in years], rotation=45)
        
        plt.savefig(f'{output_dir}/{group}_production.png', dpi=300, bbox_inches='tight')
        plt.close()
