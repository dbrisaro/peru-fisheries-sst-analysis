import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import numpy as np
import calendar
import matplotlib.lines as mlines

def plot_may_sst(lon, lat, sst, vmin, vmax):
    """
    Creates a single plot of SST data for May with port locations.
    
    Parameters:
    -----------
    lon, lat : array-like
        Longitude and latitude coordinates
    sst : xarray.DataArray
        Sea Surface Temperature data
    vmin, vmax : float
        Minimum and maximum values for the colorbar
    """
    # Define port coordinates and groups
    puertos_coords = {
        "Paita": (-5.0892, -81.1144),
        "Parachique": (-5.5745, -80.9006),
        "Chicama": (-7.8432, -79.4000),
        "Chimbote": (-9.0746, -78.5936),
        "Samanco": (-9.1845, -78.4942),
        "Casma": (-9.4745, -78.2914),
        "Huarmey": (-10.0681, -78.1508),
        "Supe": (-10.7979, -77.7094),
        "Vegueta": (-11.0281, -77.6489),
        "Huacho": (-11.1067, -77.6056),
        "Chancay": (-11.5624, -77.2705),
        "Callao": (-12.0432, -77.0283),
        "Tambo de Mora": (-13.4350, -76.1356),
        "Atico": (-16.2101, -73.6111),
        "Planchada": (-16.9833, -72.0833),
        "Mollendo": (-17.0231, -72.0145),
        "Ilo": (-17.6394, -71.3374)
    }
    
    port_groups = [
        ['Paita', 'Parachique'],
        ['Chicama', 'Chimbote', 'Samanco', 'Tambo de Mora'],
        ['Huarmey', 'Casma'],
        ['Supe', 'Vegueta', 'Huacho', 'Chancay', 'Callao'],
        ['Atico', 'Planchada', 'Mollendo', 'Ilo']
    ]
    colors = ['#FF1E1E', '#4A90E2', '#50E3C2', '#F5A623', '#8B572A']
    
    # Set up the figure
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Plot SST data
    cmap = cmocean.cm.thermal
    cmap = plt.cm.get_cmap('RdBu_r')
    im = ax.pcolormesh(lon, lat, sst.isel(time=4),  # May is index 4 (0-based)
                      cmap=cmap, vmin=vmin, vmax=vmax, 
                      transform=ccrs.PlateCarree(), alpha=1)
    
    # Add contours
    levels = np.arange(np.floor(vmin), np.ceil(vmax) + 0.5, 0.5)
    contours = ax.contour(lon, lat, sst.isel(time=4), 
                         levels=levels, colors='k', 
                         linewidths=0.25, 
                         transform=ccrs.PlateCarree())
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.1f")
    
    # Add coastline
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    
    # Set map extent
    ax.set_extent([-85, -70, -20, -4], crs=ccrs.PlateCarree())
    
    # Add gridlines
    lon_ticks = np.arange(-85, -69, 2)
    lat_ticks = np.arange(-20, -3, 2)
    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    
    ax.coastlines(zorder=4)
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray', linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.LAKES, facecolor='white', zorder=3)
    ax.add_feature(cfeature.RIVERS, edgecolor='white', zorder=3)
    # Create legend elements

    legend_elements = []
    for i, (group, color) in enumerate(zip(port_groups, colors)):
        legend_elements.append(mlines.Line2D([], [], color=color, marker='o',
                                           linestyle='None', markersize=5,
                                           label=f"Group {i+1}"))
    
    # Add ports by group
    for group, color in zip(port_groups, colors):
        # Plot ports in this group
        for port in group:
            if port in puertos_coords:
                lat, lon = puertos_coords[port]
                ax.scatter(lon, lat,
                          color=color, edgecolor='black', s=60, alpha=0.9,
                          linewidths=0.5, transform=ccrs.PlateCarree(), zorder=5)
                ax.text(lon + 0.2, lat, port,
                       fontsize=8, color='black',
                       transform=ccrs.PlateCarree(),
                       ha='left', va='center',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                
    
    # Add colorbar
    cbar = plt.colorbar(im, orientation="vertical", extend='both', pad=0.02)
    cbar.set_label("Sea Surface Temperature (°C)", fontsize=10)
    
    plt.title(f"Sea Surface Temperature - {calendar.month_name[5]}", fontsize=8, loc='left')
    
    ax.legend(handles=legend_elements, frameon=False, loc='upper right', 
             fontsize=8, title='Port Groups', title_fontsize=8)
    
    return fig

# Example usage:
# fig = plot_may_sst(lon, lat, sst, vmin, vmax)
# plt.show()
# fig.savefig('../docs/figs/oisstv2/sst_may.png', dpi=300, bbox_inches='tight')
# fig.savefig('../docs/figs/oisstv2/sst_may.pdf', bbox_inches='tight')