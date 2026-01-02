import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
import cartopy.feature as cfeature
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
from scipy import stats
from shapely.geometry import Point, Polygon, MultiPolygon
import geopandas as gpd
import requests
import matplotlib.lines as mlines

def fetch_boundaries(continent="Americas", region="South America", limit=100):
    """
    Fetches administrative boundaries from the public OpenDataSoft API for a specified continent and region.
    
    Parameters:
        continent (str): The continent to filter the boundaries (by default: "Americas").
        region (str): The region within the continent (by default: "South America").
        limit (int): The number of records to fetch (default: 100). 
        
    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing the boundaries as geometries.
    """
    url = f"https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/world-administrative-boundaries/records?limit={limit}&refine=continent%3A%22{continent}%22&refine=region%3A%22{region}%22"

    try:
        response = requests.get(url)
        response.raise_for_status()

        data = response.json()
        geometries = []

        for record in data.get("results", []):
            geo_shape = record.get("geo_shape", {})
            geometry = geo_shape.get("geometry", {})

            if geometry and geometry["type"] in ["Polygon", "MultiPolygon"]:
                coords = geometry["coordinates"]

                if geometry["type"] == "Polygon":
                    geometries.append(Polygon(coords[0]))
                elif geometry["type"] == "MultiPolygon":
                    multi_poly = MultiPolygon([Polygon(p[0]) for p in coords])
                    geometries.append(multi_poly)

        return gpd.GeoDataFrame(geometry=geometries)

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return gpd.GeoDataFrame()

def create_land_mask(lon, lat, boundaries):
    """
    Create a mask for land areas using administrative boundaries.
    
    Parameters:
        lon (numpy.ndarray): Longitude coordinates
        lat (numpy.ndarray): Latitude coordinates
        boundaries (geopandas.GeoDataFrame): Administrative boundaries
        
    Returns:
        numpy.ndarray: Boolean mask where True indicates land
    """
    lon2d, lat2d = np.meshgrid(lon, lat)
    points = [Point(x, y) for x, y in zip(lon2d.flatten(), lat2d.flatten())]
    mask = np.zeros(len(points), dtype=bool)
    for geom in boundaries.geometry:
        mask |= np.array([geom.contains(point) for point in points])
    return mask.reshape(lon2d.shape)

def perform_nested_clustering(data_array, lon, lat, n_clusters_outer=4, n_clusters_inner=2):
    """
    Perform nested clustering on ocean data.
    
    Parameters:
    -----------
    data_array : numpy.ndarray
        Input data array with shape (time, lat, lon)
    lon : numpy.ndarray
        Longitude coordinates
    lat : numpy.ndarray
        Latitude coordinates
    n_clusters_outer : int
        Number of clusters for the outer clustering
    n_clusters_inner : int
        Number of clusters for the inner clustering within each outer cluster
        
    Returns:
    --------
    dict containing:
        'outer_clusters': Outer cluster labels
        'inner_clusters': Dictionary of inner cluster labels for each outer cluster
        'spatial_clusters': Full spatial cluster map with nested structure
    """
    print("\n=== Starting Nested Clustering Process ===")
    print(f"Input data shape: {data_array.shape}")
    print(f"Number of outer clusters: {n_clusters_outer}")
    print(f"Number of inner clusters: {n_clusters_inner}")
    
    original_shape = data_array.shape
    reshaped_data = data_array.reshape(original_shape[0], -1)
    print(f"\nReshaped data shape: {reshaped_data.shape}")
    
    valid_points = ~np.isnan(reshaped_data).all(axis=0)
    valid_points = valid_points.reshape(original_shape[1:])
    print(f"\nValid points shape: {valid_points.shape}")
    print(f"Number of valid points: {np.sum(valid_points)}")
    
    print("\nFetching administrative boundaries...")
    boundaries = fetch_boundaries()
    land_mask = create_land_mask(lon, lat, boundaries)
    ocean_mask = ~land_mask
    print(f"Ocean points: {np.sum(ocean_mask)}")
    
    valid_points = valid_points & ocean_mask
    print(f"\nFinal valid points: {np.sum(valid_points)}")
    
    valid_points_flat = valid_points.reshape(-1)
    clean_data = reshaped_data[:, valid_points_flat]
    print(f"\nClean data shape: {clean_data.shape}")
    
    print("\nHandling NaN values...")
    for i in range(clean_data.shape[1]):
        column = clean_data[:, i]
        if np.any(np.isnan(column)):
            mean_val = np.nanmean(column)
            clean_data[:, i] = np.where(np.isnan(column), mean_val, column)
    
    print("\nStandardizing data...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clean_data)
    print(f"Scaled data shape: {scaled_data.shape}")
    
    print("\nPerforming outer clustering...")
    kmeans_outer = KMeans(n_clusters=n_clusters_outer, random_state=42)
    outer_clusters = kmeans_outer.fit_predict(scaled_data.T)
    print(f"Outer clusters shape: {outer_clusters.shape}")
    print("Outer cluster distribution:")
    for i in range(n_clusters_outer):
        print(f"  Cluster {i}: {np.sum(outer_clusters == i)} points")
    
    print("\nPerforming inner clustering...")
    inner_clusters = {}
    for i in range(n_clusters_outer):
        mask = outer_clusters == i
        if np.sum(mask) > 1:
            cluster_data = scaled_data[:, mask]
            kmeans_inner = KMeans(n_clusters=n_clusters_inner, random_state=42)
            inner_clusters[i] = kmeans_inner.fit_predict(cluster_data.T)
            print(f"\nInner clusters for outer cluster {i}:")
            for j in range(n_clusters_inner):
                print(f"  Subcluster {j}: {np.sum(inner_clusters[i] == j)} points")
        else:
            inner_clusters[i] = np.zeros(np.sum(mask))
            print(f"\nOuter cluster {i} has only one point")
    
    spatial_clusters = np.full(original_shape[1:], np.nan)
    
    combined_clusters = np.zeros_like(outer_clusters)
    for i in range(n_clusters_outer):
        mask = outer_clusters == i
        if i in inner_clusters:
            combined_clusters[mask] = i * n_clusters_inner + inner_clusters[i]
    
    spatial_clusters[valid_points] = combined_clusters
    
    print("\nFinal cluster distribution:")
    for i in range(n_clusters_outer * n_clusters_inner):
        print(f"Cluster {i}: {np.sum(spatial_clusters == i)} points")
    
    print("\n=== Nested Clustering Process Complete ===")
    
    return {
        'outer_clusters': outer_clusters,
        'inner_clusters': inner_clusters,
        'spatial_clusters': spatial_clusters,
        'valid_mask': valid_points,
        'n_clusters_total': n_clusters_outer * n_clusters_inner
    }

def plot_nested_clusters(cluster_results, data, n_clusters_outer=4, n_clusters_inner=2):
    """
    Plot nested clustering results.
    """
    print("\n=== Starting Cluster Visualization ===")
    print(f"Input data shape: {data['sst'].shape}")
    print(f"Number of outer clusters: {n_clusters_outer}")
    print(f"Number of inner clusters: {n_clusters_inner}")
    
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    valid_mask = cluster_results['valid_mask']
    spatial_clusters = cluster_results['spatial_clusters'].copy()
    
    print(f"\nValid mask shape: {valid_mask.shape}")
    print(f"Number of valid points: {np.sum(valid_mask)}")
    print(f"Spatial clusters shape: {spatial_clusters.shape}")
    
    cluster_visualization = np.full_like(spatial_clusters, np.nan)
    cluster_visualization[valid_mask] = spatial_clusters[valid_mask]
    
    print("\nCluster visualization statistics:")
    print(f"Number of non-NaN values: {np.sum(~np.isnan(cluster_visualization))}")
    print(f"Unique cluster values: {np.unique(cluster_visualization[~np.isnan(cluster_visualization)])}")
    
    n_total_clusters = n_clusters_outer * n_clusters_inner
    
    outer_colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#9B59B6',
        '#E67E22', '#2ECC71', '#34495E', '#16A085', '#27AE60'
    ]
    
    colors = []
    for outer_idx in range(n_clusters_outer):
        base_color = outer_colors[outer_idx % len(outer_colors)]
        r = int(base_color[1:3], 16)
        g = int(base_color[3:5], 16)
        b = int(base_color[5:7], 16)
        
        for inner_idx in range(n_clusters_inner):
            brightness_factor = 0.7 + (0.3 * inner_idx / (n_clusters_inner - 1))
            new_r = int(r * brightness_factor)
            new_g = int(g * brightness_factor)
            new_b = int(b * brightness_factor)
            new_color = f'#{new_r:02x}{new_g:02x}{new_b:02x}'
            colors.append(new_color)
    
    custom_cmap = ListedColormap(colors[:n_total_clusters])
    
    print(f"\nNumber of colors in colormap: {len(colors[:n_total_clusters])}")
    
    print("\nPlotting ocean background...")
    ax.add_feature(cfeature.OCEAN, facecolor='white', zorder=1)
    
    print("\nPlotting clusters...")
    im = ax.pcolormesh(data.lon, data.lat, cluster_visualization,
                      transform=ccrs.PlateCarree(),
                      cmap=custom_cmap,
                      vmin=0,
                      vmax=n_total_clusters-1,
                      zorder=2)
    
    print("\nAdding map features...")
    ax.coastlines(zorder=4)
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray', linewidth=0.5, zorder=3)
    ax.add_feature(cfeature.LAKES, facecolor='white', zorder=3)
    ax.add_feature(cfeature.RIVERS, edgecolor='white', zorder=3)
    
    # Add port locations and groups
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
    
    port_colors = ['#FF1E1E', '#4A90E2', '#50E3C2', '#F5A623', '#8B572A']
    
    legend_elements = []
    for i, (group, color) in enumerate(zip(port_groups, port_colors)):
        for port in group:
            if port in puertos_coords:
                lat, lon = puertos_coords[port]
                ax.plot(lon, lat, 'o', color=color, markersize=5, 
                       transform=ccrs.PlateCarree(), zorder=5)
                ax.text(lon+0.2, lat, port, fontsize=8, color='black',
                       transform=ccrs.PlateCarree(), zorder=5)
        legend_elements.append(mlines.Line2D([], [], color=color, marker='o',
                             linestyle='None', markersize=5, 
                             label=f"Group {i+1}"))
    
    lon = data['lon']
    lat = data['lat']
    lon_padding = 0
    lat_padding = 0
    lon_min, lon_max = np.round((lon.min().values - lon_padding) * 2) / 2, np.round((lon.max().values + lon_padding) * 2) / 2
    lat_min, lat_max = np.round((lat.min().values - lat_padding) * 2) / 2, np.round((lat.max().values + lat_padding) * 2) / 2
    lon_ticks = np.linspace(lon_min, lon_max, num=min(5, int((lon_max - lon_min) / 0.5) + 1))
    lat_ticks = np.linspace(lat_min, lat_max, num=min(5, int((lat_max - lat_min) / 0.5) + 1))
    
    print(f"\nMap extent:")
    print(f"Longitude: {lon_min} to {lon_max}")
    print(f"Latitude: {lat_min} to {lat_max}")
    
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax.set_xlabel('Longitude', fontsize=8)
    ax.set_ylabel('Latitude', fontsize=8)
    
    ax.set_title('Nested Clusters', fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    
    print("\nAdding colorbar...")
    cbar = plt.colorbar(im, label='Cluster', orientation='vertical', pad=0.05, shrink=0.5)
    cbar.set_ticks(range(n_total_clusters))
    cbar.set_ticklabels([f'Outer {i//n_clusters_inner} Inner {i%n_clusters_inner}' 
                        for i in range(n_total_clusters)], fontsize=8)
    
    # Add port groups legend
    ax.legend(handles=legend_elements, frameon=False, loc='lower left', 
             fontsize=8, title='Port Groups', title_fontsize=8)
    
    plt.tight_layout()
    print("\n=== Cluster Visualization Complete ===")
    
    return fig

def analyze_cluster_patterns(cluster_results, data, time_index, n_clusters_outer=4, n_clusters_inner=2):
    """
    Analyze temporal patterns for each nested cluster.
    Creates one panel per outer cluster, showing all its inner clusters' time series.
    
    Parameters:
    -----------
    cluster_results : dict
        Results from perform_nested_clustering
    data : xarray.Dataset
        Original data with coordinates
    time_index : array-like
        Time index for x-axis
    n_clusters_outer : int
        Number of outer clusters
    n_clusters_inner : int
        Number of inner clusters
    """
    print("\n=== Starting Cluster Pattern Analysis ===")
    print(f"Number of outer clusters: {n_clusters_outer}")
    print(f"Number of inner clusters: {n_clusters_inner}")
    print(f"Total number of clusters: {n_clusters_outer * n_clusters_inner}")
    
    fig = plt.figure(figsize=(10, 3*n_clusters_outer))
    
    outer_colors = [
        '#FF6B6B',
        '#4ECDC4',
        '#45B7D1',
        '#96CEB4',
        '#9B59B6',
        '#E67E22',
        '#2ECC71',
        '#34495E',
        '#16A085',
        '#27AE60'
    ]
    
    for outer_idx in range(n_clusters_outer):
        print(f"\nProcessing outer cluster {outer_idx}")
        ax = plt.subplot(n_clusters_outer, 1, outer_idx+1)
        
        base_color = outer_colors[outer_idx % len(outer_colors)]
        r = int(base_color[1:3], 16)
        g = int(base_color[3:5], 16)
        b = int(base_color[5:7], 16)
        
        for inner_idx in range(n_clusters_inner):
            cluster_idx = outer_idx * n_clusters_inner + inner_idx
            print(f"  Processing inner cluster {inner_idx} (total index: {cluster_idx})")
            
            mask = cluster_results['spatial_clusters'] == cluster_idx
            cluster_data = data['sst'].values[:, mask]
            cluster_mean = np.nanmean(cluster_data, axis=1)
            
            print(f"    Number of points in cluster: {np.sum(mask)}")
            print(f"    Mean time series shape: {cluster_mean.shape}")
            
            brightness_factor = 0.7 + (0.3 * inner_idx / (n_clusters_inner - 1))
            new_r = int(r * brightness_factor)
            new_g = int(g * brightness_factor)
            new_b = int(b * brightness_factor)
            color = f'#{new_r:02x}{new_g:02x}{new_b:02x}'
            
            ax.plot(time_index, cluster_mean, color=color, linewidth=0.5, 
                   label=f'Inner {inner_idx}')
        
        ax.set_title(f'Outer Cluster {outer_idx}', fontsize=8, loc='left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')
        ax.legend(frameon=False, loc='upper left', fontsize=8)
        
        if outer_idx == n_clusters_outer - 1:
            ax.set_xlabel('Time', fontsize=8)
        ax.set_ylabel('SST Anomaly', fontsize=8)
        ax.tick_params(axis='both', labelsize=8)

    plt.tight_layout()
    print("\n=== Cluster Pattern Analysis Complete ===")
    return fig

def get_time_index(data):
    """
    Get time index from xarray Dataset.
    
    Parameters:
    -----------
    data : xarray.Dataset
        Dataset containing time dimension
        
    Returns:
    --------
    pandas.DatetimeIndex or numpy.ndarray
        Time index from the dataset
    """
    if 'time' in data.dims:
        return data.time.values
    elif 'time' in data.coords:
        return data.time.values
    else:
        return np.arange(data['sst'].shape[0])

def create_cluster_time_series_df(cluster_results, data, n_clusters_outer=4, n_clusters_inner=2):
    """
    Create a DataFrame containing time series for each cluster.
    
    Parameters:
    -----------
    cluster_results : dict
        Results from perform_nested_clustering
    data : xarray.Dataset
        Original data with coordinates
    n_clusters_outer : int
        Number of outer clusters
    n_clusters_inner : int
        Number of inner clusters
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with time series for each cluster, columns named as 'outer_X_inner_Y'
    """
    time_index = get_time_index(data)
    n_total_clusters = cluster_results['n_clusters_total']
    df = pd.DataFrame(index=time_index)
    
    for i in range(n_total_clusters):
        mask = cluster_results['spatial_clusters'] == i
        cluster_data = data['sst'].values[:, mask]
        cluster_mean = np.nanmean(cluster_data, axis=1)
        outer_cluster = i // n_clusters_inner
        inner_cluster = i % n_clusters_inner
        col_name = f'outer_{outer_cluster}_inner_{inner_cluster}'
        df[col_name] = cluster_mean
    
    return df