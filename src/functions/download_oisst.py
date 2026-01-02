import xarray as xr
import pandas as pd
import os

def get_indices(lat_north, lat_south, lon_west, lon_east):
    """
    Converts latitude and longitude coordinates to dataset indices.
    
    Parameters:
    -----------
    lat_north : float
        Northern latitude boundary
    lat_south : float
        Southern latitude boundary
    lon_west : float
        Western longitude boundary
    lon_east : float
        Eastern longitude boundary
    
    Returns:
    --------
    tuple
        (west_idx, east_idx, south_idx, north_idx) indices for the dataset
    """
    lon_min, lon_max, lon_step = 0.125, 359.875, 0.25
    lat_min, lat_max, lat_step = -89.875, 89.875, 0.25

    # Convert negative longitudes to 0-360 format
    lon_west = lon_west + 360 if lon_west < 0 else lon_west
    lon_east = lon_east + 360 if lon_east < 0 else lon_east

    west_idx = int((lon_west - lon_min) / lon_step)
    east_idx = int((lon_east - lon_min) / lon_step)
    south_idx = int((lat_south - lat_min) / lat_step)
    north_idx = int((lat_north - lat_min) / lat_step)

    return west_idx, east_idx, south_idx, north_idx


def generate_opendap_url(year, west_idx, east_idx, south_idx, north_idx):
    """
    Generates the OpenDAP request URL for NOAA OISST data based on indices.
    
    Parameters:
    -----------
    year : int
        Year for which to generate the URL
    west_idx, east_idx, south_idx, north_idx : int
        Indices for the dataset boundaries
    
    Returns:
    --------
    str
        OpenDAP URL for the specified year and region
    """
    base_url = f"http://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.{year}.nc?"
    
    # Handle leap years
    days_in_year = 366 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 365
    last_day_idx = days_in_year - 1
    
    opendap_url = (
        f"{base_url}time[0:1:{last_day_idx}],"
        f"lon[{west_idx}:1:{east_idx}],"
        f"lat[{south_idx}:1:{north_idx}],"
        f"sst[0:1:{last_day_idx}][{south_idx}:1:{north_idx}][{west_idx}:1:{east_idx}]"
    )
    return opendap_url


def fetch_and_save_oisst(start_year, end_year, lat_north, lat_south, lon_west, lon_east, output_dir, output_filename=None):
    """
    Fetches OISST data from OpenDAP, concatenates across years, and saves to a NetCDF file.
    
    Parameters:
    -----------
    start_year, end_year : int
        Range of years to download
    lat_north, lat_south, lon_west, lon_east : float
        Geographic boundaries of the region of interest
    output_dir : str
        Directory where the output file will be saved
    output_filename : str, optional
        Name of the output file (default: "oisst_{start_year}_{end_year}.nc")
    
    Returns:
    --------
    str
        Path to the saved NetCDF file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Default output filename if not provided
    if output_filename is None:
        output_filename = f"oisst_{start_year}_{end_year}.nc"
    
    output_path = os.path.join(output_dir, output_filename)
    
    # Get indices for the region
    west_idx, east_idx, south_idx, north_idx = get_indices(lat_north, lat_south, lon_west, lon_east)
    datasets = []

    print(f"Downloading OISST data for years {start_year}-{end_year}")
    print(f"Region: Lat {lat_south}°-{lat_north}°, Lon {lon_west}°-{lon_east}°")
    
    for year in range(start_year, end_year + 1):
        url = generate_opendap_url(year, west_idx, east_idx, south_idx, north_idx)
        print(f"Fetching data for year {year}...")
        
        try:
            ds = xr.open_dataset(url, engine="netcdf4", chunks={'time': 1, 'lat': 50, 'lon': 50})
            datasets.append(ds)
        except Exception as e:
            print(f"Failed to fetch data for {year}: {e}")
    
    if not datasets:
        raise ValueError("No data was successfully downloaded")
    
    # Concatenate datasets along time dimension
    print("Concatenating datasets...")
    combined_ds = xr.concat(datasets, dim="time")
    
    # Save to NetCDF file
    print(f"Saving to {output_path}...")
    combined_ds.to_netcdf(output_path)
    print("Download complete!")
    
    return output_path


def fetch_and_return_datasets(start_year, end_year, lat_north, lat_south, lon_west, lon_east):
    """
    Fetches OISST data from OpenDAP and returns a list of datasets without saving to disk.
    
    Parameters:
    -----------
    start_year, end_year : int
        Range of years to download
    lat_north, lat_south, lon_west, lon_east : float
        Geographic boundaries of the region of interest
    
    Returns:
    --------
    list
        List of xarray datasets, one per year
    """
    west_idx, east_idx, south_idx, north_idx = get_indices(lat_north, lat_south, lon_west, lon_east)
    datasets = []

    print(f"Downloading OISST data for years {start_year}-{end_year}")
    print(f"Region: Lat {lat_south}°-{lat_north}°, Lon {lon_west}°-{lon_east}°")
    
    for year in range(start_year, end_year + 1):
        url = generate_opendap_url(year, west_idx, east_idx, south_idx, north_idx)
        print(f"Fetching data for year {year}...")
        
        try:
            ds = xr.open_dataset(url, engine="netcdf4", chunks={'time': 1, 'lat': 50, 'lon': 50})
            datasets.append(ds)
        except Exception as e:
            print(f"Failed to fetch data for {year}: {e}")

    return datasets


def process_and_save_by_year(datasets, years, output_dir, lat_north, lat_south, lon_west, lon_east, valid_min=-3, valid_max=45):
    """
    Process downloaded datasets by filtering valid values and saving each year separately.
    
    Parameters:
    -----------
    datasets : list
        List of xarray datasets, one per year
    years : list
        List of years corresponding to each dataset
    output_dir : str
        Directory where the output files will be saved
    lat_north, lat_south, lon_west, lon_east : float
        Geographic boundaries of the region of interest
    valid_min : float, optional
        Minimum valid SST value (default: -3)
    valid_max : float, optional
        Maximum valid SST value (default: 45)
    
    Returns:
    --------
    list
        Paths to the saved NetCDF files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create region string for filename
    region_str = f"N{lat_north:.1f}_S{abs(lat_south):.1f}_W{abs(lon_west):.1f}_E{abs(lon_east):.1f}"
    
    output_files = []
    
    for i, year in enumerate(years):
        print(f"Processing data for year {year}...")
        
        # Apply valid range mask
        masked_ds = datasets[i].where((datasets[i].sst >= valid_min) & (datasets[i].sst <= valid_max))
        masked_ds.attrs = {}  # Clear attributes to avoid conflicts
        
        # Process by month to reduce memory usage
        monthly_datasets = []
        for month in range(1, 13):
            print(f"Processing month {month}...")
            start_date = pd.Timestamp(f"{year}-{month:02d}-01")
            
            # Calculate end date (last day of the month)
            if month == 12:
                end_date = pd.Timestamp(f"{year}-12-31")
            else:
                end_date = pd.Timestamp(f"{year}-{month+1:02d}-01") - pd.Timedelta(days=1)
            
            # Select data for the month
            try:
                ds_month = masked_ds.sel(time=slice(start_date, end_date))
                
                # Load data into memory
                ds_month = ds_month.load()
                
                monthly_datasets.append(ds_month)
            except Exception as e:
                print(f"Error processing month {month}: {e}")
        
        if not monthly_datasets:
            print(f"No valid data for year {year}")
            continue
        
        # Concatenate monthly datasets
        final_ds = xr.concat(monthly_datasets, dim="time")
        
        # Save to NetCDF file with region information
        output_filename = f"sst.day.mean.{year}.{region_str}.nc"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"Saving year {year} to {output_path}...")
        final_ds.to_netcdf(output_path)
        
        output_files.append(output_path)
    
    print(f"Processing complete. Saved {len(output_files)} files.")
    return output_files


def download_and_process_oisst(start_year, end_year, lat_north, lat_south, lon_west, lon_east, output_dir, valid_min=-3, valid_max=45):
    """
    Download, process, and save OISST data by year.
    
    Parameters:
    -----------
    start_year, end_year : int
        Range of years to download
    lat_north, lat_south, lon_west, lon_east : float
        Geographic boundaries of the region of interest
    output_dir : str
        Directory where the output files will be saved
    valid_min : float, optional
        Minimum valid SST value (default: -3)
    valid_max : float, optional
        Maximum valid SST value (default: 45)
    
    Returns:
    --------
    list
        Paths to the saved NetCDF files
    """
    # Create list of years
    years = list(range(start_year, end_year + 1))
    
    # Download data
    print("Downloading OISST data...")
    datasets = fetch_and_return_datasets(
        start_year=start_year, 
        end_year=end_year, 
        lat_north=lat_north, 
        lat_south=lat_south, 
        lon_west=lon_west, 
        lon_east=lon_east
    )
    
    if not datasets:
        raise ValueError("No data was downloaded")
    
    # Process and save data by year
    print("Processing and saving data by year...")
    output_files = process_and_save_by_year(
        datasets=datasets,
        years=years,
        output_dir=output_dir,
        lat_north=lat_north,
        lat_south=lat_south,
        lon_west=lon_west,
        lon_east=lon_east,
        valid_min=valid_min,
        valid_max=valid_max
    )
    
    return output_files
