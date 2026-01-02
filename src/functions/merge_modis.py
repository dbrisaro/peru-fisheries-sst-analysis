import os
import glob
import xarray as xr
import pandas as pd

def merge_modis_files(input_dir, output_dir, pattern="AQUA_MODIS.*.L3m.DAY.SST*.nc", output_filename="sst_merged_daily_complete.nc"):
    """
    Merge multiple MODIS NetCDF files into a single file with complete time series.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing the input MODIS files
    output_dir : str
        Directory where the merged file will be saved
    pattern : str, optional
        Pattern to match MODIS files (default: "AQUA_MODIS.*.L3m.DAY.SST*.nc")
    output_filename : str, optional
        Name of the output merged file (default: "sst_merged_daily_complete.nc")
    
    Returns:
    --------
    str
        Path to the merged file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Full path for output file
    merged_file = os.path.join(output_dir, output_filename)
    
    # Find all files matching the pattern
    pattern = os.path.join(input_dir, pattern)
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise ValueError(f"No files found matching pattern: {pattern}")
    
    print(f"Found {len(files)} files to merge")
    
    # Process each file
    datasets = []
    for f in files:
        base = os.path.basename(f)
        parts = base.split('.')
        
        if len(parts) < 2:
            print(f"Unexpected filename format: {base}")
            continue
            
        date_str = parts[1]  # Expected format: "20020704" or "20250101"
        try:
            dt = pd.to_datetime(date_str, format="%Y%m%d")
        except Exception as e:
            print(f"Error converting date in {base}: {e}")
            continue
        
        # Open dataset and add time dimension
        try:
            ds = xr.open_dataset(f)
            ds = ds.expand_dims(time=[dt])
            datasets.append(ds)
        except Exception as e:
            print(f"Error processing file {base}: {e}")
            continue
    
    if not datasets:
        raise ValueError("No valid datasets found to merge")
    
    # Concatenate datasets along time dimension
    merged_ds = xr.concat(datasets, dim="time")
    
    # Create complete time index
    complete_time = pd.date_range(
        start=merged_ds.time.min().values,
        end=merged_ds.time.max().values,
        freq="D"
    )
    
    # Reindex dataset (days without data will be filled with NaN)
    merged_ds = merged_ds.reindex(time=complete_time)
    
    # Export merged dataset
    merged_ds.to_netcdf(merged_file)
    print(f"Merge completed. File saved as: {merged_file}")
    
    return merged_file


def compute_anomalies(processed_dir, var_name="sst", rolling_window=10):
    """
    Compute daily and monthly anomalies from MODIS data.
    
    Parameters:
    -----------
    processed_dir : str
        Directory containing processed MODIS files
    var_name : str, optional
        Variable name in the NetCDF files (default: "sst", use "chlor_a" for chlorophyll)
    rolling_window : int, optional
        Size of the rolling window for smoothing (default: 10)
    
    Returns:
    --------
    tuple
        Paths to the daily and monthly anomaly files
    """
    # Define file paths based on variable name
    prefix = "sst" if var_name == "sst" else "chl"
    
    clim_daily_file = os.path.join(processed_dir, f"{prefix}_climatology_daily_2003_2023.nc")
    clim_monthly_file = os.path.join(processed_dir, f"{prefix}_climatology_monthly_2003_2023.nc")
    obs_daily_file = os.path.join(processed_dir, f"{prefix}_merged_daily_complete.nc")
    obs_monthly_file = os.path.join(processed_dir, f"{prefix}_merged_monthly_2003_2024.nc")
    
    anom_daily_file = os.path.join(processed_dir, f"{prefix}_anomaly_daily_2002_2025.nc")
    anom_monthly_file = os.path.join(processed_dir, f"{prefix}_anomaly_monthly_2003_2024.nc")
    
    # Check if input files exist
    for file_path in [clim_daily_file, clim_monthly_file, obs_daily_file, obs_monthly_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    print("Computing daily anomalies...")
    # Climatology daily with rolling mean
    clim_daily = xr.open_dataset(clim_daily_file)[var_name]
    clim_daily_smoothed = clim_daily.rolling(time=rolling_window, center=True, min_periods=1).mean()
    
    # Daily anomalies
    obs_daily = xr.open_dataset(obs_daily_file)[var_name]
    clim_by_doy = clim_daily_smoothed.groupby("time.dayofyear").mean("time")
    anom_daily = obs_daily.rolling(time=rolling_window, center=True, min_periods=1).mean().groupby("time.dayofyear") - clim_by_doy
    anom_daily.to_netcdf(anom_daily_file)
    print(f"Daily anomalies saved to: {anom_daily_file}")
    
    print("Computing monthly anomalies...")
    # Monthly anomalies
    obs_monthly = xr.open_dataset(obs_monthly_file)[var_name]
    clim_monthly = xr.open_dataset(clim_monthly_file)[var_name]
    anom_monthly = obs_monthly - clim_monthly
    anom_monthly.to_netcdf(anom_monthly_file)
    print(f"Monthly anomalies saved to: {anom_monthly_file}")
    
    return anom_daily_file, anom_monthly_file

