# Peruvian Fisheries: SST impact analysis

This project analyzes how sea surface temperature (SST) anomalies affect Peruvian fishing ports, specifically focusing on port closures and their economic impact on fisheries production.

## Overview

The analysis examines the relationship between SST anomalies and port operations in Peru's fishing industry. By combining satellite-derived SST data with fisheries production records, we assess the economic losses associated with port closures during periods of extreme SST conditions.

## Project Structure

```
peru_produccion/
├── data/                    # Project data (not included in repository)
│   ├── MODIST/             # Sea surface temperature data
│   ├── imarpe/             # Fisheries production data
│   └── puertos/            # Port data
├── results/                 # Analysis results (not included in repository)
├── src/                     # Main analysis scripts
│   ├── port_sst_analysis_functions.py
│   ├── port_sst_analysis_quick.py
│   └── port_sst_analysis_weekly.py
├── code/                    # Supporting code and utilities
│   ├── functions/          # Helper functions
│   ├── notebooks/          # Jupyter notebooks for analysis
│   ├── observed_sst_plots_all_ports.py
│   ├── modis_chl_dbr_process_nc.sh
│   ├── modis_sst_dbr_process_nc.sh
│   └── sst_dbr_process_nc.sh
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/dbrisaro/peru-fisheries-sst-analysis.git
cd peru-fisheries-sst-analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Requirements

The analysis requires three main data sources:

1. **SST Data**:
   - Location: `data/MODIS/processed/sst_anomaly_daily_2002_2025.nc`
   - Daily SST anomalies from MODIS satellite data
   - Covers the period 2002-2025
   - Format: NetCDF (.nc)

2. **Production Data**:
   - Location: `data/imarpe/processed/df_produccion_combined_2002_2024_clean.csv`
   - Daily fisheries production records by port
   - Provided by IMARPE (Peruvian Marine Research Institute)
   - Format: CSV

3. **Port Data**:
   - Location: `data/puertos/processed/ports_with_normal_angles_corrected.xlsx`
   - Port locations and operational parameters
   - Includes normal angles for each port
   - Format: Excel (.xlsx)

Please contact the project administrator to obtain access to these data files.

## Analysis Scripts

The project includes two main analysis approaches:

1. **Daily Analysis** (`src/port_sst_analysis_quick.py`):
   - Examines daily SST anomalies and their impact
   - Provides high-resolution temporal analysis
   - Generates daily loss estimates

2. **Weekly Analysis** (`src/port_sst_analysis_weekly.py`):
   - Aggregates data to weekly time steps
   - Reduces noise in the analysis
   - Offers broader temporal patterns

3. **MODIS Data Processing**:
```bash
bash code/modis_sst_dbr_process_nc.sh
```
This script processes raw MODIS SST data into the required format.

## Output

The analysis generates two main types of results:

1. **Annual Losses** (`annual_losses.csv`):
   - Total economic losses per year
   - Port-specific breakdowns
   - Statistical summaries

2. **Exceedance Plots** (`annual_losses_exceedance.png`):
   - Visual representation of loss probabilities
   - Helps identify extreme events
   - Supports risk assessment

Results are saved in the `results/` directory, organized by analysis type (daily/weekly).

## Contributing

We welcome contributions to improve the analysis. If you'd like to contribute:

1. Fork the repository
2. Create a branch for your changes
3. Submit a pull request with a clear description of your modifications

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 