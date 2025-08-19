## CurveFitting: Data-Driven HVAC Curve Modeling (Single-Stage)

This repository implements a data-driven workflow to build and validate cooling performance curves (capacity and EIR) for a single-stage heat pump using measured weather and eGauge power data. It also provides tooling to generate EPW weather blocks from minute data and to compare the curve-based power against EnergyPlus results. The `multi/` folder is work-in-progress; most functionality is under `single/`.

### Repository Layout
- `single/`: end-to-end pipeline for the single-stage model
- `data/`: local datasets (not tracked by Git)
- `fixed_ep_model/`: EnergyPlus model and outputs for reference
- `multi/`: multi-stage (WIP)

### Data Overview (brief)
Place your local datasets under `data/`. Key inputs used by the single-stage pipeline:
- Weather (East/West house, CSVs under `data/weather/East` and `data/weather/West`)
- Open-Meteo archive (downloaded via `single/open_meteo.py` to `data/open_meteo_data.csv`)
- eGauge power (e.g., `data/single/egauge/egauge_1min_0601_0830_2024.csv`)

Required columns in merged data are explicit and not guessed. Commonly used names include:
- `Time` (minute-level timestamps)
- Weather: `Air Temperature [degC]`, `Relative Humidity [%]`, `Relative Air Pressure [kPa]`, `Wind Direction [deg]`, `Wind Speed [m/s]`, `Dew Point [degC]`, `Total [W/m2]`, `Diffuse [W/m2]`, Open-Meteo-derived columns such as `OM_DNI(W/m2)`, `OM_rain`, `OM_snowfall`, `OM_snow_depth`
- Indoor: `Thermostat`
- eGauge: `Indoor Unit [kW]`, `Outdoor Unit [kW]`

The pipeline keeps column naming strict to avoid silent errors. If a required column is missing, the scripts will stop with clear messages.

### Environment
- Python 3.10+
- Install packages:
  - `pandas`, `numpy`, `matplotlib`, `scipy`
  - `openmeteo-requests`, `requests-cache`, `retry-requests`

Example (pip):
```bash
pip install pandas numpy matplotlib scipy openmeteo-requests requests-cache retry-requests
```

### Single-Stage Workflow
The single-stage pipeline is designed to be edited and run directly in an IDE. No CLI args or implicit defaults are used; configuration lives at the top of each script.

1) Prepare and merge data (weather, Open‑Meteo, eGauge) → generate EPW blocks, inspect power distribution
- File: `single/preprocess.py`
  - `filter_weather(...)`: cleans and consolidates East/West weather days with complete hourly coverage
  - `merge_weather_pair_to_minute(...)`: aligns two weather sources to a 1‑min grid
  - `merge_openmeteo_to_minute(...)`: upsamples Open‑Meteo hourly to 1‑min and merges into weather
  - `merge_weather_with_egauge(...)`: merges weather with `Outdoor/Indoor Unit [kW]`
  - `write_epw_blocks(...)`: writes EPW segments for continuous complete days from `final_merged_weather_egauge.csv`
  - `distribution_analysis(...)`: histogram/percentile views of outdoor unit power to help select thresholds
  - `calculate_capacity_for_fitting(min_threshold, upper_perc, lower_perc)`: computes EIR from curves and derives `Capacity_W`, saving `data/single/single_data_for_fitting.csv`
  - `resample_fitting_data(..., time_interval)`: produces aggregated datasets like `..._30min.csv`

2) Compute capacity dataset directly (alternate entry)
- File: `single/calculate_capacity.py`
  - Reads `data/final_merged_weather_egauge.csv`, computes `EIR`, `Capacity_W`, writes `data/single/single_data_for_fitting.csv`

3) Fit cooling curves (biquadratic in indoor/outdoor temperatures)
- File: `single/single_fit_cooling.py`
  - Fits EIR and capacity normalized to rated points
  - Reports coefficients, R², CVRMSE and saves plots:
    - `single_scatter_plot.png`, `single_model_accuracy.png`

4) Compare actual vs simulated (and vs EnergyPlus)
- File: `single/comparison_power.py`
  - `compare_real_curve()`: compare curve-simulated power vs actual `Power_kW` from fitting CSV
  - `compare_ep_real_curve()`: align minute data with EnergyPlus, resample to multiple intervals, plot time series and compute metrics; outputs:
    - `timeseries_{1min|5min|15min|30min|60min}.png`
    - `metrics_summary.csv`, `hourly_energy_values_*.csv`, `daily_energy_values_*.csv`, and their metrics

5) Analyze power thresholds for HVAC on/off discrimination
- File: `single/power_threshold_analysis.py`
  - Percentiles, retention vs threshold, distribution plots → `power_threshold_analysis.png`

6) Fetch Open‑Meteo archive (optional)
- File: `single/open_meteo.py`
  - Downloads selected variables to `data/open_meteo_data.csv` with caching and retry

### Rated Parameters and Curve Form
Rated values (capacity, COP/EIR, airflow) are set at the top of scripts. EIR and capacity curves follow a biquadratic form in indoor/wetbulb (`Thermostat`) and outdoor/drybulb (`Air Temperature [degC]`), normalized by rated values where noted.

### EnergyPlus Notes
`single/comparison_power.py` expects an EnergyPlus Excel with an explicit column name:
`DAIKIN ONE STAGE HP COOLING COIL:Cooling Coil Electricity Rate [W](TimeStep)`
and a resolvable `Date & Time` column. The script builds aligned series for Real, Curve, and EP and computes metrics at multiple aggregation levels.

### Typical Run Sequence (IDE)
1. Edit constants at the top of scripts (paths, thresholds, rated parameters, column names)
2. Run `single/preprocess.py` (or `single/open_meteo.py` first to download Open‑Meteo)
3. Run `single/calculate_capacity.py` or `single/preprocess.py`-embedded `calculate_capacity_for_fitting(...)`
4. (Optional) `resample_fitting_data(..., "30min")`
5. Run `single/single_fit_cooling.py`
6. Run `single/comparison_power.py` to compare vs actual and EnergyPlus
7. (Optional) Run `single/power_threshold_analysis.py`

### Outputs
- `data/single/single_data_for_fitting.csv` (+ resampled variants)
- Plots: `single_scatter_plot.png`, `single_model_accuracy.png`, `power_comparison_actual_vs_simulated.png`, `timeseries_*.png`, `power_threshold_analysis.png`
- Metrics: `metrics_summary.csv`, hourly/daily energy values and metrics CSVs
- EPW files: `data/weather/college_station_YYYY-MM-DD_to_YYYY-MM-DD.epw`

### Multi-Stage (WIP)
The `multi/` folder contains early exploration for multi‑stage heat pumps and is not yet documented.

### Versioning / Git
The `data/` folder is excluded from version control by default (large local datasets). Share only the scripts and generated coefficients/plots/metrics as needed.


