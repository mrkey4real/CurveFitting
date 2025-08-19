# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 2025

@author: qizixuan
"""
import pandas as pd

STAGE = "single"
EIR_CURVE_INPUT_COL_1 = "Air Temperature [degC]"   # T_out
EIR_CURVE_INPUT_COL_2 = "Thermostat"             # T_in
POWER_COL = "Outdoor Unit [kW]"                    # P_hvac
# --- Data filtering ---
THRESHOLD_KW = 1.4                                  # keep rows with Outdoor Unit power > this threshold
# --- Rated Parameters ---
RATED_TOTAL_COOLING_CAPACITY = 8030.147323  # W
RATED_SENSIBLE_HEAT_RATIO = 0.79
RATED_COOLING_COP = 3.568954366            # Rated cooling COP [1]
RATED_AIR_FLOW_RATE = 0.471947443          # m3/s
RATED_COOLING_EIR = 1 / RATED_COOLING_COP  # Approximately 0.280194112

def calculate_simulated_eir(row, rated_eir=RATED_COOLING_EIR):
    t_outdoor = row[EIR_CURVE_INPUT_COL_1]
    t_indoor = row[EIR_CURVE_INPUT_COL_2]
    a = 0.26754729948459177
    b = 0.04247903556154983
    c = -0.0011410021852605492
    d = -0.001979620030749466
    e = 0.0005419221739507913
    f = -0.00041962720346429
    return (a + b * t_indoor + c * t_indoor**2 + d * t_outdoor + e * t_outdoor**2 + f * t_indoor * t_outdoor)*rated_eir


def calculate_capacity_for_fitting():
    # read
    print("Reading raw data.csv...")
    data_df = pd.read_csv('../data/single/final_merged_weather_egauge.csv')
    print(f"Loaded, {len(data_df)} rows in total")
    
    # check columns
    required_cols = [EIR_CURVE_INPUT_COL_1, EIR_CURVE_INPUT_COL_2, POWER_COL]
    missing_cols = [col for col in required_cols if col not in data_df.columns]
    if missing_cols:
        print(f"error:missing columns: {missing_cols}")
        return
    
    # filter by outdoor power threshold
    before_n = len(data_df)
    data_df = data_df[data_df[POWER_COL] > THRESHOLD_KW].copy()
    after_n = len(data_df)
    print(f"Applied power threshold: {THRESHOLD_KW} kW on '{POWER_COL}'. Kept {after_n} / {before_n} rows.")
    if after_n == 0:
        print("No data after threshold filtering.")
        return
    
    print("Cal EIR...")
    data_df['EIR'] = data_df.apply(calculate_simulated_eir, axis=1)
    
    print("Cal Capacity...")
    data_df['Power_W'] = data_df[POWER_COL] * 1000  # to W
    data_df['Capacity_W'] = data_df['Power_W'] / data_df['EIR']
    
    # Create out df columns
    output_df = pd.DataFrame({
        'Date & Time': data_df['Time'],
        'EIR': data_df['EIR'],
        'Capacity_W': data_df['Capacity_W'],
        'Power_kW': data_df[POWER_COL]
    })
    
    #save
    output_filename = f'../data/single/{STAGE}_data_for_fitting.csv'
    print(f"Saving to {output_filename}...")
    output_df.to_csv(output_filename, index=False)
    
    print(f"Finished！Results saved to {output_filename}")
    print(f"Processed {len(output_df)} rows")
    
    # print head
    print("\nfront 5 rows:")
    print(output_df.head())
    
    # Statistics
    print("\nStatistics:")
    print(f"EIR range: {output_df['EIR'].min():.4f} - {output_df['EIR'].max():.4f}")
    print(f"Capacity range (W): {output_df['Capacity_W'].min():.1f} - {output_df['Capacity_W'].max():.1f}")
    print(f"Power range (kW): {output_df['Power_kW'].min():.3f} - {output_df['Power_kW'].max():.3f}")

if __name__ == "__main__":
    calculate_capacity_for_fitting()