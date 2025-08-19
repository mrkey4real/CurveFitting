import os
import glob
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', message='Could not infer format', category=UserWarning)

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

#%% func
def filter_weather(folder_path: str) -> List[str]:
    csv_paths = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    csv_paths = [p for p in csv_paths if os.path.basename(p).lower() != "merged_weather_data.csv"]
    frames = []

    for path in csv_paths:
        df = pd.read_csv(path, sep='\t', encoding='utf-8-sig', low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        if "Time" not in df.columns:
            df = pd.read_csv(path, sep=',', encoding='utf-8-sig', low_memory=False)
            df.columns = [str(c).strip() for c in df.columns]
        if "Time" not in df.columns:
            df = pd.read_csv(path, sep=None, engine='python', encoding='utf-8-sig', low_memory=False)
            df.columns = [str(c).strip() for c in df.columns]
        if "Time" not in df.columns:
            continue
        dt = pd.to_datetime(df["Time"], errors="coerce")
        df = df.loc[~dt.isna()].copy()
        df["Time"] = dt.dt.floor("min")
        frames.append(df)

    if len(frames) == 0:
        output_path = os.path.join(folder_path, "merged_weather_data.csv")
        pd.DataFrame().to_csv(output_path, index=False)
        print([])
        return []

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merged.drop_duplicates(subset=["Time"], keep="first")
    merged = merged.sort_values("Time").reset_index(drop=True)

    # ensure each kept day has data in every hour (0-23)
    group = merged.groupby([merged["Time"].dt.date, merged["Time"].dt.hour]).size().unstack()
    group = group.reindex(columns=list(range(24)), fill_value=0)
    if group.shape[0] == 0:
        output_path = os.path.join(folder_path, "merged_weather_data.csv")
        merged.to_csv(output_path, index=False)
        print([])
        return []

    complete_dates = set(group.index[(group > 0).all(axis=1)])
    keep_mask = merged["Time"].dt.date.isin(complete_dates)
    filtered = merged.loc[keep_mask].copy()
    filtered = filtered.sort_values("Time").reset_index(drop=True)

    output_path = os.path.join(folder_path, "merged_weather_data.csv")
    filtered.to_csv(output_path, index=False)

    day_list = [f"{d.month}/{d.day}" for d in sorted(complete_dates)]
    print(day_list)
    return filtered

def merge_weather_pair_to_minute(df1, df2, columns1: List[str], columns2: List[str]):
    # normalize time to minute
    df1 = df1.copy()
    df2 = df2.copy()

    t1 = pd.to_datetime(df1["Time"], errors="coerce").dt.floor("min")
    t2 = pd.to_datetime(df2["Time"], errors="coerce").dt.floor("min")

    df1 = df1.loc[~t1.isna()].copy()
    df2 = df2.loc[~t2.isna()].copy()
    df1["Time"] = t1.loc[df1.index]
    df2["Time"] = t2.loc[df2.index]

    df1 = df1.drop_duplicates(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    df2 = df2.drop_duplicates(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    if df1.shape[0] == 0 or df2.shape[0] == 0:
        output_dir = os.path.join(os.path.dirname(__file__), "data", "weather")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "final_merged_weather.csv")
        pd.DataFrame().to_csv(output_path, index=False, encoding="utf-8-sig")
        return pd.DataFrame()

    start_time = max(df1["Time"].min(), df2["Time"].min())
    end_time = min(df1["Time"].max(), df2["Time"].max())

    if start_time > end_time:
        output_dir = os.path.join(os.path.dirname(__file__), "data", "weather")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "final_merged_weather.csv")
        pd.DataFrame().to_csv(output_path, index=False, encoding="utf-8-sig")
        return pd.DataFrame()

    minute_grid = pd.DataFrame({"Time": pd.date_range(start_time.floor("min"), end_time.floor("min"), freq="1min")})

    left = pd.merge_asof(
        minute_grid.sort_values("Time"),
        df1[["Time"] + columns1].sort_values("Time"),
        on="Time",
        direction="nearest",
        tolerance=pd.Timedelta("15min"),
    )
    right = pd.merge_asof(
        minute_grid.sort_values("Time"),
        df2[["Time"] + columns2].sort_values("Time"),
        on="Time",
        direction="nearest",
        tolerance=pd.Timedelta("15min"),
    )

    left_ok = left[columns1].notna().any(axis=1)
    right_ok = right[columns2].notna().any(axis=1)
    keep = left_ok & right_ok

    kept_time = left.loc[keep, ["Time"]].reset_index(drop=True)
    sel_left = left.loc[keep, columns1].reset_index(drop=True)
    sel_right = right.loc[keep, columns2].reset_index(drop=True)

    sel_left.columns = [c for c in columns1]
    sel_right.columns = [c for c in columns2]

    merged = pd.concat([kept_time, sel_left, sel_right], axis=1)
    merged = merged.sort_values("Time").reset_index(drop=True)

    # fix negative values: replace <0 by mean of previous and next valid values; if not available, set 0
    numeric_cols = [c for c in merged.columns if c != "Time" and pd.api.types.is_numeric_dtype(merged[c])]
    for c in numeric_cols:
        s = merged[c]
        if (s < 0).any():
            s_corr = s.mask(s < 0)
            prev = s_corr.ffill()
            nxt = s_corr.bfill()
            mean_val = (prev + nxt) / 2
            s_new = s.copy()
            s_new[s < 0] = mean_val[s < 0]
            s_new = s_new.fillna(0)
            merged[c] = s_new

    output_dir = "../data/weather"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "two_merged_weather.csv")
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    return merged

def merge_openmeteo_to_minute(base_csv: str,
                              om_csv: str,
                              out_csv: str,
                              prefix: str = "OM_",
                              tz_local: str = "America/Chicago"):
    # ---------- 读入 ----------
    base = pd.read_csv(base_csv)
    om   = pd.read_csv(om_csv)

    def _find_time_col(cols):
        for k in ["Time","time","datetime","DateTime","timestamp","date","Date"]:
            if k in cols: return k
        for c in cols:
            if "time" in str(c).lower(): return c
        raise ValueError("No datetime column found.")

    def _parse_to_local_minute(s, tz_local):
        t = pd.to_datetime(s, errors="coerce")
        # tz-aware -> convert to local, then drop tz
        try:
            if getattr(t.dt, "tz", None) is not None:
                t = t.dt.tz_convert(tz_local).dt.tz_localize(None)
        except Exception:
            # if it's naive or cannot convert, treat as local wall-clock
            try:
                t = t.dt.tz_localize(None)
            except Exception:
                pass
        return t.dt.floor("min")

    # ---------- 标准化时间 ----------
    bt = _find_time_col(base.columns); ot = _find_time_col(om.columns)

    base["Time"] = _parse_to_local_minute(base[bt], tz_local)
    base = (base.loc[~base["Time"].isna()]
                .drop_duplicates("Time")
                .sort_values("Time")
                .set_index("Time"))

    om["Time"] = _parse_to_local_minute(om[ot], tz_local)
    om = (om.loc[~om["Time"].isna()]
            .drop_duplicates("Time")
            .sort_values("Time")
            .set_index("Time"))

    # ---------- Open-Meteo 数值列 -> 1min 插值 ----------
    if om.shape[0] == 0:
        out = base.reset_index()
        out.to_csv(out_csv, index=False, encoding="utf-8-sig")
        return out

    om = om.apply(pd.to_numeric, errors="ignore")
    num_cols = om.select_dtypes(include="number").columns.tolist()
    om_1min = (om[num_cols]
               .resample("1min").asfreq()
               .interpolate(method="time", limit_area="inside")) if num_cols else pd.DataFrame(index=base.index)

    # ---------- 对齐 + 合并 ----------
    om_1min = om_1min.reindex(base.index)
    om_1min.columns = [prefix + c if (prefix + c) not in base.columns else f"{prefix}{c}" for c in om_1min.columns]

    out = pd.concat([base, om_1min], axis=1).reset_index()
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out

def merge_weather_with_egauge(df_weather, df_egauge, weather_cols: List[str], egauge_cols: List[str]):
    dfw = df_weather.copy()
    dfe = df_egauge.copy()

    tw = pd.to_datetime(dfw["Time"], errors="coerce").dt.floor("min")
    te = pd.to_datetime(dfe["Time"], errors="coerce").dt.floor("min")

    dfw = dfw.loc[~tw.isna()].copy()
    dfe = dfe.loc[~te.isna()].copy()
    dfw["Time"] = tw.loc[dfw.index]
    dfe["Time"] = te.loc[dfe.index]

    dfw = dfw.drop_duplicates(subset=["Time"]).sort_values("Time").reset_index(drop=True)
    dfe = dfe.drop_duplicates(subset=["Time"]).sort_values("Time").reset_index(drop=True)

    if dfw.shape[0] == 0 or dfe.shape[0] == 0:
        output_dir = "../data"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "final_merged_weather.csv")
        pd.DataFrame().to_csv(output_path, index=False, encoding="utf-8-sig")
        print([])
        return pd.DataFrame()

    start_time = max(dfw["Time"].min(), dfe["Time"].min())
    end_time = min(dfw["Time"].max(), dfe["Time"].max())
    if start_time > end_time:
        output_dir = "../data"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "final_merged_weather.csv")
        pd.DataFrame().to_csv(output_path, index=False, encoding="utf-8-sig")
        print([])
        return pd.DataFrame()

    minute_grid = pd.DataFrame({"Time": pd.date_range(start_time.floor("min"), end_time.floor("min"), freq="1min")})

    left = pd.merge_asof(
        minute_grid.sort_values("Time"),
        dfw[["Time"] + weather_cols].sort_values("Time"),
        on="Time",
        direction="nearest",
        tolerance=pd.Timedelta("15min"),
    )
    right = pd.merge_asof(
        minute_grid.sort_values("Time"),
        dfe[["Time"] + egauge_cols].sort_values("Time"),
        on="Time",
        direction="nearest",
        tolerance=pd.Timedelta("15min"),
    )

    left_ok = left[weather_cols].notna().any(axis=1)
    right_ok = right[egauge_cols].notna().any(axis=1)
    keep = left_ok & right_ok

    kept_time = left.loc[keep, ["Time"]].reset_index(drop=True)
    sel_left = left.loc[keep, weather_cols].reset_index(drop=True)
    sel_right = right.loc[keep, egauge_cols].reset_index(drop=True)

    merged = pd.concat([kept_time, sel_left, sel_right], axis=1)
    merged = merged.sort_values("Time").reset_index(drop=True)

    # compute complete days list (every hour has data)
    group = merged.groupby([merged["Time"].dt.date, merged["Time"].dt.hour]).size().unstack()
    group = group.reindex(columns=list(range(24)), fill_value=0)
    complete_dates = []
    if group.shape[0] > 0:
        complete_dates = [f"{d.month}/{d.day}" for d in group.index[(group > 0).all(axis=1)]]

    output_dir = "../data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "final_merged_weather_egauge.csv")
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(complete_dates)
    return merged

def distribution_analysis(df):
    print("=== Power Threshold Analysis ===")
    # Load data
    POWER_COL = "Outdoor Unit [kW]"
    power_data = df[POWER_COL]
    
    print(f"Total data points: {len(power_data):,}")
    print(f"Power range: {power_data.min():.6f} to {power_data.max():.3f} kW")
    print(f"Mean power: {power_data.mean():.3f} kW")
    print(f"Median power: {power_data.median():.6f} kW")
    
    # Key statistics
    print(f"\nKey Percentiles:")
    percentiles = [0, 5, 10, 15, 20, 25, 30, 50, 75, 90, 95, 99, 100]
    for p in percentiles:
        value = power_data.quantile(p/100)
        print(f"  {p:2d}th percentile: {value:.3f} kW")
    
    # Threshold analysis
    print(f"\nThreshold Impact Analysis:")
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    print(f"{'Threshold':<10} {'Remaining':<12} {'Percentage':<12} {'Removed':<12}")
    print("-" * 50)
    
    for thresh in thresholds:
        remaining = (power_data > thresh).sum()
        percentage = 100 * remaining / len(power_data)
        removed = len(power_data) - remaining
        print(f"{thresh:<10.2f} {remaining:<12,} {percentage:<12.1f}% {removed:<12,}")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Overall histogram
    axes[0, 0].hist(power_data, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Power (kW)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Overall Power Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Log scale histogram
    axes[0, 1].hist(power_data[power_data > 0], bins=100, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_xlabel('Power (kW)')
    axes[0, 1].set_ylabel('Frequency (Log Scale)')
    axes[0, 1].set_title('Power Distribution (Log Scale, >0)')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Low power zoom (0-1 kW)
    low_power = power_data[power_data <= 1.0]
    axes[0, 2].hist(low_power, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 2].set_xlabel('Power (kW)')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Low Power Distribution (0-1 kW)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Cumulative distribution
    sorted_power = np.sort(power_data)
    cumulative = np.arange(1, len(sorted_power) + 1) / len(sorted_power)
    axes[1, 0].plot(sorted_power, cumulative, linewidth=2, color='red')
    axes[1, 0].set_xlabel('Power (kW)')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].set_title('Cumulative Distribution Function')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add threshold lines
    common_thresholds = [0.1, 0.2, 0.4, 0.5, 0.8]
    for thresh in common_thresholds:
        cumul_at_thresh = np.sum(sorted_power <= thresh) / len(sorted_power)
        axes[1, 0].axvline(x=thresh, color='blue', linestyle='--', alpha=0.7)
        axes[1, 0].axhline(y=cumul_at_thresh, color='blue', linestyle='--', alpha=0.7)
        axes[1, 0].text(thresh, cumul_at_thresh + 0.05, f'{thresh:.1f}kW\n{cumul_at_thresh:.1%}', 
                       ha='center', va='bottom', fontsize=8)
    
    # 5. Data retention vs threshold
    test_thresholds = np.arange(0, 2.1, 0.05)
    remaining_percentages = []
    
    for threshold in test_thresholds:
        remaining = (power_data > threshold).sum()
        percentage = 100 * remaining / len(power_data)
        remaining_percentages.append(percentage)
    
    axes[1, 1].plot(test_thresholds, remaining_percentages, linewidth=2, marker='o', markersize=3)
    axes[1, 1].set_xlabel('Threshold (kW)')
    axes[1, 1].set_ylabel('Remaining Data (%)')
    axes[1, 1].set_title('Data Retention vs Threshold')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add threshold recommendations
    recommended_thresholds = [0.1, 0.2, 0.4, 0.5, 0.8]
    colors = ['green', 'blue', 'orange', 'red', 'purple']
    
    for thresh, color in zip(recommended_thresholds, colors):
        if thresh <= max(test_thresholds):
            idx = np.argmin(np.abs(test_thresholds - thresh))
            axes[1, 1].axvline(x=thresh, color=color, linestyle='--', alpha=0.7)
            axes[1, 1].text(thresh, remaining_percentages[idx] + 2, f'{thresh:.1f}kW', 
                           rotation=90, ha='center', va='bottom', color=color, fontweight='bold')
    
    # 6. Power vs time (sampled)
    sample_indices = range(0, len(power_data), 60)  # Every hour
    sample_power = power_data.iloc[sample_indices]
    sample_time = np.arange(len(sample_power))
    
    axes[1, 2].plot(sample_time, sample_power.values, alpha=0.7, linewidth=0.5)
    axes[1, 2].set_xlabel('Time (Hours)')
    axes[1, 2].set_ylabel('Power (kW)')
    axes[1, 2].set_title('Power Time Series (Hourly Samples)')
    axes[1, 2].grid(True, alpha=0.3)
    
    # Add threshold lines
    for thresh in recommended_thresholds[:3]:  # Show only first 3 to avoid clutter
        axes[1, 2].axhline(y=thresh, color='red', linestyle='--', alpha=0.5)
        axes[1, 2].text(0, thresh, f'{thresh:.1f}kW', va='bottom', ha='left', 
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('power_threshold_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nPlots saved as: power_threshold_analysis.png")
    plt.show()

#%% EPW blocks from discontinuous days — no interpolation, no defaults.
import pandas as pd
from pathlib import Path

def write_epw_blocks(
    src="../data/final_merged_weather_egauge.csv",
    out_dir="../data/weather",
):
    SRC = Path(src)
    OUTDIR = Path(out_dir); OUTDIR.mkdir(parents=True, exist_ok=True)

    # ----- explicit column mapping (按你的列名，不猜测) -----
    MAP = {
        "time": "Time",
        "drybulb_C": "Air Temperature [degC]",
        "relhum_pct": "Relative Humidity [%]",
        "press_kPa": "Relative Air Pressure [kPa]",
        "ghi_Wm2": "Total [W/m2]",
        "dhi_Wm2": "Diffuse [W/m2]",
        "dni_Wm2": "OM_DNI(W/m2)",
        "wind_dir_deg": "Wind Direction [deg]",
        "wind_spd_mps": "Wind Speed [m/s]",
        # optional
        "dewp_C": "Dew Point [degC]",      # 如无此列，将写 EPW 缺测标记
        "precip_mm": "OM_rain",            # 如无此列，不造数据，写缺测标记
    }

    # ----- read & index -----
    df = pd.read_csv(SRC, parse_dates=[MAP["time"]])
    df = df.sort_values(MAP["time"]).set_index(MAP["time"])

    # 小工具：构造“小时-期末”索引
    def hour_ending_index(dt_min, dt_max):
        start = (dt_min.floor("h") + pd.Timedelta(hours=1)) if (dt_min != dt_min.floor("H")) else (dt_min + pd.Timedelta(hours=1))
        end   = dt_max.floor("h") + pd.Timedelta(hours=1)
        return pd.date_range(start, end, freq="h")

    hr_index = hour_ending_index(df.index.min(), df.index.max())

    # 聚合器（不插值）
    def rmean(col):
        s = pd.to_numeric(df[col], errors="coerce")
        return s.resample("h", label="right", closed="right").mean().reindex(hr_index)
    def rsum(col):
        s = pd.to_numeric(df[col], errors="coerce")
        return s.resample("h", label="right", closed="right").sum().reindex(hr_index)

    # 必需字段（只要有一个小时缺，就断开）
    drybulb = rmean(MAP["drybulb_C"])
    relhum  = rmean(MAP["relhum_pct"]).clip(0, 100)
    press   = rmean(MAP["press_kPa"]) * 1000.0          # kPa -> Pa
    ghi     = rmean(MAP["ghi_Wm2"])              # W/m2 -> Wh/m2
    dhi     = rmean(MAP["dhi_Wm2"])
    dni     = rmean(MAP["dni_Wm2"])
    wdir    = (rmean(MAP["wind_dir_deg"]) % 360.0)
    wspd    = rmean(MAP["wind_spd_mps"])

    # 可选字段（若无列，就用 EPW 缺测标记）
    dewp    = rmean(MAP["dewp_C"]) if MAP["dewp_C"] in df.columns else pd.Series(pd.NA, index=hr_index)
    precip  = rsum(MAP["precip_mm"]) if MAP["precip_mm"] in df.columns else pd.Series(pd.NA, index=hr_index)

    # 组合并形成“有效小时”掩码
    H = pd.DataFrame({
        "drybulb": drybulb,
        "relhum": relhum,
        "press": press,
        "ghi": ghi,
        "dni": dni,
        "dhi": dhi,
        "wdir": wdir,
        "wspd": wspd,
        "dewp": dewp,
        "precip": precip,
    })

    required = ["drybulb","relhum","press","ghi","dni","dhi","wdir","wspd"]
    valid_mask = H[required].notna().all(axis=1)

    # 仅保留有效小时，并计算对应的“有效日”和“EPW 小时(1..24)”
    if not valid_mask.any():
        raise RuntimeError("没有任何完整小时可用于导出。请检查必需列。")

    Hv = H.loc[valid_mask].copy()
    idx = Hv.index
    eff_date = []
    epw_hour = []
    for ts in idx:
        if ts.hour == 0:
            eff_date.append((ts - pd.Timedelta(days=1)).date())
            epw_hour.append(24)
        else:
            eff_date.append(ts.date())
            epw_hour.append(ts.hour)
    Hv["_eff_date"] = eff_date
    Hv["_epw_hour"] = epw_hour

    # 找到“完整天”（小时包含 1..24）
    hours_by_day = Hv.groupby("_eff_date")["_epw_hour"].agg(lambda s: set(s))
    complete_days = sorted([d for d, hs in hours_by_day.items() if all(h in hs for h in range(1, 25))])
    if len(complete_days) == 0:
        raise RuntimeError("没有任何完整天(24小时齐全)可用于导出。")

    # 将完整天切分为连续日期区间
    runs = []
    run = [complete_days[0]]
    for d in complete_days[1:]:
        if (pd.Timestamp(d) - pd.Timestamp(run[-1])).days == 1:
            run.append(d)
        else:
            runs.append(run)
            run = [d]
    runs.append(run)

    # ---- EPW 固定字段（保持简洁） ----
    lat, lon, tz, elev = 30.628, -96.334, -6.0, 96.0
    VIS_km, CEIL_m = 30, 77777
    PW_mm, AOD_mil = 20, 0
    SNOW_cm, DAYS_SNOW, ALBEDO = 0, 99, 0.2

    # EPW 缺测标记（官方建议值）
    M_T = 99.9        # temperature类
    M_P = 999999      # pressure Pa
    M_RAD = 9999      # radiation Wh/m2
    M_WDIR = 999      # wind dir
    M_WSPD = 99.9     # wind spd
    M_RAIN = 99.0     # precipitation mm
    ETR_H = M_RAD
    ETR_N = M_RAD
    H_IR  = M_RAD
    ILL   = 0
    written = []

    for k, days in enumerate(runs, start=1):
        start_day = pd.Timestamp(days[0])
        end_day = pd.Timestamp(days[-1])
        name = f"college_station_{start_day.date()}_to_{end_day.date()}.epw"
        outp = OUTDIR / name

        header = [
            f"LOCATION,College Station,TX,USA,UserCSV,0,{lat:.3f},{lon:.3f},{tz:.1f},{elev:.1f}",
            "DESIGN CONDITIONS,0",
            "TYPICAL/EXTREME PERIODS,0",
            "GROUND TEMPERATURES,0",
            "HOLIDAYS/DAYLIGHT SAVINGS,No,0,0,0",
            "COMMENTS 1,Generated strictly from source (no interpolation, no fabricated hours).",
            "COMMENTS 2,Hourly = minute-mean/sum → hour-ending EPW units; only hours with all required fields kept.",
            f"DATA PERIODS,1,1,Data,{start_day.day_name()},{start_day.month}/{start_day.day},{end_day.month}/{end_day.day}"
        ]

        with outp.open("w", newline="") as f:
            for line in header: f.write(line + "\n")
            for d in days:
                for hh in range(1, 25):
                    sel = Hv[(Hv["_eff_date"] == d) & (Hv["_epw_hour"] == hh)]
                    if sel.empty:
                        continue
                    row = sel.iloc[0]

                    Y, M, Dd = pd.Timestamp(d).year, pd.Timestamp(d).month, pd.Timestamp(d).day
                    Hh, mm = hh, 60

                    dry = round(float(row["drybulb"]), 1)
                    dwp = (round(float(row["dewp"]), 1) if pd.notna(row["dewp"]) else M_T)
                    rh  = int(round(float(row["relhum"])))
                    prs = int(round(float(row["press"])) ) if pd.notna(row["press"]) else M_P
                    GHI = int(round(float(row["ghi"])) ) if pd.notna(row["ghi"]) else M_RAD
                    DNI = int(round(float(row["dni"])) ) if pd.notna(row["dni"]) else M_RAD
                    DHI = int(round(float(row["dhi"])) ) if pd.notna(row["dhi"]) else M_RAD
                    wdr = int(round(float(row["wdir"])) ) if pd.notna(row["wdir"]) else M_WDIR
                    wsp = round(float(row["wspd"]), 1) if pd.notna(row["wspd"]) else M_WSPD
                    rmm = (round(float(row["precip"]), 2) if pd.notna(row["precip"]) else M_RAIN)

                    cols = [
                        Y, M, Dd, Hh, mm, 0,
                        dry, dwp, rh, prs,
                        ETR_H, ETR_N, H_IR,
                        GHI, DNI, DHI,
                        ILL, ILL, ILL, ILL,
                        wdr, wsp,
                        0, 0,
                        VIS_km, CEIL_m,
                        0, 0,
                        PW_mm, AOD_mil,
                        SNOW_cm, DAYS_SNOW,
                        ALBEDO,
                        rmm,
                        1
                    ]
                    f.write(",".join(map(str, cols)) + "\n")

        written.append(outp.as_posix())

    return written


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


def calculate_capacity_for_fitting(min_threshold, upper_perc, lower_perc):
    # read
    print("Reading raw data.csv...")
    data_df = pd.read_csv('../data/final_merged_weather_egauge.csv')
    print(f"Loaded, {len(data_df)} rows in total")
    
    # check columns
    required_cols = [EIR_CURVE_INPUT_COL_1, EIR_CURVE_INPUT_COL_2, POWER_COL]
    missing_cols = [col for col in required_cols if col not in data_df.columns]
    if missing_cols:
        print(f"error:missing columns: {missing_cols}")
        return
    
    # filter by outdoor power threshold
    before_n = len(data_df)
    data_df = data_df[data_df[POWER_COL] > min_threshold].copy()
    after_n = len(data_df)
    print(f"Applied power threshold: {min_threshold} kW on '{POWER_COL}'. Kept {after_n} / {before_n} rows.")
    if after_n == 0:
        print("No data after threshold filtering.")
        return
    
    print("Cal EIR...")
    data_df['EIR'] = data_df.apply(calculate_simulated_eir, axis=1)
    
    print("Cal Capacity...")
    data_df['Power_W'] = data_df[POWER_COL] * 1000  # to W
    data_df['Capacity_W'] = data_df['Power_W'] / data_df['EIR']
    
    # Filter extremes by Capacity only (quantiles)
    assert 0 < lower_perc < upper_perc < 1
    q_low = data_df['Capacity_W'].quantile(lower_perc)
    q_high = data_df['Capacity_W'].quantile(upper_perc)
    before_cap = len(data_df)
    data_df = data_df[(data_df['Capacity_W'] >= q_low) & (data_df['Capacity_W'] <= q_high)].copy()
    after_cap = len(data_df)
    print(f"Applied Capacity quantile filter: lower={lower_perc:.3f} ({q_low:.1f} W), upper={upper_perc:.3f} ({q_high:.1f} W). Kept {after_cap} / {before_cap} rows.")
    
    # Create out df columns
    output_df = pd.DataFrame({
        'Date & Time': data_df['Time'],
        'Air Temperature [degC]': data_df['Air Temperature [degC]'],
        'Thermostat': data_df['Thermostat'],
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


def resample_fitting_data(input_csv: str, time_interval: str):
    print(f"Resampling fitting data: '{input_csv}' → '{time_interval}' ...")
    df = pd.read_csv(input_csv)
    if 'Date & Time' not in df.columns:
        print("error: column 'Date & Time' not found")
        return
    t = pd.to_datetime(df['Date & Time'], errors='coerce')
    df = df.loc[~t.isna()].copy()
    df['Date & Time'] = t
    numeric_cols = [c for c in df.columns if c != 'Date & Time' and pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 0:
        print("error: no numeric columns to resample")
        return

    ts = df.set_index('Date & Time')[numeric_cols]
    means = ts.resample(time_interval, label='right', closed='right').mean()
    counts = ts.resample(time_interval, label='right', closed='right').count()

    # Only keep bins that actually contain data in all key columns present
    required_keys = [c for c in ['EIR','Capacity_W','Power_kW','Air Temperature [degC]','Thermostat'] if c in counts.columns]
    if len(required_keys) == 0:
        # fall back to at least any numeric present
        required_keys = numeric_cols

    nonempty_mask = (counts[required_keys] > 0).all(axis=1)
    grouped = means.loc[nonempty_mask].reset_index()

    before_n = len(df)
    total_bins = len(means)
    kept_bins = int(nonempty_mask.sum())
    dropped_empty_bins = total_bins - kept_bins

    # Drop rows containing non-finite values in key columns to avoid downstream fitting errors
    key_cols = [c for c in ['EIR','Capacity_W','Power_kW','Air Temperature [degC]','Thermostat'] if c in grouped.columns]
    if key_cols:
        finite_mask = np.isfinite(grouped[key_cols]).all(axis=1)
        dropped_nonfinite = int((~finite_mask).sum())
        grouped = grouped.loc[finite_mask].copy()
    else:
        dropped_nonfinite = 0

    after_n = len(grouped)
    print(f"Original rows: {before_n}")
    print(f"Total {time_interval} bins in span: {total_bins}")
    print(f"Dropped empty bins (no data in required keys): {dropped_empty_bins}")
    print(f"Dropped bins with NaN/inf in key columns: {dropped_nonfinite}")
    print(f"Resampled output rows: {after_n}")

    out_csv = input_csv[:-4] + f"_{time_interval}.csv"
    grouped.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

#%% quick run
#get east home & west home weather data
east_df = filter_weather(r"C:\Users\qizixuan\OneDrive\texas\research_group\CurveFitting\data\weather\East")
west_df = filter_weather(r"C:\Users\qizixuan\OneDrive\texas\research_group\CurveFitting\data\weather\West")

#merge them into a full set weather data
merged_weather = merge_weather_pair_to_minute(
    east_df,
    west_df,
    columns1=["Air Temperature [degC]", "Relative Humidity [%]", "Relative Air Pressure [kPa]",
              "Wind Direction [deg]","Dew Point [degC]","Wind Chill Temperature [degC]",
              "Wind Speed [m/s]","Thermostat"],  
    columns2=["Total [W/m2]", "Diffuse [W/m2]"]  
)

merged_om = merge_openmeteo_to_minute(
    base_csv="../data/weather/two_merged_weather.csv",
    om_csv="../data/weather/open_meteo_data.csv",
    out_csv="../data/weather/final_merged_weather.csv",
    prefix="OM_",
    tz_local="America/Chicago"  # College Station
)

#merge weather data & egauge data
egauge_df = pd.read_csv("../data/single/egauge/egauge_1min_0601_0830_2024.csv")
merged_egauge_weather = merge_weather_with_egauge(
    merged_om,
    egauge_df,
    weather_cols=['Air Temperature [degC]', 'Relative Humidity [%]',
           'Relative Air Pressure [kPa]', 'Wind Direction [deg]',
           'Dew Point [degC]', 'Wind Chill Temperature [degC]', 'Wind Speed [m/s]',
           'Thermostat', 'Total [W/m2]', 'Diffuse [W/m2]', 'OM_DNI(W/m2)',
           'OM_precipitation', 'OM_rain', 'OM_snowfall', 'OM_snow_depth'],  
    egauge_cols=["Indoor Unit [kW]", "Outdoor Unit [kW]"]  
)

#gen .epw weather file
files = write_epw_blocks(
    src="../data/final_merged_weather_egauge.csv",
    out_dir="../data/weather"
)
print("EPW written:", *files, sep="\n- ")

# see power distribution
distribution_analysis(egauge_df)

# set threshold and create final date for fitting
threshold = 1.4 #kW
lower_perc = 0.01
upper_perc = 0.99
calculate_capacity_for_fitting(threshold, upper_perc, lower_perc)

# optional resample
resample_fitting_data("../data/single/single_data_for_fitting.csv", "30min")

#%% notes
'''
East_valid_weather_dates = ['6/5', '6/6', '6/7', '6/10', '6/11', '6/12', '6/13', '6/14', 
               '6/15', '6/16', '6/17', '6/18', '6/19', '6/22', '6/24', '6/25', 
               '6/26', '6/27', '6/28', '6/29', '6/30', '7/1', '7/2', '7/3', 
               '7/4', '7/5', '7/6', '7/7', '7/19', '7/20', '7/21', '7/22', 
               '7/23', '7/24', '7/25', '7/26', '7/27', '7/28', '7/30', '8/1', 
               '8/2', '8/3', '8/4', '8/8', '8/9', '8/10', '8/11', '8/12', '8/23', 
               '8/24', '8/25', '8/26', '8/28', '8/29', '8/31']

West_valid_weather_dates = ['6/5', '6/6', '6/7', '6/8', '6/9', '6/10', '6/11', '6/12', 
                            '6/13', '6/14', '6/15', '6/16', '6/17', '6/18', '6/19', 
                            '6/20', '6/21', '6/22', '6/23', '6/24', '6/25', '6/26', 
                            '6/27', '6/28', '6/29', '6/30', '7/1', '7/2', '7/3', '7/4', 
                            '7/5', '7/6', '7/7', '7/8', '7/9', '7/10', '7/15', '7/16', 
                            '7/17', '7/18', '7/19', '7/20', '7/21', '7/22', '7/23',
                            '7/24', '7/25', '7/26', '7/27', '7/28', '7/29', '7/30', '7/31', 
                            '8/1', '8/2', '8/3', '8/4', '8/5', '8/6', '8/7', '8/8', '8/9', 
                            '8/10', '8/11', '8/12', '8/13', '8/14', '8/15', '8/16', '8/17', 
                            '8/18', '8/19', '8/20', '8/21', '8/22', '8/23', '8/24', '8/25', 
                            '8/26', '8/28', '8/29']

'''


