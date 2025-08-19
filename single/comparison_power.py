import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from datetime import timedelta
import sys # 用于退出脚本

# --- Rated Parameters (as provided by user) ---
RATED_TOTAL_COOLING_CAPACITY = 8030.147323  # W
RATED_SENSIBLE_HEAT_RATIO = 0.79
RATED_COOLING_COP = 3.568954366            # Rated cooling COP [1]
RATED_AIR_FLOW_RATE = 0.471947443          # m3/s

RATED_COOLING_EIR = 1 / RATED_COOLING_COP  # Approximately 0.280194112

# --- User Configuration ---
# DATA_FILE_PATH = "../data/single/single_data_for_fitting.csv"
DATA_FILE_PATH = "../data/single/single_data_for_fitting_30min.csv"
SHOW_PLOTS = True  # 是否在运行时弹出图形窗口

# 列名配置 (基于合并后数据的实际列名)
# --- Capacity Curve Inputs ---
CAP_CURVE_INPUT_COL_1 = "Air Temperature [degC]"
CAP_CURVE_INPUT_COL_2 = "Thermostat"

# --- EIR Curve Inputs ---
EIR_CURVE_INPUT_COL_1 = "Air Temperature [degC]"   
EIR_CURVE_INPUT_COL_2 = "Thermostat"   

ACTUAL_POWER_COL = "Power_kW" # 实际制冷功率列名 (kW)

def calculate_simulated_capacity(row, rated_capacity=RATED_TOTAL_COOLING_CAPACITY):
    t_outdoor = row[CAP_CURVE_INPUT_COL_1]
    t_indoor = row[CAP_CURVE_INPUT_COL_2]
    a = 0.2761932711686852
    b = 0.04150664485852297
    c = -0.0011126434076651152
    d = -0.0018033172570506553
    e = 0.000543700449719611
    f = -0.0004318693022460218
    return (a + b * t_indoor + c * t_indoor**2 + d * t_outdoor + e * t_outdoor**2 + f * t_indoor * t_outdoor)*rated_capacity
    
def calculate_simulated_eir(row, rated_eir=RATED_COOLING_EIR):
    t_outdoor = row[EIR_CURVE_INPUT_COL_1]
    t_indoor = row[EIR_CURVE_INPUT_COL_2]
    # plr = row[EIR_CURVE_INPUT_COL_PLR] # 如果您的EIR曲线需要PLR
    a = 0.17276845241387664
    b = 0.07605067441555025
    c = -0.0013137117463228867
    d = -0.01984177505084244
    e = 0.00017408003229580333
    f = 0.0003143469708324316

    return (a + b * t_indoor + c * t_indoor**2 + d * t_outdoor + e * t_outdoor**2 + f * t_indoor * t_outdoor)*rated_eir

def calculate_simulated_power_kw(simulated_capacity_watts, simulated_eir):
    if simulated_capacity_watts is None or simulated_eir is None:
        return None
    simulated_power_watts = simulated_capacity_watts * simulated_eir
    simulated_power_kw = simulated_power_watts / 1000
    return simulated_power_kw

def compare_real_curve():
    # 1. 加载数据
    try:
        data_df = pd.read_csv(DATA_FILE_PATH)
    except FileNotFoundError:
        print(f"错误: 数据文件未找到 '{DATA_FILE_PATH}'")
        print("请确保您已在脚本顶部正确配置 `DATA_FILE_PATH` 并将您的数据文件放在正确的位置。")
        sys.exit(1) # 退出脚本
    except Exception as e:
        print(f"加载数据文件 '{DATA_FILE_PATH}' 时发生错误: {e}")
        sys.exit(1) # 退出脚本

    print(f"成功从 '{DATA_FILE_PATH}' 加载 {len(data_df)} 行数据。")
    print("数据前5行:")
    print(data_df.head())
    print("\n数据列名:")
    print(data_df.columns.tolist())
    print("-" * 30)

    # 收集所有在配置中定义的输入列名
    cap_input_cols = []
    if CAP_CURVE_INPUT_COL_1: cap_input_cols.append(CAP_CURVE_INPUT_COL_1)
    if CAP_CURVE_INPUT_COL_2: cap_input_cols.append(CAP_CURVE_INPUT_COL_2)
    # Add more CAP_CURVE_INPUT_COL_X if defined by user

    eir_input_cols = []
    if EIR_CURVE_INPUT_COL_1: eir_input_cols.append(EIR_CURVE_INPUT_COL_1)
    if EIR_CURVE_INPUT_COL_2: eir_input_cols.append(EIR_CURVE_INPUT_COL_2)
    # if EIR_CURVE_INPUT_COL_PLR: eir_input_cols.append(EIR_CURVE_INPUT_COL_PLR)
    # Add more EIR_CURVE_INPUT_COL_X if defined by user
    
    all_required_data_cols = list(set(cap_input_cols + eir_input_cols + [ACTUAL_POWER_COL]))
    
    missing_cols = [col for col in all_required_data_cols if col not in data_df.columns]
    if missing_cols:
        print(f"错误: 数据文件 '{DATA_FILE_PATH}' 中缺少以下必需的列: {missing_cols}")
        print("请检查您的CSV文件以及脚本顶部的列名配置部分:")
        print(f"  配置的Capacity curve输入列: {cap_input_cols}")
        print(f"  配置的EIR curve输入列: {eir_input_cols}")
        print(f"  配置的Actual power列: {ACTUAL_POWER_COL}")
        sys.exit(1) # 退出脚本

    # Filter data based on ACTUAL_POWER_COL > 0.4
    # First ensure the column is numeric, handling potential errors
    original_row_count = len(data_df)
    data_df[ACTUAL_POWER_COL] = pd.to_numeric(data_df[ACTUAL_POWER_COL], errors='coerce')
    
    # Drop rows where conversion to numeric failed for the filter column
    rows_with_conversion_errors = data_df[ACTUAL_POWER_COL].isnull().sum()
    if rows_with_conversion_errors > 0:
        print(f"警告: 在筛选前, '{ACTUAL_POWER_COL}' 列中发现 {rows_with_conversion_errors} 个非数值或空值，这些行将被移除。")
        data_df.dropna(subset=[ACTUAL_POWER_COL], inplace=True)

    num_before_filter = len(data_df)
    data_df = data_df[data_df[ACTUAL_POWER_COL] > 1.4]  # 筛选功率 > 0.4kW 的数据 (HVAC开启状态)
    num_after_filter = len(data_df)
    print(f"根据 '{ACTUAL_POWER_COL}' > 1.4 进行筛选。筛选前数据行数: {num_before_filter}, 筛选后数据行数: {num_after_filter}。")

    if num_after_filter == 0:
        print(f"错误: 筛选后没有数据满足 '{ACTUAL_POWER_COL}' > 0.4 的条件。请检查您的数据和筛选阈值。")
        sys.exit(1)

    # 2. 计算模拟值
    print("开始计算模拟制冷量...")
    data_df['SimulatedCapacity_W'] = data_df.apply(
        lambda row: calculate_simulated_capacity(row, RATED_TOTAL_COOLING_CAPACITY),
        axis=1
    )
    print("模拟制冷量计算完成。")

    print("开始计算模拟EIR...")
    data_df['SimulatedEIR'] = data_df.apply(
        lambda row: calculate_simulated_eir(row, RATED_COOLING_EIR),
        axis=1
    )
    print("模拟EIR计算完成。")

    data_df['SimulatedPower_kW'] = calculate_simulated_power_kw(
        data_df['SimulatedCapacity_W'],
        data_df['SimulatedEIR']
    )
    print("模拟功率计算完成。")

    actual_power_values = data_df[ACTUAL_POWER_COL] # Already filtered and numeric
    simulated_power_values = data_df['SimulatedPower_kW']

    valid_indices = actual_power_values.notna() & simulated_power_values.notna()
    actual_power_plot = actual_power_values[valid_indices]
    simulated_power_plot = simulated_power_values[valid_indices]

    if actual_power_plot.empty or simulated_power_plot.empty:
        print("错误: 没有有效的实际功率或模拟功率数据点可供绘图。")
        sys.exit(1)

    # 4. 可视化比较
    plt.figure(figsize=(10, 8))
    plt.scatter(actual_power_plot, simulated_power_plot, alpha=0.6, edgecolors='w', linewidth=0.5, label="points")

    min_val = min(actual_power_plot.min(), simulated_power_plot.min())
    max_val = max(actual_power_plot.max(), simulated_power_plot.max())
    if pd.notna(min_val) and pd.notna(max_val):
         plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='IDEAL')
    else:
        print("警告: 无法确定绘图范围的最小值/最大值，y=x线可能不准确或未绘制。")

    plt.title('Actual vs. Sim', fontsize=16)
    plt.xlabel(f'Actual ({ACTUAL_POWER_COL})', fontsize=14)
    plt.ylabel('Calculated (kW)', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if pd.notna(min_val) and pd.notna(max_val) and max_val > min_val :
        padding = (max_val - min_val) * 0.05
        plt.xlim(min_val - padding, max_val + padding)
        plt.ylim(min_val - padding, max_val + padding)

    plt.tight_layout()
    plt.show()
    plot_filename = "power_comparison_actual_vs_simulated.png"
    try:
        plt.savefig(plot_filename)
        print(f"\n对比图已保存为: {plot_filename}")
    except Exception as e:
        print(f"错误: 保存图像 '{plot_filename}' 失败: {e}")
    
    # plt.show()

    print("\n--- 脚本执行完毕 ---")
    print("请检查生成的图像 `power_comparison_actual_vs_simulated.png`。")
    print("如果模拟结果不符合预期，或脚本执行出错，请检查并修改脚本顶部的配置以及曲线函数。")

def compare_ep_real_curve():
    # 路径配置
    ep_xlsx_path = "../data/single/EnergyPlus/1min.xlsx"
    real_csv_path = "../data/final_merged_weather_egauge.csv"

    # 读取真实数据（Weather+Egauge合并数据）
    real_df = pd.read_csv(real_csv_path)
    assert "Time" in real_df.columns, "CSV缺少 'Time' 列"
    assert "Outdoor Unit [kW]" in real_df.columns, "CSV缺少 'Outdoor Unit [kW]' 列"
    assert CAP_CURVE_INPUT_COL_1 in real_df.columns and CAP_CURVE_INPUT_COL_2 in real_df.columns, "CSV缺少曲线输入列"

    real_df["DateTime"] = pd.to_datetime(real_df["Time"])  # 期望为分钟级
    real_df.sort_values("DateTime", inplace=True)
    real_df.set_index("DateTime", inplace=True)

    # 计算曲线功率（kW）
    real_df["SimulatedCapacity_W"] = real_df.apply(
        lambda row: calculate_simulated_capacity(row, RATED_TOTAL_COOLING_CAPACITY), axis=1
    )
    real_df["SimulatedEIR"] = real_df.apply(
        lambda row: calculate_simulated_eir(row, RATED_COOLING_EIR), axis=1
    )
    real_df["Curve_raw_kW"] = calculate_simulated_power_kw(
        real_df["SimulatedCapacity_W"], real_df["SimulatedEIR"]
    )
    real_df["Real_kW"] = pd.to_numeric(real_df["Outdoor Unit [kW]"])
    # 条件曲线：Real>1.4 使用曲线计算，否则使用 Real 值
    real_df["Curve_kW"] = np.where(real_df["Real_kW"] > 1.4, real_df["Curve_raw_kW"], real_df["Real_kW"]).astype(float)

    # 读取EnergyPlus结果
    ep_df = pd.read_excel(ep_xlsx_path, sheet_name=0)
    ep_cols_lower = [c.lower() for c in ep_df.columns]

    # 寻找时间列
    preferred_time_cols = ["Date & Time", "Date/Time", "Date Time", "DateTime"]
    time_col = None
    for c in preferred_time_cols:
        if c in ep_df.columns:
            time_col = c
            break
    if time_col is None:
        candidates = [c for c in ep_df.columns if ("date" in c.lower() or "time" in c.lower())]
        assert len(candidates) > 0, "EnergyPlus文件未找到时间列"
        time_col = candidates[0]

    # 仅使用精确列名，不做模糊匹配
    exact = "DAIKIN ONE STAGE HP COOLING COIL:Cooling Coil Electricity Rate [W](TimeStep)"
    assert exact in ep_df.columns, "EnergyPlus文件未找到制冷电能列（缺少精确列名）"
    cool_col = exact

    # 解析EnergyPlus时间列（如 "06/05  00:01:00"，无年份）
    ep_time_raw = ep_df[time_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    # 推断是否包含年份
    has_year = ep_time_raw.str.contains(r"\b\d{4}\b", regex=True).any()
    if has_year:
        ep_df["DateTime"] = pd.to_datetime(ep_time_raw)
    else:
        # 假设为 MM/DD HH:MM:SS，补全年份为 2024，并处理 24:00:00
        parsed = []
        for v in ep_time_raw:
            if not v:
                parsed.append(pd.NaT)
                continue
            parts = v.split(" ")
            assert len(parts) >= 2, f"Unexpected EP Date/Time format: '{v}'"
            date_part = parts[0]
            time_part = parts[-1]
            add_day = False
            if time_part.startswith("24:"):
                time_part = "00:" + time_part.split(":", 1)[1]
                add_day = True
            dt = pd.to_datetime(f"2024-{date_part} {time_part}", format="%Y-%m/%d %H:%M:%S")
            if add_day:
                dt = dt + pd.Timedelta(days=1)
            parsed.append(dt)
        ep_df["DateTime"] = pd.to_datetime(pd.Series(parsed))
    ep_df.sort_values("DateTime", inplace=True)
    ep_df.set_index("DateTime", inplace=True)
    ep_df["EP_kW"] = pd.to_numeric(ep_df[cool_col])/1000

    # 对齐三个时间序列
    base_df = pd.DataFrame(index=real_df.index)
    base_df["Real_kW"] = real_df["Real_kW"]
    base_df["Curve_kW"] = real_df["Curve_kW"]
    base_df = base_df.join(ep_df[["EP_kW"]], how="inner")
    base_df = base_df.sort_index()
    base_df = base_df.dropna(subset=["Real_kW", "Curve_kW", "EP_kW"])  # 仅比较三者都有的时刻

    assert len(base_df) > 0, "没有可比较的重叠时间点"

    # 评估指标函数
    def compute_pair_metrics(series_true: pd.Series, series_pred: pd.Series) -> dict:
        s_true = series_true.astype(float)
        s_pred = series_pred.astype(float)
        mask = s_true.notna() & s_pred.notna()
        s_true = s_true[mask]
        s_pred = s_pred[mask]
        n = len(s_true)
        assert n > 0, "指标计算无样本"

        diff = s_pred - s_true
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        # MAPE：只在真实值非零处计算
        nz = s_true != 0
        mape = float(np.mean(np.abs(diff[nz] / s_true[nz])) * 100) if nz.any() else np.nan
        # R2
        ss_res = float(np.sum(diff ** 2))
        ss_tot = float(np.sum((s_true - float(np.mean(s_true))) ** 2))
        r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else np.nan
        return {"n": n, "MAE": mae, "RMSE": rmse, "MAPE_%": mape, "R2": r2}

    def compute_energy_kwh(series_kw: pd.Series, interval_minutes: int) -> float:
        s = series_kw.astype(float).dropna()
        return float(np.sum(s) * (interval_minutes / 60.0))

    # 重采样粒度
    intervals = [
        ("1min", 1),
        ("5min", 5),
        ("15min", 15),
        ("30min", 30),
        ("60min", 60),
    ]

    summary_rows = []

    # 生成各粒度的时间序列对比与指标
    for rule, minutes in intervals:
        if rule == "1min":
            df_res = base_df.copy()
        else:
            df_res = base_df.resample(rule).mean()
            df_res = df_res.dropna(subset=["Real_kW", "Curve_kW", "EP_kW"])  # 仅保留三者同时存在
        if len(df_res) == 0:
            continue

        # 总量（kWh）
        energy_ep = compute_energy_kwh(df_res["EP_kW"], minutes)
        energy_real = compute_energy_kwh(df_res["Real_kW"], minutes)
        energy_curve = compute_energy_kwh(df_res["Curve_kW"], minutes)

        # 指标（相对于Real）
        met_ep_vs_real = compute_pair_metrics(df_res["Real_kW"], df_res["EP_kW"])
        met_curve_vs_real = compute_pair_metrics(df_res["Real_kW"], df_res["Curve_kW"]) 
        met_ep_vs_curve = compute_pair_metrics(df_res["Curve_kW"], df_res["EP_kW"]) 

        summary_rows.append({
            "interval": rule,
            "points": len(df_res),
            "Energy_real_kWh": energy_real,
            "Energy_ep_kWh": energy_ep,
            "Energy_curve_kWh": energy_curve,
            "MAE_ep_vs_real": met_ep_vs_real["MAE"],
            "RMSE_ep_vs_real": met_ep_vs_real["RMSE"],
            "MAPE%_ep_vs_real": met_ep_vs_real["MAPE_%"],
            "R2_ep_vs_real": met_ep_vs_real["R2"],
            "MAE_curve_vs_real": met_curve_vs_real["MAE"],
            "RMSE_curve_vs_real": met_curve_vs_real["RMSE"],
            "MAPE%_curve_vs_real": met_curve_vs_real["MAPE_%"],
            "R2_curve_vs_real": met_curve_vs_real["R2"],
            "MAE_ep_vs_curve": met_ep_vs_curve["MAE"],
            "RMSE_ep_vs_curve": met_ep_vs_curve["RMSE"],
            "MAPE%_ep_vs_curve": met_ep_vs_curve["MAPE_%"],
            "R2_ep_vs_curve": met_ep_vs_curve["R2"],
        })

        # 保存时间序列对比图（处理时间空洞：reindex到规则频率以产生NaN间断）
        freq_map = {"1min": "T", "5min": "5T", "15min": "15T", "30min": "30T", "60min": "60T"}
        use_freq = freq_map.get(rule, rule)
        full_index = pd.date_range(start=df_res.index.min(), end=df_res.index.max(), freq=use_freq)
        df_plot = df_res.reindex(full_index)

        plt.figure(figsize=(18, 4))
        # 半透明三色：EP(蓝, 实线)、Curve(橙, 虚线)、Real(绿, 实线)
        plt.plot(
            df_plot.index,
            df_plot["EP_kW"],
            label="EnergyPlus",
            color="tab:blue",
            linewidth=2.2,
            alpha=0.75,
            linestyle="-",
            zorder=2,
        )
        plt.plot(
            df_plot.index,
            df_plot["Curve_kW"],
            label="Curve",
            color="tab:orange",
            linewidth=2.2,
            alpha=0.75,
            linestyle="-",
            zorder=3,
        )
        plt.plot(
            df_plot.index,
            df_plot["Real_kW"],
            label="Real",
            color="tab:green",
            linewidth=2.2,
            alpha=0.75,
            linestyle="--",
            zorder=4,
        )

        plt.title(f"EnergyPlus vs Real vs Curve ({rule}, mean)")
        plt.ylabel("Power (kW)")
        plt.legend(ncol=3)
        plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(f"timeseries_{rule}.png")
        if SHOW_PLOTS:
            plt.show()
        plt.close()

        # Energy-based window metrics (do not stress pointwise synchrony)
        step_factor = minutes / 60.0
        energy_step = pd.DataFrame(index=df_res.index)
        energy_step["Real_kWh"] = (df_res["Real_kW"].astype(float)) * step_factor
        energy_step["EP_kWh"] = (df_res["EP_kW"].astype(float)) * step_factor
        energy_step["Curve_kWh"] = (df_res["Curve_kW"].astype(float)) * step_factor

        hourly_energy = energy_step.resample("H").sum()
        daily_energy = energy_step.resample("D").sum()

        # Save windowed energy values
        hourly_energy.to_csv(f"hourly_energy_values_{rule}.csv")
        daily_energy.to_csv(f"daily_energy_values_{rule}.csv")

        # Compute windowed energy metrics vs Real
        def window_metrics(df_win: pd.DataFrame, pred_col: str) -> dict:
            dfw = df_win[["Real_kWh", pred_col]].dropna()
            if len(dfw) == 0:
                return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "MAPE_%": np.nan, "R2": np.nan}
            return compute_pair_metrics(dfw["Real_kWh"], dfw[pred_col])

        hourly_met_ep = window_metrics(hourly_energy, "EP_kWh")
        hourly_met_curve = window_metrics(hourly_energy, "Curve_kWh")
        daily_met_ep = window_metrics(daily_energy, "EP_kWh")
        daily_met_curve = window_metrics(daily_energy, "Curve_kWh")

        pd.DataFrame([{
            "n": hourly_met_ep["n"],
            "MAE_EP_vs_Real": hourly_met_ep["MAE"],
            "RMSE_EP_vs_Real": hourly_met_ep["RMSE"],
            "MAPE%_EP_vs_Real": hourly_met_ep["MAPE_%"],
            "R2_EP_vs_Real": hourly_met_ep["R2"],
            "MAE_Curve_vs_Real": hourly_met_curve["MAE"],
            "RMSE_Curve_vs_Real": hourly_met_curve["RMSE"],
            "MAPE%_Curve_vs_Real": hourly_met_curve["MAPE_%"],
            "R2_Curve_vs_Real": hourly_met_curve["R2"],
        }]).to_csv(f"hourly_energy_metrics_{rule}.csv", index=False)

        pd.DataFrame([{
            "n": daily_met_ep["n"],
            "MAE_EP_vs_Real": daily_met_ep["MAE"],
            "RMSE_EP_vs_Real": daily_met_ep["RMSE"],
            "MAPE%_EP_vs_Real": daily_met_ep["MAPE_%"],
            "R2_EP_vs_Real": daily_met_ep["R2"],
            "MAE_Curve_vs_Real": daily_met_curve["MAE"],
            "RMSE_Curve_vs_Real": daily_met_curve["RMSE"],
            "MAPE%_Curve_vs_Real": daily_met_curve["MAPE_%"],
            "R2_Curve_vs_Real": daily_met_curve["R2"],
        }]).to_csv(f"daily_energy_metrics_{rule}.csv", index=False)

    # 汇总表
    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df) > 0:
        summary_df.to_csv("metrics_summary.csv", index=False)
        print(summary_df)
    else:
        print("无可用汇总指标（可能无重叠数据或重采样后为空）")

if __name__ == '__main__':
    compare_ep_real_curve()
    
