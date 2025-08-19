import pandas as pd
import matplotlib.pyplot as plt
import sys # 用于退出脚本

# --- Rated Parameters (as provided by user) ---
RATED_TOTAL_COOLING_CAPACITY = 8030.147323  # W
RATED_SENSIBLE_HEAT_RATIO = 0.79
RATED_COOLING_COP = 3.568954366            # Rated cooling COP [1]
RATED_AIR_FLOW_RATE = 0.471947443          # m3/s

# User provided "Rated cooling EIR [1]: 0.352463 0.280194112"
# We are using the EIR derived from COP (1 / 3.568954366 = 0.280194112)
# If the value 0.352463 is intended for a different purpose or a different rated EIR,
# please adjust RATED_COOLING_EIR accordingly.
RATED_COOLING_EIR = 1 / RATED_COOLING_COP  # Approximately 0.280194112
# ALTERNATIVE_RATED_EIR = 0.352463 # Mentioned by user, kept here for reference

# --- User Configuration ---
# 使用合并后的1分钟数据文件
# DATA_FILE_PATH = "./final_merged_1min_data.csv"
DATA_FILE_PATH = "./merged_true_1min_data.csv"


# 列名配置 (基于合并后数据的实际列名)
# --- Capacity Curve Inputs ---
CAP_CURVE_INPUT_COL_1 = "Air Temperature [degC]"
CAP_CURVE_INPUT_COL_2 = "Thermostat_x"
# CAP_CURVE_INPUT_COL_AIRFLOW = "AirFlowRateRatio"
# 如果只有两个输入：
# CAP_CURVE_INPUT_COL_1 = "TEMP_OUT_AIR_DRY_BULB"  # 示例: 室外干球温度 (°C) - 请替换为您的列名
# CAP_CURVE_INPUT_COL_2 = "TEMP_IN_AIR_DRY_BULB"   # 示例: 室内干球温度 (°C) - 请替换为您的列名
# 如果需要更多输入, 请按需添加 CAP_CURVE_INPUT_COL_3, CAP_CURVE_INPUT_COL_4 等

# --- EIR Curve Inputs ---
# 示例: 如果您的EIR曲线输入是室外温度、室内温度和部分负荷比等。
# 请根据您的模型需要的输入参数数量和名称进行调整。
EIR_CURVE_INPUT_COL_1 = "Air Temperature [degC]"   # 示例: 室外干球温度 (°C) - 请替换为您的列名
EIR_CURVE_INPUT_COL_2 = "Thermostat_x"   # 示例: 室内干球温度 (°C) - 请替换为您的列名
# EIR_CURVE_INPUT_COL_PLR = "PartLoadRatio"       # 示例: 部分负荷比 - 请替换为您的列名 (如果EIR曲线需要PLR)
# 如果需要更多输入, 请按需添加 EIR_CURVE_INPUT_COL_X, EIR_CURVE_INPUT_COL_Y 等

ACTUAL_POWER_COL = "Outdoor Unit [kW]" # 实际制冷功率列名 (kW)
# 注意: 如果您的实际功率单位不是kW, 请在后续计算中进行转换, 或调整绘图标签

# --- Placeholder Curve Functions ---
# TODO: 用户需要根据已训练的曲线模型替换以下函数实现

def calculate_simulated_capacity(row, rated_capacity=RATED_TOTAL_COOLING_CAPACITY):
    """
    根据输入参数计算模拟的制冷量 (W)。
    用户需要用实际的Capacity Curve替换此函数的逻辑。
    Inputs:
        row: pandas DataFrame的一行, 包含曲线输入列的数据
        rated_capacity: 额定制冷量 (W)
    Returns:
        simulated_capacity_watts (float): 模拟的制冷量 (W)
    """
    # --- !!! 用户替换区域开始 !!! ---
    # 请根据您的Capacity curve (例如, 多项式回归系数) 修改此处
    # 访问输入参数示例:
    t_outdoor = row[CAP_CURVE_INPUT_COL_1]
    t_indoor = row[CAP_CURVE_INPUT_COL_2]
    # air_flow_ratio = row[CAP_CURVE_INPUT_COL_AIRFLOW] # 如果您定义并使用了这个列名
    
    
    a = 1.5392492248392489
    b = -0.089457139510662
    c = 0.0024016296417590263
    d = 0.010228079391473497
    e = -0.00040877588550816923
    f = 0.00046277054246848955
    # 假设您的容量曲线是一个修正系数乘以额定容量:
    return (a + b * t_indoor + c * t_indoor**2 + d * t_outdoor + e * t_outdoor**2 + f * t_indoor * t_outdoor)*rated_capacity
    
    # simulated_capacity_watts = rated_capacity * capacity_modifier
    # --- !!! 用户替换区域结束 !!! ---
    raise NotImplementedError("用户需实现: calculate_simulated_capacity。请参考函数内注释修改。")

def calculate_simulated_eir(row, rated_eir=RATED_COOLING_EIR):
    """
    根据输入参数计算模拟的EIR (Energy Input Ratio, 无量纲)。
    用户需要用实际的EIR Curve替换此函数的逻辑。
    Inputs:
        row: pandas DataFrame的一行, 包含曲线输入列的数据
        rated_eir: 额定EIR
    Returns:
        simulated_eir (float): 模拟的EIR
    """
    # --- !!! 用户替换区域开始 !!! ---
    # 请根据您的EIR curve (例如, 多项式回归系数) 修改此处
    # 访问输入参数示例:
    t_outdoor = row[EIR_CURVE_INPUT_COL_1]
    t_indoor = row[EIR_CURVE_INPUT_COL_2]
    # plr = row[EIR_CURVE_INPUT_COL_PLR] # 如果您的EIR曲线需要PLR
    a = 0.26754729965852525
    b = 0.04247903558916574
    c = -0.0011410021860023213
    d = -0.0019796200320364448
    e = 0.0005419221743031003
    f = -0.00041962720373709664
    
    # simulated_eir = rated_eir * eir_modifier
    # --- !!! 用户替换区域结束 !!! ---
    return (a + b * t_indoor + c * t_indoor**2 + d * t_outdoor + e * t_outdoor**2 + f * t_indoor * t_outdoor)*rated_eir

def calculate_simulated_power_kw(simulated_capacity_watts, simulated_eir):
    """
    根据模拟的制冷量和模拟的EIR计算模拟的功率 (kW)。
    Power (W) = Capacity (W) * EIR (dimensionless)
    Power (kW) = Power (W) / 1000
    """
    if simulated_capacity_watts is None or simulated_eir is None:
        return None
    simulated_power_watts = simulated_capacity_watts * simulated_eir
    simulated_power_kw = simulated_power_watts / 1000
    return simulated_power_kw

def main():
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
    print(f"根据 '{ACTUAL_POWER_COL}' > 0.4 进行筛选。筛选前数据行数: {num_before_filter}, 筛选后数据行数: {num_after_filter}。")

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

    # if EIR_CURVE_INPUT_COL_PLR and EIR_CURVE_INPUT_COL_PLR not in data_df.columns:
    #     print(f"信息: EIR曲线需要 '{EIR_CURVE_INPUT_COL_PLR}'。正在基于 SimulatedCapacity_W 和 RATED_TOTAL_COOLING_CAPACITY 计算...")
    #     data_df[EIR_CURVE_INPUT_COL_PLR] = data_df['SimulatedCapacity_W'] / RATED_TOTAL_COOLING_CAPACITY
    #     print(f"'{EIR_CURVE_INPUT_COL_PLR}' 列已计算并添加到DataFrame。")

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

    # 3. 准备用于绘图的数据
    # ACTUAL_POWER_COL has already been converted to numeric and filtered
    # data_df[ACTUAL_POWER_COL] = pd.to_numeric(data_df[ACTUAL_POWER_COL], errors='coerce') # This line is no longer needed here
    
    # if data_df[ACTUAL_POWER_COL].isnull().any(): # This check might still be relevant if other operations introduce NaNs
    #     num_invalid = data_df[ACTUAL_POWER_COL].isnull().sum()
    #     print(f"警告: '{ACTUAL_POWER_COL}' 列中发现 {num_invalid} 个非数值或空值，这些行将在绘图中被忽略。")

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

if __name__ == '__main__':
    main()
