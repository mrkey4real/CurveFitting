# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 22:26:34 2025

@author: qizixuan
"""

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

# EIR曲线输入列定义
EIR_CURVE_INPUT_COL_1 = "Air Temperature [degC]"   # 室外温度
EIR_CURVE_INPUT_COL_2 = "Thermostat_x"             # 室内温度

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
    a = 0.26754729948459177
    b = 0.04247903556154983
    c = -0.0011410021852605492
    d = -0.001979620030749466
    e = 0.0005419221739507913
    f = -0.00041962720346429
    
    # simulated_eir = rated_eir * eir_modifier
    # --- !!! 用户替换区域结束 !!! ---
    return (a + b * t_indoor + c * t_indoor**2 + d * t_outdoor + e * t_outdoor**2 + f * t_indoor * t_outdoor)*rated_eir
