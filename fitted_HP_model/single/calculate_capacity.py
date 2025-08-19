# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 2025

@author: qizixuan
"""

import pandas as pd
from HP_cooling_real import calculate_simulated_eir, RATED_COOLING_EIR

# 定义输入列名
EIR_CURVE_INPUT_COL_1 = "Air Temperature [degC]"   # 室外温度
EIR_CURVE_INPUT_COL_2 = "Thermostat_x"             # 室内温度
POWER_COL = "Outdoor Unit [kW]"                    # 功率列名

def main():
    # 读取数据
    print("正在读取final_merged_1min_data.csv...")
    data_df = pd.read_csv('final_merged_1min_data.csv')
    print(f"数据加载完成，共 {len(data_df)} 行数据")
    
    # 检查必要的列是否存在
    required_cols = [EIR_CURVE_INPUT_COL_1, EIR_CURVE_INPUT_COL_2, POWER_COL]
    missing_cols = [col for col in required_cols if col not in data_df.columns]
    if missing_cols:
        print(f"错误：缺少必要的列: {missing_cols}")
        return
    
    print("正在计算EIR...")
    # 计算EIR
    data_df['EIR'] = data_df.apply(calculate_simulated_eir, axis=1)
    
    print("正在计算容量...")
    # 计算容量 (Capacity = Power / EIR)
    # 功率是kW，转换为W后计算容量
    data_df['Power_W'] = data_df[POWER_COL] * 1000  # 转换为瓦特
    data_df['Capacity_W'] = data_df['Power_W'] / data_df['EIR']
    
    # 创建输出DataFrame，包含需要的列
    output_df = pd.DataFrame({
        'Date & Time': data_df['Date & Time'],
        'EIR': data_df['EIR'],
        'Capacity_W': data_df['Capacity_W'],
        'Power_kW': data_df[POWER_COL]
    })
    
    # 保存到新文件
    output_filename = 'final_merged_1min_calculated_capacity_data.csv'
    print(f"正在保存结果到 {output_filename}...")
    output_df.to_csv(output_filename, index=False)
    
    print(f"处理完成！结果已保存到 {output_filename}")
    print(f"共处理 {len(output_df)} 行数据")
    
    # 显示前几行结果
    print("\n前5行结果:")
    print(output_df.head())
    
    # 显示统计信息
    print("\n统计信息:")
    print(f"EIR范围: {output_df['EIR'].min():.4f} - {output_df['EIR'].max():.4f}")
    print(f"容量范围 (W): {output_df['Capacity_W'].min():.1f} - {output_df['Capacity_W'].max():.1f}")
    print(f"功率范围 (kW): {output_df['Power_kW'].min():.3f} - {output_df['Power_kW'].max():.3f}")

if __name__ == "__main__":
    main() 