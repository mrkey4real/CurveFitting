#!/usr/bin/env python3
"""
正确的数据合并脚本
==================

将15分钟温度数据插值到1分钟功率数据的时间点上
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ======================== 配置 ========================
POWER_1MIN_FILE = "Compare_1min.xlsx"      # 1分钟功率数据
TEMP_15MIN_FILE = "Curve comparison.csv"   # 15分钟温度数据
OUTPUT_CSV = "merged_1min_correct.csv"

TIME_COL = "Date & Time"
POWER_COL = "Outdoor Unit [kW]"
TEMP_OUTDOOR_COL = "Air Temperature [degC]"
TEMP_INDOOR_COL = "Thermostat_x"
# =====================================================

def load_and_clean_1min_data():
    """加载并清理1分钟功率数据"""
    print("=== 加载1分钟功率数据 ===")
    
    df = pd.read_excel(POWER_1MIN_FILE)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    
    # 按时间排序
    df = df.sort_values(TIME_COL)
    
    # 移除重复时间
    df = df.drop_duplicates(subset=[TIME_COL], keep='first')
    
    # 设置时间为索引
    df.set_index(TIME_COL, inplace=True)
    
    # 只保留功率列
    df = df[[POWER_COL]].copy()
    
    # 检查时间间隔
    time_diffs = df.index.to_series().diff().dropna()
    interval_counts = time_diffs.value_counts().head(5)
    
    print(f"1分钟数据总行数: {len(df)}")
    print(f"时间范围: {df.index.min()} 到 {df.index.max()}")
    print(f"最常见的时间间隔:")
    for interval, count in interval_counts.items():
        print(f"  {interval}: {count} 个 ({100*count/len(df):.1f}%)")
    
    return df

def load_and_clean_15min_data():
    """加载并清理15分钟温度数据"""
    print("\n=== 加载15分钟温度数据 ===")
    
    df = pd.read_csv(TEMP_15MIN_FILE)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    
    # 按时间排序
    df = df.sort_values(TIME_COL)
    
    # 移除重复时间
    df = df.drop_duplicates(subset=[TIME_COL], keep='first')
    
    # 设置时间为索引
    df.set_index(TIME_COL, inplace=True)
    
    # 只保留温度列
    df = df[[TEMP_OUTDOOR_COL, TEMP_INDOOR_COL]].copy()
    
    print(f"15分钟数据总行数: {len(df)}")
    print(f"时间范围: {df.index.min()} 到 {df.index.max()}")
    
    return df

def interpolate_temp_to_1min(df_1min_power, df_15min_temp):
    """将15分钟温度数据插值到1分钟功率数据的时间点上"""
    print("\n=== 插值温度数据到1分钟时间点 ===")
    
    # 找到时间重叠区间
    start_time = max(df_1min_power.index.min(), df_15min_temp.index.min())
    end_time = min(df_1min_power.index.max(), df_15min_temp.index.max())
    
    print(f"时间重叠区间: {start_time} 到 {end_time}")
    
    # 筛选1分钟功率数据到重叠区间
    df_power_filtered = df_1min_power[(df_1min_power.index >= start_time) & 
                                      (df_1min_power.index <= end_time)].copy()
    
    print(f"重叠区间内的1分钟功率数据点数: {len(df_power_filtered)}")
    
    # 创建包含所有1分钟时间点的温度DataFrame
    df_temp_interpolated = pd.DataFrame(index=df_power_filtered.index)
    
    # 为每个温度列进行插值
    for temp_col in [TEMP_OUTDOOR_COL, TEMP_INDOOR_COL]:
        print(f"插值 {temp_col}...")
        
        # 合并15分钟温度数据和1分钟时间索引
        temp_series = df_15min_temp[temp_col]
        
        # 创建一个包含所有时间点的Series
        all_times = df_power_filtered.index.union(temp_series.index).sort_values()
        temp_extended = temp_series.reindex(all_times)
        
        # 线性插值
        temp_interpolated = temp_extended.interpolate(method='linear')
        
        # 提取1分钟时间点的插值结果
        df_temp_interpolated[temp_col] = temp_interpolated.reindex(df_power_filtered.index)
    
    # 填充可能的边界缺失值
    df_temp_interpolated = df_temp_interpolated.fillna(method='bfill').fillna(method='ffill')
    
    print(f"插值完成，生成 {len(df_temp_interpolated)} 个温度数据点")
    
    # 合并功率和温度数据
    df_merged = pd.concat([df_power_filtered, df_temp_interpolated], axis=1)
    
    # 移除任何包含NaN的行
    df_clean = df_merged.dropna()
    
    print(f"合并后数据点数: {len(df_merged)}")
    print(f"移除NaN后数据点数: {len(df_clean)}")
    
    return df_clean

def save_and_analyze(df_merged):
    """保存并分析合并数据"""
    print(f"\n=== 保存合并数据到 {OUTPUT_CSV} ===")
    
    # 重置索引
    df_to_save = df_merged.reset_index()
    
    # 保存
    df_to_save.to_csv(OUTPUT_CSV, index=False)
    
    print(f"数据已保存!")
    print(f"最终统计:")
    print(f"  总行数: {len(df_to_save)}")
    print(f"  时间范围: {df_to_save[TIME_COL].min()} 到 {df_to_save[TIME_COL].max()}")
    print(f"  列名: {df_to_save.columns.tolist()}")
    
    # 检查时间间隔
    df_to_save[TIME_COL] = pd.to_datetime(df_to_save[TIME_COL])
    time_diffs = df_to_save[TIME_COL].diff().dropna()
    interval_counts = time_diffs.value_counts().head(5)
    
    print(f"\n时间间隔验证:")
    for interval, count in interval_counts.items():
        print(f"  {interval}: {count} 个 ({100*count/len(df_to_save):.1f}%)")
    
    # 数据质量统计
    print(f"\n数据质量统计:")
    for col in df_merged.columns:
        if df_merged[col].dtype in ['float64', 'int64']:
            print(f"  {col}:")
            print(f"    范围: {df_merged[col].min():.3f} 到 {df_merged[col].max():.3f}")
            print(f"    平均值: {df_merged[col].mean():.3f}")
    
    print(f"\n前10行数据验证:")
    print(df_to_save.head(10))
    
    return df_to_save

def main():
    print("=== 正确的数据合并工具 ===")
    print("将15分钟温度数据正确插值到1分钟功率数据上")
    print()
    
    try:
        # 1. 加载数据
        df_power_1min = load_and_clean_1min_data()
        df_temp_15min = load_and_clean_15min_data()
        
        # 2. 插值合并
        df_merged = interpolate_temp_to_1min(df_power_1min, df_temp_15min)
        
        if len(df_merged) == 0:
            print("合并后没有有效数据")
            return
        
        # 3. 保存分析
        df_final = save_and_analyze(df_merged)
        
        print("\n=== 数据合并完成 ===")
        print(f"输出文件: {OUTPUT_CSV}")
        print(f"现在你有了真正的1分钟数据用于模型对比!")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 