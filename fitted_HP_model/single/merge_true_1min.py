#!/usr/bin/env python3
"""
真正的1分钟数据合并
===================

只提取真正1分钟间隔的功率数据，与15分钟温度数据插值合并
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ======================== 配置 ========================
POWER_FILE = "Compare_1min.xlsx"
TEMP_FILE = "Curve comparison.csv"
OUTPUT_CSV = "merged_true_1min_data.csv"

TIME_COL = "Date & Time"
POWER_COL = "Outdoor Unit [kW]"
TEMP_OUTDOOR_COL = "Air Temperature [degC]"
TEMP_INDOOR_COL = "Thermostat_x"
# =====================================================

def extract_true_1min_power_data():
    """提取真正1分钟间隔的功率数据"""
    print("=== 提取真正1分钟间隔功率数据 ===")
    
    df = pd.read_excel(POWER_FILE)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    
    # 计算时间间隔
    time_diffs = df[TIME_COL].diff()
    
    # 找到1分钟间隔的数据点
    one_min_mask = time_diffs == pd.Timedelta(minutes=1)
    
    # 找到第一个1分钟间隔的位置
    first_1min_idx = one_min_mask.idxmax() - 1  # 减1是为了包含第一个1分钟数据点
    
    print(f"原始数据总行数: {len(df)}")
    print(f"第一个1分钟间隔出现在行 {first_1min_idx + 1}")
    print(f"1分钟数据开始时间: {df.iloc[first_1min_idx][TIME_COL]}")
    
    # 提取1分钟间隔数据
    df_1min = df.iloc[first_1min_idx:].copy().reset_index(drop=True)
    
    # 验证时间间隔
    time_diffs_1min = df_1min[TIME_COL].diff().dropna()
    interval_counts = time_diffs_1min.value_counts()
    
    print(f"提取的1分钟数据行数: {len(df_1min)}")
    print(f"时间范围: {df_1min[TIME_COL].min()} 到 {df_1min[TIME_COL].max()}")
    print(f"时间间隔验证:")
    for interval, count in interval_counts.head(5).items():
        print(f"  {interval}: {count} 个 ({100*count/len(df_1min):.1f}%)")
    
    # 设置时间为索引
    df_1min.set_index(TIME_COL, inplace=True)
    df_1min = df_1min[[POWER_COL]].copy()
    
    return df_1min

def load_temp_data():
    """加载15分钟温度数据"""
    print("\n=== 加载15分钟温度数据 ===")
    
    df = pd.read_csv(TEMP_FILE)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL)
    df = df.drop_duplicates(subset=[TIME_COL], keep='first')
    df.set_index(TIME_COL, inplace=True)
    df = df[[TEMP_OUTDOOR_COL, TEMP_INDOOR_COL]].copy()
    
    print(f"15分钟温度数据行数: {len(df)}")
    print(f"时间范围: {df.index.min()} 到 {df.index.max()}")
    
    return df

def interpolate_and_merge(df_power_1min, df_temp_15min):
    """插值温度数据并合并"""
    print("\n=== 插值并合并数据 ===")
    
    # 找到时间重叠区间
    start_time = max(df_power_1min.index.min(), df_temp_15min.index.min())
    end_time = min(df_power_1min.index.max(), df_temp_15min.index.max())
    
    print(f"时间重叠区间: {start_time} 到 {end_time}")
    
    # 筛选功率数据到重叠区间
    df_power_filtered = df_power_1min[(df_power_1min.index >= start_time) & 
                                      (df_power_1min.index <= end_time)].copy()
    
    print(f"重叠区间内1分钟功率数据点数: {len(df_power_filtered)}")
    
    # 为每个温度列创建插值数据
    df_temp_interpolated = pd.DataFrame(index=df_power_filtered.index)
    
    for temp_col in [TEMP_OUTDOOR_COL, TEMP_INDOOR_COL]:
        print(f"插值 {temp_col}...")
        
        # 获取温度数据
        temp_series = df_temp_15min[temp_col]
        
        # 创建扩展的时间索引（包含功率数据时间点和温度数据时间点）
        all_times = df_power_filtered.index.union(temp_series.index).sort_values()
        
        # 将温度数据重新索引到扩展时间
        temp_extended = temp_series.reindex(all_times)
        
        # 线性插值
        temp_interpolated = temp_extended.interpolate(method='linear')
        
        # 提取1分钟时间点的插值结果
        df_temp_interpolated[temp_col] = temp_interpolated.reindex(df_power_filtered.index)
    
    # 填充可能的边界缺失值
    df_temp_interpolated = df_temp_interpolated.fillna(method='bfill').fillna(method='ffill')
    
    # 合并功率和温度数据
    df_merged = pd.concat([df_power_filtered, df_temp_interpolated], axis=1)
    df_clean = df_merged.dropna()
    
    print(f"插值完成，温度数据点数: {len(df_temp_interpolated)}")
    print(f"合并后数据点数: {len(df_merged)}")
    print(f"移除NaN后数据点数: {len(df_clean)}")
    
    return df_clean

def save_and_verify(df_merged):
    """保存并验证合并数据"""
    print(f"\n=== 保存到 {OUTPUT_CSV} ===")
    
    # 重置索引
    df_to_save = df_merged.reset_index()
    
    # 保存
    df_to_save.to_csv(OUTPUT_CSV, index=False)
    
    print(f"数据已保存!")
    print(f"最终统计:")
    print(f"  总行数: {len(df_to_save)}")
    print(f"  时间范围: {df_to_save[TIME_COL].min()} 到 {df_to_save[TIME_COL].max()}")
    print(f"  列名: {df_to_save.columns.tolist()}")
    
    # 验证时间间隔
    df_to_save[TIME_COL] = pd.to_datetime(df_to_save[TIME_COL])
    time_diffs = df_to_save[TIME_COL].diff().dropna()
    interval_counts = time_diffs.value_counts().head(5)
    
    print(f"\n最终时间间隔验证:")
    for interval, count in interval_counts.items():
        print(f"  {interval}: {count} 个 ({100*count/len(df_to_save):.1f}%)")
    
    # 数据质量统计
    print(f"\n数据质量统计:")
    for col in df_merged.columns:
        if df_merged[col].dtype in ['float64', 'int64']:
            print(f"  {col}:")
            print(f"    范围: {df_merged[col].min():.3f} 到 {df_merged[col].max():.3f}")
            print(f"    平均值: {df_merged[col].mean():.3f}")
    
    print(f"\n前10行验证 (应该全是1分钟间隔):")
    for i in range(min(10, len(df_to_save))):
        print(f"  {df_to_save.iloc[i][TIME_COL]} | 功率: {df_to_save.iloc[i][POWER_COL]:.6f} kW")
    
    return df_to_save

def main():
    print("=== 真正的1分钟数据合并工具 ===")
    print("只提取1分钟间隔的功率数据，与插值温度数据合并")
    print()
    
    try:
        # 1. 提取真正1分钟功率数据
        df_power_1min = extract_true_1min_power_data()
        
        # 2. 加载温度数据
        df_temp_15min = load_temp_data()
        
        # 3. 插值合并
        df_merged = interpolate_and_merge(df_power_1min, df_temp_15min)
        
        if len(df_merged) == 0:
            print("合并后没有有效数据")
            return
        
        # 4. 保存验证
        df_final = save_and_verify(df_merged)
        
        print("\n=== 数据合并完成 ===")
        print(f"输出文件: {OUTPUT_CSV}")
        print("现在你有了真正的1分钟间隔数据，可以进行准确的模型对比了!")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 