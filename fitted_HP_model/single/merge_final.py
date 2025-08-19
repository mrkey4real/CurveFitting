#!/usr/bin/env python3
"""
数据合并脚本
==============

将15分钟温度数据重新采样到1分钟间隔，与1分钟功率数据合并
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ======================== 配置 ========================
# 数据文件
POWER_1MIN_FILE = "Compare_1min.xlsx"      # 1分钟功率数据
TEMP_15MIN_FILE = "Curve comparison.csv"   # 15分钟温度数据

# 输出文件
OUTPUT_CSV = "merged_1min_final.csv"

# 列名
TIME_COL = "Date & Time"
POWER_COL = "Outdoor Unit [kW]"
TEMP_OUTDOOR_COL = "Air Temperature [degC]"
TEMP_INDOOR_COL = "Thermostat_x"
# =====================================================

def load_1min_power_data():
    """加载1分钟功率数据"""
    print("正在加载1分钟功率数据...")
    
    df = pd.read_excel(POWER_1MIN_FILE)
    print(f"原始1分钟数据行数: {len(df)}")
    
    # 转换时间
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    
    # 设置时间为索引
    df.set_index(TIME_COL, inplace=True)
    df.sort_index(inplace=True)
    
    # 只保留功率列
    df = df[[POWER_COL]].copy()
    
    print(f"1分钟功率数据时间范围: {df.index.min()} 到 {df.index.max()}")
    print(f"1分钟数据点数: {len(df)}")
    
    return df

def load_15min_temp_data():
    """加载15分钟温度数据"""
    print("正在加载15分钟温度数据...")
    
    df = pd.read_csv(TEMP_15MIN_FILE)
    print(f"原始15分钟数据行数: {len(df)}")
    
    # 转换时间
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    
    # 设置时间为索引
    df.set_index(TIME_COL, inplace=True)
    df.sort_index(inplace=True)
    
    # 只保留温度列
    df = df[[TEMP_OUTDOOR_COL, TEMP_INDOOR_COL]].copy()
    
    # 移除重复时间
    df = df[~df.index.duplicated(keep='first')]
    
    print(f"15分钟温度数据时间范围: {df.index.min()} 到 {df.index.max()}")
    print(f"15分钟数据点数: {len(df)}")
    
    return df

def find_time_overlap(df_1min, df_15min):
    """找到两个数据集的时间重叠区间"""
    print("计算时间重叠区间...")
    
    start_1min = df_1min.index.min()
    end_1min = df_1min.index.max()
    start_15min = df_15min.index.min()
    end_15min = df_15min.index.max()
    
    print(f"1分钟数据时间范围: {start_1min} 到 {end_1min}")
    print(f"15分钟数据时间范围: {start_15min} 到 {end_15min}")
    
    # 计算重叠区间
    overlap_start = max(start_1min, start_15min)
    overlap_end = min(end_1min, end_15min)
    
    print(f"重叠时间区间: {overlap_start} 到 {overlap_end}")
    
    if overlap_start >= overlap_end:
        print("警告: 没有时间重叠!")
        return None, None
    
    return overlap_start, overlap_end

def resample_temp_to_1min(df_15min, start_time, end_time):
    """将15分钟温度数据重新采样到1分钟"""
    print("重新采样15分钟温度数据到1分钟间隔...")
    
    # 筛选时间范围
    df_filtered = df_15min[(df_15min.index >= start_time) & (df_15min.index <= end_time)]
    print(f"筛选后的15分钟数据点数: {len(df_filtered)}")
    
    # 创建1分钟时间索引
    time_index_1min = pd.date_range(start=start_time, end=end_time, freq='1min')
    print(f"1分钟时间索引点数: {len(time_index_1min)}")
    
    # 重新索引到1分钟
    df_resampled = df_filtered.reindex(time_index_1min)
    
    # 线性插值
    df_resampled = df_resampled.interpolate(method='linear')
    
    # 填充边界值
    df_resampled = df_resampled.fillna(method='bfill').fillna(method='ffill')
    
    print(f"重新采样后数据点数: {len(df_resampled)}")
    print("前5行重新采样数据:")
    print(df_resampled.head())
    
    return df_resampled

def merge_datasets(df_power_1min, df_temp_1min, start_time, end_time):
    """合并1分钟功率数据和重新采样的温度数据"""
    print("合并数据集...")
    
    # 筛选功率数据到重叠时间范围
    df_power_filtered = df_power_1min[(df_power_1min.index >= start_time) & 
                                      (df_power_1min.index <= end_time)]
    print(f"筛选后的1分钟功率数据点数: {len(df_power_filtered)}")
    
    # 合并数据
    df_merged = pd.concat([df_power_filtered, df_temp_1min], axis=1, join='inner')
    
    # 移除包含NaN的行
    df_clean = df_merged.dropna()
    
    print(f"合并后数据点数: {len(df_merged)}")
    print(f"移除NaN后数据点数: {len(df_clean)}")
    print(f"最终列名: {df_clean.columns.tolist()}")
    
    return df_clean

def save_merged_data(df_merged):
    """保存合并数据"""
    print(f"保存合并数据到 {OUTPUT_CSV}...")
    
    # 重置索引，时间作为列
    df_to_save = df_merged.reset_index()
    
    # 保存
    df_to_save.to_csv(OUTPUT_CSV, index=False)
    
    print(f"数据已保存到 {OUTPUT_CSV}")
    print(f"最终数据统计:")
    print(f"  总行数: {len(df_to_save)}")
    print(f"  时间范围: {df_to_save[TIME_COL].min()} 到 {df_to_save[TIME_COL].max()}")
    print(f"  列名: {df_to_save.columns.tolist()}")
    
    # 数据质量统计
    print(f"\n数据质量统计:")
    for col in df_merged.columns:
        if df_merged[col].dtype in ['float64', 'int64']:
            print(f"  {col}:")
            print(f"    范围: {df_merged[col].min():.3f} 到 {df_merged[col].max():.3f}")
            print(f"    平均值: {df_merged[col].mean():.3f}")
            print(f"    标准差: {df_merged[col].std():.3f}")
    
    return df_to_save

def plot_overview(df_merged):
    """绘制数据概览"""
    print("绘制数据概览图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('合并数据概览', fontsize=16)
    
    # 功率时间序列
    axes[0, 0].plot(df_merged.index, df_merged[POWER_COL], alpha=0.7)
    axes[0, 0].set_title('功率数据 (1分钟)')
    axes[0, 0].set_ylabel('功率 (kW)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 温度时间序列
    axes[0, 1].plot(df_merged.index, df_merged[TEMP_OUTDOOR_COL], label='室外温度', alpha=0.7)
    axes[0, 1].plot(df_merged.index, df_merged[TEMP_INDOOR_COL], label='室内温度', alpha=0.7)
    axes[0, 1].set_title('温度数据 (重新采样到1分钟)')
    axes[0, 1].set_ylabel('温度 (°C)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 室外温度 vs 功率
    axes[1, 0].scatter(df_merged[TEMP_OUTDOOR_COL], df_merged[POWER_COL], alpha=0.5)
    axes[1, 0].set_xlabel('室外温度 (°C)')
    axes[1, 0].set_ylabel('功率 (kW)')
    axes[1, 0].set_title('室外温度 vs 功率')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 室内温度 vs 功率
    axes[1, 1].scatter(df_merged[TEMP_INDOOR_COL], df_merged[POWER_COL], alpha=0.5)
    axes[1, 1].set_xlabel('室内温度 (°C)')
    axes[1, 1].set_ylabel('功率 (kW)')
    axes[1, 1].set_title('室内温度 vs 功率')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = "merged_data_overview.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"概览图已保存为: {plot_file}")

def main():
    print("=== 数据合并工具 ===")
    print("将15分钟温度数据重新采样到1分钟，与1分钟功率数据合并")
    print()
    
    try:
        # 1. 加载数据
        df_power_1min = load_1min_power_data()
        df_temp_15min = load_15min_temp_data()
        
        # 2. 找时间重叠
        start_time, end_time = find_time_overlap(df_power_1min, df_temp_15min)
        if start_time is None:
            print("没有时间重叠，无法合并数据")
            return
        
        # 3. 重新采样温度数据
        df_temp_1min = resample_temp_to_1min(df_temp_15min, start_time, end_time)
        
        # 4. 合并数据
        df_merged = merge_datasets(df_power_1min, df_temp_1min, start_time, end_time)
        
        if len(df_merged) == 0:
            print("合并后没有有效数据")
            return
        
        # 5. 保存数据
        df_final = save_merged_data(df_merged)
        
        # 6. 绘制概览
        plot_overview(df_merged)
        
        print("\n=== 数据合并完成 ===")
        print(f"输出文件: {OUTPUT_CSV}")
        print(f"最终数据点数: {len(df_final)}")
        print("现在可以使用这个合并后的数据进行模型对比了!")
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 