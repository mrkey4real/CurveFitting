#!/usr/bin/env python3
"""
数据合并工具
==================================

将15分钟间隔的温度数据重新采样到1分钟间隔，
与1分钟的eGauge功率数据合并，以时间交集为准

Dependencies:
    pip install pandas
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ========================== CONFIG ==========================
# 数据文件路径
EGAUGE_1MIN_FILE = "egauge_1min_data.csv"   # 1分钟eGauge数据
CURVE_15MIN_FILE = "Curve comparison.csv"   # 15分钟温度数据

# 输出文件
OUTPUT_CSV = "merged_1min_data.csv"

# 15分钟数据中的关键列名（请根据实际情况修改）
TEMP_OUTDOOR_COL = "Air Temperature [degC]"      # 室外温度列名
TEMP_INDOOR_COL = "Thermostat_x"                # 室内温度列名  
TIME_COL_15MIN = "Date & Time"                   # 15分钟数据的时间列名

# 1分钟数据中的列名
TIME_COL_1MIN = "datetime"                       # 1分钟数据的时间列名
POWER_COLS_1MIN = ["S14_kW", "Indoor_Unit_Extra_kW"]  # 功率数据列名
# ============================================================

def load_and_prepare_1min_data():
    """加载并准备1分钟eGauge数据"""
    print("正在加载1分钟eGauge数据...")
    
    df_1min = pd.read_csv(EGAUGE_1MIN_FILE)
    print(f"1分钟数据原始行数: {len(df_1min)}")
    print(f"1分钟数据列名: {df_1min.columns.tolist()}")
    
    # 转换时间列为datetime
    df_1min[TIME_COL_1MIN] = pd.to_datetime(df_1min[TIME_COL_1MIN])
    
    # 设置时间为索引
    df_1min.set_index(TIME_COL_1MIN, inplace=True)
    
    # 按时间排序
    df_1min.sort_index(inplace=True)
    
    print(f"1分钟数据时间范围: {df_1min.index.min()} 到 {df_1min.index.max()}")
    print(f"1分钟数据前5行:")
    print(df_1min.head())
    
    return df_1min

def load_and_prepare_15min_data():
    """加载并准备15分钟温度数据"""
    print("\n正在加载15分钟温度数据...")
    
    df_15min = pd.read_csv(CURVE_15MIN_FILE)
    print(f"15分钟数据原始行数: {len(df_15min)}")
    print(f"15分钟数据列名: {df_15min.columns.tolist()}")
    
    # 检查必要的列是否存在
    required_cols = [TIME_COL_15MIN, TEMP_OUTDOOR_COL, TEMP_INDOOR_COL]
    missing_cols = [col for col in required_cols if col not in df_15min.columns]
    if missing_cols:
        print(f"错误: 15分钟数据中缺少以下列: {missing_cols}")
        print("请检查列名配置")
        return None
    
    # 转换时间列为datetime
    try:
        df_15min[TIME_COL_15MIN] = pd.to_datetime(df_15min[TIME_COL_15MIN])
    except Exception as e:
        print(f"时间列转换失败: {e}")
        print(f"时间列前5个值: {df_15min[TIME_COL_15MIN].head()}")
        return None
    
    # 只保留需要的列
    df_15min = df_15min[[TIME_COL_15MIN, TEMP_OUTDOOR_COL, TEMP_INDOOR_COL]].copy()
    
    # 设置时间为索引
    df_15min.set_index(TIME_COL_15MIN, inplace=True)
    
    # 按时间排序
    df_15min.sort_index(inplace=True)
    
    # 移除重复的时间索引
    df_15min = df_15min[~df_15min.index.duplicated(keep='first')]
    
    print(f"15分钟数据时间范围: {df_15min.index.min()} 到 {df_15min.index.max()}")
    print(f"15分钟数据前5行:")
    print(df_15min.head())
    
    return df_15min

def find_time_intersection(df_1min, df_15min):
    """找到两个数据集的时间交集"""
    print("\n计算时间交集...")
    
    # 获取时间范围
    start_1min = df_1min.index.min()
    end_1min = df_1min.index.max()
    start_15min = df_15min.index.min()
    end_15min = df_15min.index.max()
    
    print(f"1分钟数据时间范围: {start_1min} 到 {end_1min}")
    print(f"15分钟数据时间范围: {start_15min} 到 {end_15min}")
    
    # 计算交集
    intersection_start = max(start_1min, start_15min)
    intersection_end = min(end_1min, end_15min)
    
    print(f"时间交集范围: {intersection_start} 到 {intersection_end}")
    
    if intersection_start >= intersection_end:
        print("警告: 没有时间交集!")
        return None, None
    
    return intersection_start, intersection_end

def resample_15min_to_1min(df_15min, start_time, end_time):
    """将15分钟数据重新采样到1分钟间隔"""
    print("\n重新采样15分钟数据到1分钟间隔...")
    
    # 筛选时间范围
    df_15min_filtered = df_15min[(df_15min.index >= start_time) & (df_15min.index <= end_time)]
    print(f"筛选后15分钟数据行数: {len(df_15min_filtered)}")
    
    # 创建1分钟时间索引
    time_range_1min = pd.date_range(start=start_time, end=end_time, freq='1min')
    print(f"创建1分钟时间索引，共 {len(time_range_1min)} 个时间点")
    
    # 重新采样 - 使用线性插值
    df_resampled = df_15min_filtered.reindex(time_range_1min)
    
    # 线性插值填充缺失值
    df_resampled = df_resampled.interpolate(method='linear')
    
    # 对于开头和结尾的缺失值，使用最近的有效值填充
    df_resampled = df_resampled.fillna(method='bfill').fillna(method='ffill')
    
    print(f"重新采样后数据行数: {len(df_resampled)}")
    print(f"重新采样后数据前5行:")
    print(df_resampled.head())
    
    return df_resampled

def merge_data(df_1min, df_resampled_temps, start_time, end_time):
    """合并1分钟功率数据和重新采样的温度数据"""
    print("\n合并数据...")
    
    # 筛选1分钟数据到交集时间范围
    df_1min_filtered = df_1min[(df_1min.index >= start_time) & (df_1min.index <= end_time)]
    print(f"筛选后1分钟功率数据行数: {len(df_1min_filtered)}")
    
    # 合并数据
    df_merged = pd.concat([df_1min_filtered, df_resampled_temps], axis=1, join='inner')
    
    # 移除任何包含NaN的行
    df_merged_clean = df_merged.dropna()
    
    print(f"合并后数据行数: {len(df_merged)}")
    print(f"移除NaN后数据行数: {len(df_merged_clean)}")
    print(f"合并后数据列名: {df_merged_clean.columns.tolist()}")
    print(f"合并后数据前5行:")
    print(df_merged_clean.head())
    
    return df_merged_clean

def save_merged_data(df_merged):
    """保存合并后的数据"""
    print(f"\n保存合并数据到 {OUTPUT_CSV}...")
    
    # 重置索引，将时间作为列
    df_to_save = df_merged.reset_index()
    
    # 保存到CSV
    df_to_save.to_csv(OUTPUT_CSV, index=False)
    
    print(f"数据已保存到 {OUTPUT_CSV}")
    print(f"最终数据统计:")
    print(f"  总行数: {len(df_to_save)}")
    print(f"  时间范围: {df_to_save[TIME_COL_1MIN].min()} 到 {df_to_save[TIME_COL_1MIN].max()}")
    print(f"  列名: {df_to_save.columns.tolist()}")
    
    return df_to_save

def plot_data_overview(df_merged):
    """绘制数据概览图"""
    print("\n绘制数据概览图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('合并数据概览', fontsize=16)
    
    # 温度数据
    axes[0, 0].plot(df_merged.index, df_merged[TEMP_OUTDOOR_COL], label='室外温度', alpha=0.7)
    axes[0, 0].plot(df_merged.index, df_merged[TEMP_INDOOR_COL], label='室内温度', alpha=0.7)
    axes[0, 0].set_title('温度数据 (重新采样到1分钟)')
    axes[0, 0].set_ylabel('温度 (°C)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 功率数据
    for i, power_col in enumerate(POWER_COLS_1MIN):
        if power_col in df_merged.columns:
            axes[0, 1].plot(df_merged.index, df_merged[power_col], label=power_col, alpha=0.7)
    axes[0, 1].set_title('功率数据 (1分钟)')
    axes[0, 1].set_ylabel('功率 (kW)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 室外温度 vs 功率散点图
    if POWER_COLS_1MIN[0] in df_merged.columns:
        axes[1, 0].scatter(df_merged[TEMP_OUTDOOR_COL], df_merged[POWER_COLS_1MIN[0]], alpha=0.5)
        axes[1, 0].set_xlabel('室外温度 (°C)')
        axes[1, 0].set_ylabel(f'{POWER_COLS_1MIN[0]} (kW)')
        axes[1, 0].set_title('室外温度 vs 功率')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 室内温度 vs 功率散点图
    if POWER_COLS_1MIN[0] in df_merged.columns:
        axes[1, 1].scatter(df_merged[TEMP_INDOOR_COL], df_merged[POWER_COLS_1MIN[0]], alpha=0.5)
        axes[1, 1].set_xlabel('室内温度 (°C)')
        axes[1, 1].set_ylabel(f'{POWER_COLS_1MIN[0]} (kW)')
        axes[1, 1].set_title('室内温度 vs 功率')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_filename = "merged_data_overview.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"概览图已保存为: {plot_filename}")
    
    # plt.show()

def main():
    print("=== 数据合并工具 ===")
    print("将15分钟温度数据重新采样到1分钟，与1分钟功率数据合并")
    print()
    
    try:
        # 1. 加载数据
        df_1min = load_and_prepare_1min_data()
        df_15min = load_and_prepare_15min_data()
        
        if df_15min is None:
            return
        
        # 2. 找到时间交集
        start_time, end_time = find_time_intersection(df_1min, df_15min)
        if start_time is None:
            return
        
        # 3. 重新采样15分钟数据到1分钟
        df_resampled_temps = resample_15min_to_1min(df_15min, start_time, end_time)
        
        # 4. 合并数据
        df_merged = merge_data(df_1min, df_resampled_temps, start_time, end_time)
        
        if len(df_merged) == 0:
            print("错误: 合并后没有数据")
            return
        
        # 5. 保存数据
        df_final = save_merged_data(df_merged)
        
        # 6. 绘制概览图
        plot_data_overview(df_merged)
        
        print("\n=== 数据合并完成 ===")
        print(f"最终数据文件: {OUTPUT_CSV}")
        print(f"数据点数: {len(df_final)}")
        print(f"数据完整性: 100% (已移除所有NaN)")
        
        # 显示数据质量统计
        print(f"\n数据质量统计:")
        for col in df_merged.columns:
            if df_merged[col].dtype in ['float64', 'int64']:
                print(f"  {col}: 最小值={df_merged[col].min():.3f}, 最大值={df_merged[col].max():.3f}, 平均值={df_merged[col].mean():.3f}")
        
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 