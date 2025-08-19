#!/usr/bin/env python3
"""
简化的eGauge数据获取脚本
==============================

直接使用requests获取1分钟间隔的功率数据，不依赖eGauge库

Dependencies:
    pip install requests
"""

import requests
import json
import time
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ======================== 配置 ========================
DEVICE_URI = "https://egauge66683.d.egauge.net"
USERNAME = "mingjun"
PASSWORD = "mingjun"

# 目标寄存器
TARGET_REGISTERS = {
    'S14': 29,               # S14功率
    'Indoor_Unit_Extra': 33  # 室内机额外功率
}

# 输出文件
OUTPUT_CSV = "egauge_1min_data.csv"

# 数据获取参数
DURATION_HOURS = 24      # 获取24小时数据
INTERVAL_MINUTES = 1     # 1分钟间隔
# ===================================================

def get_auth_token():
    """获取JWT认证令牌"""
    auth_url = f"{DEVICE_URI}/cgi-bin/egauge-show"
    try:
        response = requests.get(auth_url, auth=(USERNAME, PASSWORD), timeout=10)
        if response.status_code == 200:
            print("认证成功")
            return True
        else:
            print(f"认证失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"认证请求失败: {e}")
        return False

def get_single_datapoint(timestamp):
    """获取指定时间戳的单个数据点"""
    try:
        # 构建API URL - 使用rate参数获取瞬时功率
        api_url = f"{DEVICE_URI}/cgi-bin/egauge-show"
        params = {
            'time': int(timestamp),
            'rate': '',  # 获取瞬时功率而不是累积能量
            'format': 'json'
        }
        
        response = requests.get(
            api_url, 
            params=params,
            auth=(USERNAME, PASSWORD),
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # 解析数据
            if 'registers' in data and 'ranges' in data:
                registers = data['registers']
                ranges = data['ranges']
                
                if ranges and len(ranges) > 0:
                    range_data = ranges[0]
                    if 'rows' in range_data and len(range_data['rows']) > 0:
                        row = range_data['rows'][0]
                        
                        result = {
                            'timestamp': timestamp,
                            'datetime': datetime.fromtimestamp(timestamp),
                            'success': True
                        }
                        
                        # 提取目标寄存器数据
                        for reg_name, reg_index in TARGET_REGISTERS.items():
                            if reg_index < len(row) and row[reg_index] is not None:
                                # rate值已经是kW，不需要除以1000
                                result[f'{reg_name}_kW'] = row[reg_index]
                            else:
                                result[f'{reg_name}_kW'] = None
                                print(f"警告: 寄存器{reg_index}数据缺失")
                        
                        return result
        
        print(f"API请求失败: {response.status_code}")
        return {
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp),
            'success': False
        }
        
    except Exception as e:
        print(f"获取时间戳 {timestamp} 的数据时出错: {e}")
        return {
            'timestamp': timestamp,
            'datetime': datetime.fromtimestamp(timestamp),
            'success': False
        }

def collect_1min_data():
    """收集1分钟间隔的数据"""
    print(f"开始收集 {DURATION_HOURS} 小时的1分钟间隔数据...")
    
    # 计算时间范围
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=DURATION_HOURS)
    
    # 转换为epoch时间戳
    start_timestamp = int(start_time.timestamp())
    end_timestamp = int(end_time.timestamp())
    
    print(f"时间范围: {start_time} 到 {end_time}")
    print(f"时间戳范围: {start_timestamp} 到 {end_timestamp}")
    
    # 生成1分钟间隔的时间戳列表
    timestamps = []
    current_timestamp = start_timestamp
    while current_timestamp <= end_timestamp:
        timestamps.append(current_timestamp)
        current_timestamp += 60  # 增加60秒（1分钟）
    
    total_points = len(timestamps)
    print(f"预计获取 {total_points} 个数据点")
    
    # 收集数据
    all_data = []
    success_count = 0
    
    for i, timestamp in enumerate(timestamps):
        if i % 100 == 0:  # 每100个点打印进度
            print(f"进度: {i}/{total_points} ({100*i/total_points:.1f}%)")
        
        result = get_single_datapoint(timestamp)
        all_data.append(result)
        
        if result.get('success', False):
            success_count += 1
        
        # 添加小延迟避免过快请求
        time.sleep(0.1)
    
    success_rate = 100 * success_count / total_points
    print(f"\n数据收集完成!")
    print(f"成功获取: {success_count}/{total_points} ({success_rate:.1f}%)")
    
    return all_data

def save_data(data_list):
    """保存数据到CSV文件"""
    print(f"\n保存数据到 {OUTPUT_CSV}...")
    
    # 转换为DataFrame
    df = pd.DataFrame(data_list)
    
    # 过滤成功的数据点
    df_success = df[df['success'] == True].copy()
    
    # 移除不需要的列
    columns_to_keep = ['datetime', 'timestamp'] + [f'{reg}_kW' for reg in TARGET_REGISTERS.keys()]
    df_final = df_success[columns_to_keep].copy()
    
    # 按时间排序
    df_final = df_final.sort_values('datetime').reset_index(drop=True)
    
    # 保存到CSV
    df_final.to_csv(OUTPUT_CSV, index=False)
    
    print(f"数据已保存到 {OUTPUT_CSV}")
    print(f"最终数据统计:")
    print(f"  数据点数: {len(df_final)}")
    print(f"  时间范围: {df_final['datetime'].min()} 到 {df_final['datetime'].max()}")
    print(f"  列名: {df_final.columns.tolist()}")
    
    # 显示数据质量
    print(f"\n数据质量:")
    for col in df_final.columns:
        if '_kW' in col:
            non_null_count = df_final[col].notna().sum()
            total_count = len(df_final)
            print(f"  {col}: {non_null_count}/{total_count} ({100*non_null_count/total_count:.1f}%) 有效数据")
            
            if non_null_count > 0:
                values = df_final[col].dropna()
                print(f"    范围: {values.min():.3f} 到 {values.max():.3f} kW")
                print(f"    平均值: {values.mean():.3f} kW")
    
    print(f"\n前5行数据:")
    print(df_final.head())
    
    return df_final

def main():
    print("=== 简化eGauge数据获取工具 ===")
    print(f"设备URI: {DEVICE_URI}")
    print(f"获取时长: {DURATION_HOURS} 小时")
    print(f"数据间隔: {INTERVAL_MINUTES} 分钟")
    print()
    
    try:
        # 1. 测试认证
        if not get_auth_token():
            print("认证失败，请检查用户名和密码")
            return
        
        # 2. 收集数据
        data_list = collect_1min_data()
        
        if not data_list:
            print("没有收集到任何数据")
            return
        
        # 3. 保存数据
        df_final = save_data(data_list)
        
        print("\n=== 数据获取完成 ===")
        print(f"输出文件: {OUTPUT_CSV}")
        print(f"数据点数: {len(df_final)}")
        
    except Exception as e:
        print(f"程序执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 