#!/usr/bin/env python3
"""
eGauge 快速测试 - 获取24小时数据
"""

from egauge import webapi
import datetime as dt
import csv

# ========================== CONFIG ==========================
DEVICE_URI = "https://egauge66683.d.egauge.net"
USERNAME   = "mingjun"
PASSWORD   = "mingjun"
REGISTERS  = [29, 33]  # 目标寄存器ID列表
OUTPUT_CSV = "egauge_24hours_quick.csv"
# ============================================================

def get_24hours_data(dev):
    """获取最近24小时的数据，每小时一个数据点"""
    print("开始获取最近24小时的数据，每小时一个数据点...")
    
    now = dt.datetime.now()
    all_data_points = []
    
    for hour_offset in range(24):
        target_time = now - dt.timedelta(hours=hour_offset)
        target_epoch = int(target_time.timestamp())
        
        print(f"获取 {target_time.strftime('%Y-%m-%d %H:%M:%S')} 的数据...")
        
        data = dev.get(f"/register?time={target_epoch}")
        
        if 'error' not in data and 'ranges' in data and len(data['ranges']) > 0:
            rows = data['ranges'][0].get('rows', [])
            if len(rows) > 0:
                # 提取目标寄存器的数据
                register_values = []
                for reg_id in REGISTERS:
                    if reg_id < len(rows[0]):
                        register_values.append(rows[0][reg_id])
                    else:
                        register_values.append(None)
                
                all_data_points.append({
                    'timestamp': target_epoch,
                    'datetime': target_time,
                    'values': register_values
                })
                
                print(f"  ✅ 成功获取数据: {register_values}")
            else:
                print(f"  ❌ 无数据")
        else:
            error_msg = data.get('error', '未知错误')
            print(f"  ❌ 错误: {error_msg}")
    
    return all_data_points

def save_data_to_csv(data_points, csv_path):
    """保存数据到CSV文件"""
    if not data_points:
        print("没有数据可保存")
        return
    
    print(f"正在保存 {len(data_points)} 个数据点到 {csv_path}...")
    
    header = ["datetime", "epoch_timestamp"] + [f"register_{r}" for r in REGISTERS]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        # 按时间排序
        sorted_data = sorted(data_points, key=lambda x: x['timestamp'])
        
        for point in sorted_data:
            row_data = [
                point['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                point['timestamp']
            ] + point['values']
            writer.writerow(row_data)
    
    print(f"数据保存完成: {csv_path}")

def main():
    print("=== eGauge 24小时快速测试工具 ===")
    
    # 初始化设备连接
    print("正在连接到eGauge设备...")
    dev = webapi.device.Device(DEVICE_URI, webapi.JWTAuth(USERNAME, PASSWORD))
    
    # 验证连接
    print("验证设备连接...")
    hostname = dev.get("/config/net/hostname")
    print(f"连接成功，设备主机名: {hostname.get('result', 'Unknown')}")
    
    print(f"目标寄存器: {REGISTERS}")
    print()
    
    # 获取数据
    data_points = get_24hours_data(dev)
    
    if data_points:
        save_data_to_csv(data_points, OUTPUT_CSV)
        print()
        print("=== 获取完成 ===")
        print(f"输出文件: {OUTPUT_CSV}")
        print(f"数据点数: {len(data_points)}")
        print(f"目标寄存器: {REGISTERS}")
    else:
        print("未能获取到任何数据")

if __name__ == "__main__":
    main() 