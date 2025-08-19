#!/usr/bin/env python3
"""
eGauge 历史数据获取测试工具 - 获取7天数据
"""

from egauge import webapi
import datetime as dt
import csv
import time

# ========================== CONFIG ==========================
DEVICE_URI = "https://egauge66683.d.egauge.net"
USERNAME   = "mingjun"
PASSWORD   = "mingjun"
REGISTERS  = [29, 33]  # 目标寄存器ID列表
OUTPUT_CSV = "egauge_7days_test.csv"
# ============================================================

def get_historical_data_7days(dev):
    """获取最近7天的数据，每10分钟一个数据点"""
    print("开始获取最近7天的数据，每10分钟一个数据点...")
    
    now = dt.datetime.now()
    all_data_points = []
    
    # 7天，每10分钟一个点
    interval_minutes = 10
    total_points = (7 * 24 * 60) // interval_minutes  # 7天的总分钟数除以间隔
    
    print(f"预计获取 {total_points} 个数据点")
    
    successful_points = 0
    failed_points = 0
    
    for point_num in range(total_points):
        # 计算目标时间
        minutes_back = point_num * interval_minutes
        target_time = now - dt.timedelta(minutes=minutes_back)
        target_epoch = int(target_time.timestamp())
        
        try:
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
                    
                    successful_points += 1
                    
                    # 每100个成功点打印一次进度
                    if successful_points % 100 == 0:
                        print(f"已获取 {successful_points}/{total_points} 个数据点 (失败: {failed_points})")
                        print(f"当前时间: {target_time.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    failed_points += 1
            else:
                failed_points += 1
                
        except Exception as e:
            failed_points += 1
            if failed_points % 100 == 0:
                print(f"API异常: {str(e)}")
        
        # 每10个请求休息一下
        if (successful_points + failed_points) % 10 == 0:
            time.sleep(0.1)
    
    print(f"数据获取完成！")
    print(f"成功: {successful_points} 个数据点")
    print(f"失败: {failed_points} 个数据点")
    print(f"成功率: {successful_points/(successful_points+failed_points)*100:.1f}%")
    
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
    
    # 显示统计信息
    if sorted_data:
        start_time = sorted_data[0]['datetime']
        end_time = sorted_data[-1]['datetime']
        duration = end_time - start_time
        
        print(f"数据时间范围: {start_time.strftime('%Y-%m-%d %H:%M:%S')} 到 {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"数据跨度: {duration.days} 天 {duration.seconds//3600} 小时")

def main():
    print("=== eGauge 7天数据获取测试工具 ===")
    
    # 初始化设备连接
    print("正在连接到eGauge设备...")
    dev = webapi.device.Device(DEVICE_URI, webapi.JWTAuth(USERNAME, PASSWORD))
    
    # 验证连接
    print("验证设备连接...")
    hostname = dev.get("/config/net/hostname")
    if 'error' in hostname:
        print(f"连接失败: {hostname['error']}")
        return
    else:
        print(f"连接成功，设备主机名: {hostname.get('result', 'Unknown')}")
    
    print(f"目标寄存器: {REGISTERS}")
    print()
    
    # 获取数据
    data_points = get_historical_data_7days(dev)
    
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