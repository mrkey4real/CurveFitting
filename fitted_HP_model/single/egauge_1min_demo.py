#!/usr/bin/env python3
"""
eGauge 1分钟间隔演示工具 - 获取3天数据
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
OUTPUT_CSV = "egauge_3days_1min.csv"
# ============================================================

def get_1min_data_demo(dev, days=3):
    """获取最近几天的1分钟间隔数据"""
    print(f"开始获取最近{days}天的数据，1分钟间隔...")
    
    now = dt.datetime.now()
    all_data_points = []
    
    # 计算总数据点数（每天1440分钟）
    total_points = days * 24 * 60
    print(f"预计获取 {total_points} 个数据点")
    
    successful_points = 0
    failed_points = 0
    
    for minute_offset in range(total_points):
        target_time = now - dt.timedelta(minutes=minute_offset)
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
                    
                    # 每500个成功点打印一次进度
                    if successful_points % 500 == 0:
                        progress = (minute_offset + 1) / total_points * 100
                        print(f"进度: {progress:.1f}% - 已获取 {successful_points}/{total_points} 个数据点 (失败: {failed_points})")
                        print(f"当前时间: {target_time.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    failed_points += 1
            else:
                failed_points += 1
                
        except Exception as e:
            failed_points += 1
            if failed_points % 100 == 0:
                print(f"API异常(第{failed_points}次): {str(e)}")
        
        # 每50个请求休息一下
        if (minute_offset + 1) % 50 == 0:
            time.sleep(0.02)
    
    print(f"数据获取完成！")
    print(f"成功: {successful_points} 个数据点")
    print(f"失败: {failed_points} 个数据点")
    if successful_points + failed_points > 0:
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
        print(f"数据跨度: {duration.days} 天 {duration.seconds//3600} 小时 {(duration.seconds//60)%60} 分钟")
        expected_points = int(duration.total_seconds() / 60) + 1
        print(f"数据完整度: {len(sorted_data)}/{expected_points} = {len(sorted_data)/expected_points*100:.1f}%")

def main():
    print("=== eGauge 3天1分钟间隔演示工具 ===")
    
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
    data_points = get_1min_data_demo(dev, days=3)
    
    if data_points:
        save_data_to_csv(data_points, OUTPUT_CSV)
        print()
        print("=== 获取完成 ===")
        print(f"输出文件: {OUTPUT_CSV}")
        print(f"数据点数: {len(data_points)}")
        print(f"目标寄存器: {REGISTERS}")
        print(f"数据间隔: 1分钟")
    else:
        print("未能获取到任何数据")

if __name__ == "__main__":
    main() 