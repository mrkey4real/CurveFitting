#!/usr/bin/env python3
"""
eGauge 大批量历史数据获取工具
==================================

基于发现的工作API调用方式，获取尽可能多的历史数据

Dependencies:
    pip install egauge-webapi
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
OUTPUT_CSV = "egauge_full_history.csv"
INTERVAL_MINUTES = 1  # 每1分钟获取一个数据点
# ============================================================

def convert_epoch_to_datetime(epoch_timestamp):
    """将epoch时间戳转换为可读时间格式"""
    if isinstance(epoch_timestamp, str):
        epoch_timestamp = float(epoch_timestamp)
    dt_object = dt.datetime.fromtimestamp(epoch_timestamp)
    return dt_object.strftime('%Y-%m-%d %H:%M:%S')

def get_full_historical_data(dev, max_days=365):
    """获取最大时间范围的历史数据"""
    print(f"开始获取最近 {max_days} 天的1分钟级别数据...")
    
    now = dt.datetime.now()
    all_data_points = []
    
    # 计算总的数据点数量
    total_minutes = max_days * 24 * 60
    print(f"预计总数据点: {total_minutes}")
    
    successful_points = 0
    failed_points = 0
    
    for minute_offset in range(total_minutes):
        # 计算目标时间
        target_time = now - dt.timedelta(minutes=minute_offset)
        target_epoch = int(target_time.timestamp())
        
        try:
            # 使用发现的工作API调用方式
            data = dev.get(f"/register?time={target_epoch}")
            
            if 'error' not in data and 'ranges' in data and len(data['ranges']) > 0:
                rows = data['ranges'][0].get('rows', [])
                if len(rows) > 0:
                    # 获取寄存器索引映射
                    register_map = {}
                    if 'registers' in data:
                        for reg in data['registers']:
                            register_map[reg['idx']] = reg
                    
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
                    
                    # 每1000个成功点打印一次进度
                    if successful_points % 1000 == 0:
                        print(f"已获取 {successful_points} 个数据点 (失败: {failed_points})")
                        print(f"当前时间: {target_time.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    failed_points += 1
            else:
                failed_points += 1
                # 连续失败太多次就停止
                if failed_points > successful_points and failed_points > 100:
                    print(f"连续失败次数过多，停止获取")
                    break
                    
        except Exception as e:
            failed_points += 1
            if failed_points % 1000 == 0:
                print(f"API异常: {str(e)}")
            
            # 连续失败太多次就停止
            if failed_points > successful_points and failed_points > 100:
                print(f"连续异常次数过多，停止获取")
                break
        
        # 每100个请求休息一下，避免过载
        if (successful_points + failed_points) % 100 == 0:
            time.sleep(0.1)
    
    print(f"数据获取完成！")
    print(f"成功: {successful_points} 个数据点")
    print(f"失败: {failed_points} 个数据点")
    print(f"成功率: {successful_points/(successful_points+failed_points)*100:.1f}%")
    
    return all_data_points

def get_batch_historical_data(dev, max_days=365, batch_hours=6):
    """分批获取历史数据，每批获取几小时的数据"""
    print(f"开始分批获取最近 {max_days} 天的数据...")
    print(f"每批获取 {batch_hours} 小时的数据")
    
    now = dt.datetime.now()
    all_data_points = []
    
    total_batches = (max_days * 24) // batch_hours
    
    for batch_num in range(total_batches):
        batch_start_time = now - dt.timedelta(hours=batch_num * batch_hours)
        batch_end_time = now - dt.timedelta(hours=(batch_num + 1) * batch_hours)
        
        print(f"批次 {batch_num + 1}/{total_batches}: {batch_end_time.strftime('%Y-%m-%d %H:%M')} 到 {batch_start_time.strftime('%Y-%m-%d %H:%M')}")
        
        # 在这个时间段内每分钟获取一个数据点
        minutes_in_batch = batch_hours * 60
        batch_points = 0
        
        for minute_offset in range(minutes_in_batch):
            target_time = batch_start_time - dt.timedelta(minutes=minute_offset)
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
                        
                        batch_points += 1
            except Exception:
                continue  # 忽略单个请求错误
            
            # 每50个请求休息一下
            if minute_offset % 50 == 0:
                time.sleep(0.05)
        
        print(f"  批次 {batch_num + 1} 完成，获取到 {batch_points} 个数据点")
        
        # 每批之间休息
        time.sleep(1)
    
    print(f"总共获取到 {len(all_data_points)} 个数据点")
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
    print("=== eGauge 大批量历史数据获取工具 ===")
    
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
    print(f"数据间隔: {INTERVAL_MINUTES} 分钟")
    print()
    
    # 选择获取方式
    print("选择数据获取方式:")
    print("1. 快速模式 - 分批获取 (推荐)")
    print("2. 完整模式 - 逐点获取 (较慢)")
    
    # 这里默认使用快速模式
    print("使用快速模式...")
    
    # 获取数据
    data_points = get_batch_historical_data(dev, max_days=365, batch_hours=6)
    
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