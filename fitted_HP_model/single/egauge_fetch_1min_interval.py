#!/usr/bin/env python3
"""
eGauge 1分钟间隔历史数据获取工具
==================================

获取尽可能多的1分钟间隔历史数据

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
OUTPUT_CSV = "egauge_1min_interval.csv"
INTERVAL_MINUTES = 1  # 1分钟间隔
# ============================================================

def get_historical_data_1min(dev, max_days=90):
    """获取历史数据，1分钟间隔"""
    print(f"开始获取最近{max_days}天的数据，1分钟间隔...")
    
    now = dt.datetime.now()
    all_data_points = []
    
    # 计算总数据点数
    total_points = max_days * 24 * 60  # 每天1440个点
    print(f"预计获取 {total_points} 个数据点")
    
    successful_points = 0
    failed_points = 0
    consecutive_failures = 0
    
    for point_num in range(total_points):
        # 计算目标时间
        minutes_back = point_num * INTERVAL_MINUTES
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
                    consecutive_failures = 0  # 重置连续失败计数
                    
                    # 每1000个成功点打印一次进度
                    if successful_points % 1000 == 0:
                        progress = (point_num + 1) / total_points * 100
                        print(f"进度: {progress:.1f}% - 已获取 {successful_points}/{total_points} 个数据点 (失败: {failed_points})")
                        print(f"当前时间: {target_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        # 保存中间结果
                        if successful_points % 10000 == 0:
                            save_checkpoint(all_data_points, f"checkpoint_{successful_points}.csv")
                            
                else:
                    failed_points += 1
                    consecutive_failures += 1
            else:
                failed_points += 1
                consecutive_failures += 1
                
        except Exception as e:
            failed_points += 1
            consecutive_failures += 1
            if failed_points % 1000 == 0:
                print(f"API异常: {str(e)}")
        
        # 如果连续失败过多，可能已经超出数据范围，停止获取
        if consecutive_failures > 1440:  # 连续失败超过1天的数据点
            print(f"连续失败超过1天({consecutive_failures}个点)，可能已达到数据边界，停止获取")
            break
        
        # 每100个请求休息一下，避免过载
        if (point_num + 1) % 100 == 0:
            time.sleep(0.05)
        
        # 每1000个请求休息久一点
        if (point_num + 1) % 1000 == 0:
            time.sleep(0.5)
    
    print(f"数据获取完成！")
    print(f"成功: {successful_points} 个数据点")
    print(f"失败: {failed_points} 个数据点")
    if successful_points + failed_points > 0:
        print(f"成功率: {successful_points/(successful_points+failed_points)*100:.1f}%")
    
    return all_data_points

def save_checkpoint(data_points, csv_path):
    """保存检查点文件"""
    if not data_points:
        return
        
    print(f"保存检查点: {csv_path} ({len(data_points)} 个数据点)")
    
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
        print(f"数据完整度: {len(sorted_data)}/{duration.total_seconds()/60:.0f} = {len(sorted_data)/(duration.total_seconds()/60)*100:.1f}%")

def get_optimized_batch_data(dev, max_days=90, batch_size=60):
    """优化的批量获取，每批获取连续的时间点"""
    print(f"使用优化批量模式获取最近{max_days}天的数据...")
    print(f"每批获取 {batch_size} 分钟的数据")
    
    now = dt.datetime.now()
    all_data_points = []
    
    total_batches = (max_days * 24 * 60) // batch_size
    
    for batch_num in range(total_batches):
        batch_start_minutes = batch_num * batch_size
        batch_end_minutes = (batch_num + 1) * batch_size
        
        batch_start_time = now - dt.timedelta(minutes=batch_end_minutes)
        batch_end_time = now - dt.timedelta(minutes=batch_start_minutes)
        
        print(f"批次 {batch_num + 1}/{total_batches}: {batch_start_time.strftime('%Y-%m-%d %H:%M')} 到 {batch_end_time.strftime('%Y-%m-%d %H:%M')}")
        
        batch_points = 0
        batch_failures = 0
        
        for minute_offset in range(batch_size):
            target_time = batch_start_time + dt.timedelta(minutes=minute_offset)
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
                    else:
                        batch_failures += 1
                else:
                    batch_failures += 1
                    
            except Exception:
                batch_failures += 1
            
            # 每10个请求休息一下
            if minute_offset % 10 == 0:
                time.sleep(0.02)
        
        print(f"  批次 {batch_num + 1} 完成: 成功 {batch_points} 个，失败 {batch_failures} 个")
        
        # 如果这个批次失败太多，可能已经超出数据范围
        if batch_failures > batch_size * 0.8:  # 超过80%失败
            print(f"批次失败率过高({batch_failures}/{batch_size})，可能已达到数据边界")
            break
        
        # 每批之间休息
        time.sleep(0.2)
        
        # 每10批保存一次检查点
        if (batch_num + 1) % 10 == 0:
            save_checkpoint(all_data_points, f"checkpoint_batch_{batch_num + 1}.csv")
    
    print(f"总共获取到 {len(all_data_points)} 个数据点")
    return all_data_points

def main():
    print("=== eGauge 1分钟间隔历史数据获取工具 ===")
    
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
    
    # 选择获取模式
    print("选择数据获取模式:")
    print("1. 优化批量模式 (推荐，较快)")
    print("2. 逐点获取模式 (较慢但更精确)")
    
    # 默认使用优化批量模式
    mode = 1
    print(f"使用模式 {mode}: 优化批量模式")
    
    if mode == 1:
        data_points = get_optimized_batch_data(dev, max_days=90, batch_size=60)
    else:
        data_points = get_historical_data_1min(dev, max_days=90)
    
    if data_points:
        save_data_to_csv(data_points, OUTPUT_CSV)
        print()
        print("=== 获取完成 ===")
        print(f"输出文件: {OUTPUT_CSV}")
        print(f"数据点数: {len(data_points)}")
        print(f"目标寄存器: {REGISTERS}")
        print(f"数据间隔: {INTERVAL_MINUTES} 分钟")
    else:
        print("未能获取到任何数据")

if __name__ == "__main__":
    main() 