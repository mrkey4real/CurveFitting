#!/usr/bin/env python3
"""
eGauge 1天1分钟间隔数据获取工具 - 最终版
==================================

获取1天的1分钟间隔kW数据

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
REGISTERS  = [29, 33]  # 目标寄存器ID列表 - 都是kW
OUTPUT_CSV = "egauge_1day_final.csv"
# ============================================================

def get_1day_1min_data(dev):
    """获取1天的1分钟间隔数据"""
    print("开始获取最近1天的数据，1分钟间隔...")
    
    now = dt.datetime.now()
    all_data_points = []
    
    # 1天 = 1440分钟
    total_points = 90 * 24 * 60
    print(f"预计获取 {total_points} 个数据点")
    
    successful_points = 0
    failed_points = 0
    
    for minute_offset in range(total_points):
        target_time = now - dt.timedelta(minutes=minute_offset)
        target_epoch = int(target_time.timestamp())
        
        try:
            # 使用rate参数获取功率数据
            data = dev.get(f"/register?time={target_epoch}&rate")
            
            if 'error' not in data and 'registers' in data:
                # 从registers字典中获取rate数据
                register_values = []
                register_info = {}
                
                # 建立寄存器映射
                for reg in data['registers']:
                    if 'idx' in reg and reg['idx'] in REGISTERS:
                        register_info[reg['idx']] = reg
                
                # 提取目标寄存器的rate数据 (已经是kW单位)
                for reg_id in REGISTERS:
                    if reg_id in register_info and 'rate' in register_info[reg_id]:
                        kw_value = register_info[reg_id]['rate']  # 直接使用，已经是kW
                        register_values.append(kw_value)
                    else:
                        register_values.append(None)
                
                if any(v is not None for v in register_values):
                    all_data_points.append({
                        'timestamp': target_epoch,
                        'datetime': target_time,
                        'values': register_values
                    })
                    
                    successful_points += 1
                    
                    # 每100个成功点打印一次进度
                    if successful_points % 100 == 0:
                        progress = (minute_offset + 1) / total_points * 100
                        print(f"进度: {progress:.1f}% - 已获取 {successful_points}/{total_points} 个数据点 (失败: {failed_points})")
                        print(f"当前时间: {target_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        if len(register_values) >= 2:
                            print(f"  最新数据: S14={register_values[0]:.3f}kW, Indoor_Unit={register_values[1]:.3f}kW")
                else:
                    failed_points += 1
            else:
                failed_points += 1
                
        except Exception as e:
            failed_points += 1
            if failed_points % 100 == 0:
                print(f"API异常: {str(e)}")
        
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
    
    header = ["datetime", "epoch_timestamp", "S14_kW", "Indoor_Unit_Extra_kW"]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        # 按时间排序
        sorted_data = sorted(data_points, key=lambda x: x['timestamp'])
        
        for point in sorted_data:
            row_data = [
                point['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                point['timestamp']
            ] + [f"{v:.6f}" if v is not None else "N/A" for v in point['values']]
            writer.writerow(row_data)
    
    print(f"数据保存完成: {csv_path}")
    
    # 显示统计信息
    if sorted_data:
        start_time = sorted_data[0]['datetime']
        end_time = sorted_data[-1]['datetime']
        duration = end_time - start_time
        
        print()
        print("=== 数据统计 ===")
        print(f"数据时间范围: {start_time.strftime('%Y-%m-%d %H:%M:%S')} 到 {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"数据跨度: {duration.days} 天 {duration.seconds//3600} 小时 {(duration.seconds//60)%60} 分钟")
        expected_points = int(duration.total_seconds() / 60) + 1
        print(f"数据完整度: {len(sorted_data)}/{expected_points} = {len(sorted_data)/expected_points*100:.1f}%")
        
        # 显示数据统计
        reg29_values = [p['values'][0] for p in sorted_data if p['values'][0] is not None]
        reg33_values = [p['values'][1] for p in sorted_data if p['values'][1] is not None]
        
        if reg29_values:
            print(f"S14 (kW): 最小={min(reg29_values):.6f}, 最大={max(reg29_values):.6f}, 平均={sum(reg29_values)/len(reg29_values):.6f}")
        if reg33_values:
            print(f"Indoor Unit Extra (kW): 最小={min(reg33_values):.6f}, 最大={max(reg33_values):.6f}, 平均={sum(reg33_values)/len(reg33_values):.6f}")

def main():
    print("=== eGauge 1天1分钟间隔数据获取工具 (最终版) ===")
    
    # 初始化设备连接
    print("正在连接到eGauge设备...")
    dev = webapi.device.Device(DEVICE_URI, webapi.JWTAuth(USERNAME, PASSWORD))
    
    # 验证连接
    print("验证设备连接...")
    hostname = dev.get("/config/net/hostname")
    print(f"连接成功，设备主机名: {hostname.get('result', 'Unknown')}")
    
    print(f"目标寄存器: {REGISTERS}")
    print("  - 寄存器29: S14 (kW)")
    print("  - 寄存器33: Indoor Unit Extra (kW)")
    print()
    
    # 获取数据
    data_points = get_1day_1min_data(dev)
    
    if data_points:
        save_data_to_csv(data_points, OUTPUT_CSV)
        print()
        print("=== 获取完成 ===")
        print(f"输出文件: {OUTPUT_CSV}")
        print(f"数据点数: {len(data_points)}")
        print(f"数据间隔: 1分钟")
        print(f"数据单位: kW")
    else:
        print("未能获取到任何数据")

if __name__ == "__main__":
    main() 