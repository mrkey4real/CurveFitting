#!/usr/bin/env python3
"""
eGauge 历史数据获取工具 - 使用 webapi 库
==================================

获取历史数据并保存为CSV文件

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
REGISTERS  = [29, 33]  # 寄存器ID列表
OUTPUT_CSV = "egauge_historical_data.csv"
# ============================================================

def convert_epoch_to_datetime(epoch_timestamp):
    """将epoch时间戳转换为可读时间格式"""
    if isinstance(epoch_timestamp, str):
        epoch_timestamp = float(epoch_timestamp)
    dt_object = dt.datetime.fromtimestamp(epoch_timestamp)
    return dt_object.strftime('%Y-%m-%d %H:%M:%S')

def get_historical_data_range(dev, days_back=30):
    """获取指定天数的历史数据"""
    print(f"正在获取最近 {days_back} 天的数据...")
    
    # 计算时间范围
    now = dt.datetime.now()
    begin = now - dt.timedelta(days=days_back)
    
    # 将时间转换为epoch秒
    begin_epoch = int(begin.timestamp())
    now_epoch = int(now.timestamp())
    
    print(f"时间范围: {begin.strftime('%Y-%m-%d %H:%M:%S')} 到 {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 尝试不同的查询参数组合
    query_attempts = [
        f"/register?time={begin_epoch}:{now_epoch}&rate&interval=60",
        f"/register?time={begin_epoch}:{now_epoch}&interval=60", 
        f"/register?begin={begin_epoch}&end={now_epoch}&interval=60",
        f"/register?from={begin_epoch}&to={now_epoch}&interval=60",
        f"/register?start={begin_epoch}&stop={now_epoch}&interval=60"
    ]
    
    for i, query in enumerate(query_attempts):
        print(f"尝试查询方式 {i+1}: {query}")
        data = dev.get(query)
        
        if 'error' in data:
            print(f"  错误: {data['error']}")
            continue
        elif 'ranges' in data and len(data['ranges']) > 0:
            rows = data['ranges'][0]['rows']
            print(f"  成功获取到 {len(rows)} 行数据")
            return data
        else:
            print(f"  没有数据，返回结构: {list(data.keys())}")
    
    print("所有查询方式都失败了")
    return None

def get_recent_data_incrementally(dev, max_days=365):
    """递增式获取最近的数据，找到可用的最大时间范围"""
    print("正在测试数据可用性...")
    
    all_data = []
    available_days = 0
    
    # 从短时间开始测试
    test_periods = [1, 3, 7, 14, 30, 60, 90, 180, 365]
    
    for days in test_periods:
        if days > max_days:
            break
            
        print(f"测试 {days} 天的数据...")
        data = get_historical_data_range(dev, days)
        
        if data and 'ranges' in data and len(data['ranges']) > 0:
            rows = data['ranges'][0]['rows']
            if len(rows) > 0:
                available_days = days
                all_data = data
                print(f"✅ {days} 天数据可用，共 {len(rows)} 行")
            else:
                print(f"❌ {days} 天无数据")
                break
        else:
            print(f"❌ {days} 天数据不可用")
            break
    
    print(f"最大可用数据范围: {available_days} 天")
    return all_data, available_days

def test_api_endpoints(dev):
    """测试不同的API端点"""
    print("=== 测试API端点 ===")
    
    test_queries = [
        "/register?time=now",
        "/register?time=now&rate",
        "/register",
        "/config/registers"
    ]
    
    for query in test_queries:
        print(f"测试: {query}")
        try:
            data = dev.get(query)
            if 'error' in data:
                print(f"  错误: {data['error']}")
            else:
                print(f"  成功: {list(data.keys())}")
                if 'ranges' in data:
                    print(f"    ranges数量: {len(data['ranges'])}")
                # 显示一些数据样例
                if query == "/register?time=now&rate":
                    if 'registers' in data:
                        if isinstance(data['registers'], dict):
                            print(f"    可用寄存器: {list(data['registers'].keys())[:10]}...")  # 只显示前10个
                        else:
                            print(f"    寄存器数据类型: {type(data['registers'])}")
                            print(f"    寄存器数据样例: {data['registers'][:3] if len(data['registers']) > 0 else '无'}")
                    if 'ranges' in data and len(data['ranges']) > 0:
                        ranges = data['ranges'][0]
                        print(f"    ranges结构: {list(ranges.keys())}")
                        if 'rows' in ranges:
                            print(f"    数据行数: {len(ranges['rows'])}")
                            if len(ranges['rows']) > 0:
                                print(f"    第一行数据样例: {ranges['rows'][0][:5]}...")  # 只显示前5个值
        except Exception as e:
            print(f"  API调用异常: {str(e)}")

def get_historical_data_different_times(dev):
    """尝试获取不同时间点的数据"""
    print("=== 尝试获取历史数据 ===")
    
    now = dt.datetime.now()
    time_points = []
    
    # 生成不同的时间点（最近几小时、几天）
    for hours_back in [1, 6, 12, 24, 48, 72]:  # 1小时到3天前
        time_point = now - dt.timedelta(hours=hours_back)
        epoch_time = int(time_point.timestamp())
        time_points.append((f"{hours_back}小时前", epoch_time, time_point))
    
    for days_back in [7, 14, 30, 60, 90]:  # 更远的时间点
        time_point = now - dt.timedelta(days=days_back)
        epoch_time = int(time_point.timestamp())
        time_points.append((f"{days_back}天前", epoch_time, time_point))
    
    available_data = []
    
    for description, epoch_time, datetime_obj in time_points:
        print(f"测试 {description} ({datetime_obj.strftime('%Y-%m-%d %H:%M')})")
        
        # 尝试不同的查询方式
        queries = [
            f"/register?time={epoch_time}",
            f"/register?time={epoch_time}&rate"
        ]
        
        for query in queries:
            try:
                data = dev.get(query)
                
                if 'error' not in data and 'ranges' in data and len(data['ranges']) > 0:
                    rows = data['ranges'][0].get('rows', [])
                    if len(rows) > 0:
                        print(f"  ✅ {query}: 获取到 {len(rows)} 行数据")
                        available_data.append({
                            'description': description,
                            'datetime': datetime_obj,
                            'epoch': epoch_time,
                            'data': data,
                            'query': query
                        })
                        break  # 找到数据就停止尝试其他查询
                    else:
                        print(f"  ❌ {query}: 无数据")
                else:
                    error_msg = data.get('error', '未知错误')
                    print(f"  ❌ {query}: {error_msg}")
            except Exception as e:
                print(f"  ❌ {query}: API异常 - {str(e)}")
    
    return available_data

def get_time_series_data(dev, max_points=1000):
    """尝试获取时间序列数据"""
    print("=== 尝试获取时间序列数据 ===")
    
    # 先获取当前数据以了解结构
    current_data = dev.get("/register?time=now&rate")
    
    if 'ranges' not in current_data or len(current_data['ranges']) == 0:
        print("无法获取当前数据")
        return None
    
    # 获取时间戳和间隔信息
    ts_now = current_data.get('ts', 0)
    print(f"当前时间戳: {ts_now} ({convert_epoch_to_datetime(ts_now)})")
    
    # 尝试获取历史时间序列
    all_data_points = []
    
    # 从现在开始，向前推若干分钟
    interval_minutes = 1  # 1分钟间隔
    
    for i in range(min(max_points, 1440)):  # 最多1440个点（1天）
        minutes_back = i * interval_minutes
        target_time = dt.datetime.now() - dt.timedelta(minutes=minutes_back)
        target_epoch = int(target_time.timestamp())
        
        data = dev.get(f"/register?time={target_epoch}&rate")
        
        if 'error' not in data and 'ranges' in data and len(data['ranges']) > 0:
            rows = data['ranges'][0].get('rows', [])
            if len(rows) > 0:
                all_data_points.append({
                    'timestamp': target_epoch,
                    'datetime': target_time,
                    'data': rows[0]  # 取第一行数据
                })
                if i < 10:  # 只打印前10个成功的点
                    print(f"  ✅ {target_time.strftime('%H:%M:%S')}: {rows[0]}")
            else:
                if i < 10:
                    print(f"  ❌ {target_time.strftime('%H:%M:%S')}: 无数据")
                break  # 没有数据就停止
        else:
            if i < 10:
                error_msg = data.get('error', '未知错误')
                print(f"  ❌ {target_time.strftime('%H:%M:%S')}: {error_msg}")
            break  # 出错就停止
    
    print(f"成功获取 {len(all_data_points)} 个数据点")
    return all_data_points

def save_data_to_csv(data, csv_path, days_back):
    """保存数据到CSV文件"""
    if not data or 'ranges' not in data or len(data['ranges']) == 0:
        print("没有数据可保存")
        return
    
    ts0 = data.get('ts', 0)
    interval = data.get('interval', 60)
    rows = data['ranges'][0]['rows']
    
    print(f"正在保存 {len(rows)} 行数据到 {csv_path}...")
    
    header = ["timestamp"] + [f"register_{r}" for r in REGISTERS]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for i, values in enumerate(rows):
            timestamp = ts0 + i * interval
            readable_time = convert_epoch_to_datetime(timestamp)
            writer.writerow([readable_time] + values)
    
    print(f"数据保存完成: {csv_path}")

def main():
    print("=== eGauge 历史数据获取工具 ===")
    
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
    
    # 测试API端点
    test_api_endpoints(dev)
    print()
    
    # 尝试获取不同时间的数据
    historical_data = get_historical_data_different_times(dev)
    
    if historical_data:
        print(f"\n找到 {len(historical_data)} 个可用的历史数据点")
        # 保存找到的历史数据
        save_historical_data_points(historical_data, "egauge_historical_points.csv")
    
    # 尝试获取时间序列数据
    time_series_data = get_time_series_data(dev, max_points=500)
    
    if time_series_data:
        save_time_series_data(time_series_data, "egauge_time_series.csv")

def save_historical_data_points(data_points, csv_path):
    """保存历史数据点到CSV文件"""
    print(f"正在保存 {len(data_points)} 个历史数据点到 {csv_path}...")
    
    header = ["datetime", "epoch", "description", "query"] + [f"register_{r}" for r in REGISTERS]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for point in data_points:
            if 'ranges' in point['data'] and len(point['data']['ranges']) > 0:
                rows = point['data']['ranges'][0].get('rows', [])
                if len(rows) > 0:
                    row_data = [
                        point['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                        point['epoch'],
                        point['description'],
                        point['query']
                    ] + rows[0][:len(REGISTERS)]  # 只取需要的寄存器数据
                    writer.writerow(row_data)
    
    print(f"历史数据点保存完成: {csv_path}")

def save_time_series_data(data_points, csv_path):
    """保存时间序列数据到CSV文件"""
    print(f"正在保存 {len(data_points)} 个时间序列数据点到 {csv_path}...")
    
    header = ["datetime", "epoch"] + [f"register_{r}" for r in REGISTERS]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for point in sorted(data_points, key=lambda x: x['timestamp']):
            row_data = [
                point['datetime'].strftime('%Y-%m-%d %H:%M:%S'),
                point['timestamp']
            ] + point['data'][:len(REGISTERS)]  # 只取需要的寄存器数据
            writer.writerow(row_data)
    
    print(f"时间序列数据保存完成: {csv_path}")

if __name__ == "__main__":
    main() 