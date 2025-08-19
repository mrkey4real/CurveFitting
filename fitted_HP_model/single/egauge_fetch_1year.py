#!/usr/bin/env python3
"""
eGauge minute‑level data downloader - 1年数据版本
==================================

获取最近1年的1分钟级别eGauge数据

Dependencies:
    pip install requests

使用JWT令牌认证方式
"""

import requests
import datetime as dt
import csv
import hashlib
from secrets import token_hex

# ========================== CONFIG ==========================
DEVICE_URI = "https://egauge66683.d.egauge.net"  # ← 你的仪表地址
USERNAME   = "mingjun"                            # ← eGauge 账号
PASSWORD   = "mingjun"                            # ← eGauge 密码
REGISTERS  = [29, 33]                             # ← 寄存器 ID 列表
OUTPUT_CSV = "egauge_data_1year.csv"             # 输出文件名
INTERVAL   = 60                                   # 60 秒 = 1 分钟
# ============================================================

def get_jwt_token():
    """获取JWT认证令牌"""
    print("正在获取JWT认证令牌...")
    
    # 获取realm和server nonce
    auth_req = requests.get(f"{DEVICE_URI}/api/auth/unauthorized").json()
    realm = auth_req["rlm"]
    nnc = auth_req["nnc"]
    
    cnnc = str(token_hex(64))  # 生成客户端nonce
    
    # 生成hash
    # ha1 = MD5(usr:rlm:pwd)
    # hash = MD5(ha1:nnc:cnnc)
    ha1_content = f"{USERNAME}:{realm}:{PASSWORD}"
    ha1 = hashlib.md5(ha1_content.encode("utf-8")).hexdigest()
    
    hash_content = f"{ha1}:{nnc}:{cnnc}"
    hash_value = hashlib.md5(hash_content.encode("utf-8")).hexdigest()
    
    # 生成payload
    payload = {
        "rlm": realm,
        "usr": USERNAME,
        "nnc": nnc,
        "cnnc": cnnc,
        "hash": hash_value
    }
    
    # POST请求获取JWT
    auth_login = requests.post(f"{DEVICE_URI}/api/auth/login", json=payload).json()
    jwt = auth_login["jwt"]
    
    print("JWT令牌获取成功")
    return jwt

def _iso8601(ts: dt.datetime) -> str:
    """Return ISO‑8601 string without timezone offset (meter local time)."""
    return ts.strftime("%Y-%m-%dT%H:%M:%S")

def query_egauge_batch(begin: dt.datetime, end: dt.datetime, interval: int = INTERVAL):
    """分批查询eGauge数据，避免单次请求数据量过大"""
    jwt = get_jwt_token()
    headers = {"Authorization": f"Bearer {jwt}"}
    
    all_rows = []
    current_begin = begin
    
    # 每次查询7天的数据，避免数据量过大
    batch_days = 7
    total_batches = ((end - begin).days // batch_days) + 1
    current_batch = 0
    
    while current_begin < end:
        current_batch += 1
        current_end = min(current_begin + dt.timedelta(days=batch_days), end)
        
        print(f"正在获取批次 {current_batch}/{total_batches}: {current_begin.strftime('%Y-%m-%d')} 到 {current_end.strftime('%Y-%m-%d')} 的数据...")
        
        params = {
            "format": "json",
            "begin": _iso8601(current_begin),
            "end": _iso8601(current_end),
            "interval": interval,
        }
        
        if REGISTERS:
            params["registers"] = ",".join(map(str, REGISTERS))
        
        # 使用新的API端点
        url = f"{DEVICE_URI}/api/register"
        
        response = requests.get(url, params=params, headers=headers, timeout=120)
        response.raise_for_status()
        
        batch_data = response.json()
        
        if 'ranges' in batch_data and len(batch_data['ranges']) > 0:
            batch_rows = batch_data['ranges'][0]['rows']
            all_rows.extend(batch_rows)
            print(f"  获取到 {len(batch_rows)} 行数据")
        else:
            print("  没有获取到数据")
        
        current_begin = current_end
        
        # 每隔几个批次重新获取JWT令牌，防止令牌过期
        if current_batch % 10 == 0:
            jwt = get_jwt_token()
            headers = {"Authorization": f"Bearer {jwt}"}
    
    print(f"数据获取完成，总共 {len(all_rows)} 行")
    
    # 构造完整的数据结构
    return {
        "ts": int(begin.timestamp()),
        "interval": interval,
        "ranges": [{"rows": all_rows}]
    }

def save_csv(json_data: dict, csv_path: str, begin_time: dt.datetime):
    """Save rows from JSON payload to a timestamped CSV file."""
    interval  = json_data["interval"]         # sampling interval (s)
    rows      = json_data["ranges"][0]["rows"]
    header    = ["timestamp"] + [f"register_{r}" for r in REGISTERS]

    print(f"正在保存数据到 {csv_path}...")
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for i, values in enumerate(rows):
            # 从开始时间计算每一行的时间戳
            t = begin_time + dt.timedelta(seconds=i * interval)
            writer.writerow([t.isoformat()] + values)
    
    print(f"数据保存完成")

def main():
    print("=== eGauge 1年数据获取工具 ===")
    
    now = dt.datetime.now()
    begin = now - dt.timedelta(days=365)      # 最近 1 年
    
    print(f"开始时间: {begin.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"寄存器列表: {REGISTERS}")
    print(f"数据间隔: {INTERVAL} 秒")
    print()
    
    data = query_egauge_batch(begin, now)
    save_csv(data, OUTPUT_CSV, begin)
    
    total_rows = len(data['ranges'][0]['rows'])
    expected_rows = int((now - begin).total_seconds() / INTERVAL)
    
    print()
    print("=== 数据获取完成 ===")
    print(f"输出文件: {OUTPUT_CSV}")
    print(f"实际行数: {total_rows}")
    print(f"预期行数: {expected_rows}")
    print(f"数据完整度: {total_rows/expected_rows*100:.1f}%" if expected_rows > 0 else "无法计算")
    print(f"时间跨度: {total_rows * INTERVAL / (3600*24):.1f} 天")

if __name__ == "__main__":
    main() 