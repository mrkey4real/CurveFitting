#!/usr/bin/env python3
"""
eGauge 测试脚本 - 测试能获取多久的历史数据
"""

import requests
import datetime as dt
import hashlib
from secrets import token_hex

# ========================== CONFIG ==========================
DEVICE_URI = "https://egauge66683.d.egauge.net"  
USERNAME   = "mingjun"                            
PASSWORD   = "mingjun"                            
REGISTERS  = [29, 33]                             
INTERVAL   = 60                                   
# ============================================================

def get_jwt_token():
    """获取JWT认证令牌"""
    auth_req = requests.get(f"{DEVICE_URI}/api/auth/unauthorized").json()
    realm = auth_req["rlm"]
    nnc = auth_req["nnc"]
    
    cnnc = str(token_hex(64))
    
    ha1_content = f"{USERNAME}:{realm}:{PASSWORD}"
    ha1 = hashlib.md5(ha1_content.encode("utf-8")).hexdigest()
    
    hash_content = f"{ha1}:{nnc}:{cnnc}"
    hash_value = hashlib.md5(hash_content.encode("utf-8")).hexdigest()
    
    payload = {
        "rlm": realm,
        "usr": USERNAME,
        "nnc": nnc,
        "cnnc": cnnc,
        "hash": hash_value
    }
    
    auth_login = requests.post(f"{DEVICE_URI}/api/auth/login", json=payload).json()
    return auth_login["jwt"]

def _iso8601(ts: dt.datetime) -> str:
    return ts.strftime("%Y-%m-%dT%H:%M:%S")

def test_data_availability():
    """测试不同时间范围的数据可用性"""
    jwt = get_jwt_token()
    headers = {"Authorization": f"Bearer {jwt}"}
    
    now = dt.datetime.now()
    
    # 测试不同的时间范围
    test_periods = [
        ("1小时", dt.timedelta(hours=1)),
        ("6小时", dt.timedelta(hours=6)), 
        ("1天", dt.timedelta(days=1)),
        ("3天", dt.timedelta(days=3)),
        ("7天", dt.timedelta(days=7)),
        ("30天", dt.timedelta(days=30)),
        ("90天", dt.timedelta(days=90)),
        ("180天", dt.timedelta(days=180)),
        ("365天", dt.timedelta(days=365))
    ]
    
    for period_name, period_delta in test_periods:
        begin = now - period_delta
        
        params = {
            "format": "json",
            "begin": _iso8601(begin),
            "end": _iso8601(now),
            "interval": INTERVAL,
        }
        
        if REGISTERS:
            params["registers"] = ",".join(map(str, REGISTERS))
        
        url = f"{DEVICE_URI}/api/register"
        
        response = requests.get(url, params=params, headers=headers, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            
            # 检查是否有错误
            if 'error' in data:
                print(f"❌ {period_name}: API错误 - {data['error']}")
                continue
                
            if 'ranges' in data and len(data['ranges']) > 0:
                rows = data['ranges'][0]['rows']
                print(f"✅ {period_name}: 获取到 {len(rows)} 行数据")
                if len(rows) > 0:
                    print(f"   数据范围: {begin.strftime('%Y-%m-%d %H:%M')} 到 {now.strftime('%Y-%m-%d %H:%M')}")
                    # 显示前几行数据样例
                    if len(rows) >= 1:
                        print(f"   第一行数据: {rows[0]}")
            else:
                print(f"❌ {period_name}: 没有数据")
                print(f"   返回的数据结构: {list(data.keys())}")
        else:
            print(f"❌ {period_name}: HTTP错误 {response.status_code}")
            print(f"   响应内容: {response.text[:200]}")

def main():
    print("=== eGauge 数据可用性测试 ===")
    print(f"设备URI: {DEVICE_URI}")
    print(f"寄存器: {REGISTERS}")
    print()
    
    test_data_availability()

if __name__ == "__main__":
    main() 