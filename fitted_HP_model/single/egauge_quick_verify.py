#!/usr/bin/env python3
"""
eGauge 快速验证脚本 - 检查数据单位
"""

from egauge import webapi
import datetime as dt

# ========================== CONFIG ==========================
DEVICE_URI = "https://egauge66683.d.egauge.net"
USERNAME   = "mingjun"
PASSWORD   = "mingjun"
REGISTERS  = [29, 33]
# ============================================================

def main():
    print("=== eGauge 数据单位验证工具 ===")
    
    # 连接设备
    dev = webapi.device.Device(DEVICE_URI, webapi.JWTAuth(USERNAME, PASSWORD))
    
    print("获取最近10个数据点进行分析...")
    
    now = dt.datetime.now()
    
    for i in range(10):
        target_time = now - dt.timedelta(minutes=i)
        target_epoch = int(target_time.timestamp())
        
        print(f"\n时间点 {i+1}: {target_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 测试rate数据
        data = dev.get(f"/register?time={target_epoch}&rate")
        
        if 'error' not in data and 'registers' in data:
            register_info = {}
            for reg in data['registers']:
                if 'idx' in reg and reg['idx'] in REGISTERS:
                    register_info[reg['idx']] = reg
            
            for reg_id in REGISTERS:
                if reg_id in register_info:
                    reg = register_info[reg_id]
                    rate_value = reg['rate']
                    print(f"  寄存器{reg_id} ({reg.get('name', 'Unknown')}): {rate_value} ({reg.get('type', 'Unknown')})")
                    print(f"    原始值: {rate_value}")
                    print(f"    除以1000: {rate_value/1000:.6f}")
                    print(f"    作为kW: {rate_value:.6f}")
                else:
                    print(f"  寄存器{reg_id}: 未找到")
        else:
            error_msg = data.get('error', '未知错误')
            print(f"  错误: {error_msg}")

if __name__ == "__main__":
    main() 