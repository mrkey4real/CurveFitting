# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:28:47 2025

@author: qizixuan
"""

import numpy as np
import matplotlib.pyplot as plt

# 定义曲线公式
def biquadratic_curve_function(x, y, c1, c2, c3, c4, c5, c6):
    return c1 + c2*x + c3*x**2 + c4*y + c5*y**2 + c6*x*y

def cubic_curve_function(x, c1, c2, c3, c4):
    return c1 + c2*x + c3*x**2 + c4*x**3

#%% default
# 第一个曲线的系数 (Default cooling cap)
c1 = -5.32142427412103E-03
c2 = 2.14018312686104E-02
c3 = -5.15203062629478E-04
c4 = 1.33325987657944E-03
c5 = 1.91413838050937E-04
c6 = -2.16256735399078E-04

# # 第一个曲线的系数 (Default cooling eir)
# c1 = 0.342414409
# c2 = 0.034885008
# c3 = -0.0006237
# c4 = 0.004977216
# c5 = 0.000437951
# c6 = -0.000728028

# 第一个曲线的系数 (Default heating cap)
# c1 = 0.758746
# c2 = 0.027626
# c3 = 0.000148716
# c4 = 0.0000034992

# # 第一个曲线的系数 (Default heating eir)
# c1 = 1.19248
# c2 = -0.0300438
# c3 = 0.00103745
# c4 = -0.000023328

#%% fitted
# IDF,Curve:Biquadratic,Daikin Fit Heating Cap Normal-FT,7413.51381177336,-2.44734920536475E-05,-1.54057198166142E-16,163.391700071471,-0.20752748694072,-8.92081898185491E-05,-50,46,0,46.11111,,,,,;
# 第二个曲线的系数 (Fitted DAIKIN)

c1_2 = 6873.81898143704
c2_2 = -180.24571323053
c3_2 = 11.1721324730084
c4_2 = 155.652300354184
c5_2 = -2.35465856705292
c6_2 = -2.99200773993705

# 定义 x 和 y 的范围
x = np.linspace(-50, 46, 100)
y = np.linspace(-50, 46, 100)
x, y = np.meshgrid(x, y)

# 计算第一个曲线的值 (Default)
z1 = biquadratic_curve_function(x, y, c1, c2, c3, c4, c5, c6)
# z1 = cubic_curve_function(x, c1, c2, c3, c4)

# 计算第二个曲线的值 (Fitted DAIKIN)
z2 = biquadratic_curve_function(x, y, c1_2, c2_2, c3_2, c4_2, c5_2, c6_2)

# 绘制总览图
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
surf1 = ax.plot_surface(x, y, z1, cmap='viridis', edgecolor='none', alpha=0.7)
surf2 = ax.plot_surface(x, y, z2, cmap='plasma', edgecolor='none', alpha=0.7)

ax.set_title("Overall View: Heating Boost cap Default vs Fitted", fontsize=14)
ax.set_xlabel("X", fontsize=12)
ax.set_ylabel("Y", fontsize=12)
ax.set_zlabel("Z", fontsize=12)
fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=10, label="Default Curve")
fig.colorbar(surf2, ax=ax, shrink=0.5, aspect=10, label="Fitted Curve")
plt.show()

# 重新设置视角以突出 z 轴
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
surf1 = ax.plot_surface(x, y, z1, cmap='viridis', edgecolor='none', alpha=0.7)
surf2 = ax.plot_surface(x, y, z2, cmap='plasma', edgecolor='none', alpha=0.7)

ax.set_title("Focused View: Emphasizing Z-axis (Heating Boost cap Default vs Fitted)", fontsize=14)
ax.set_xlabel("X", fontsize=12)
ax.set_ylabel("Y", fontsize=12)
ax.set_zlabel("Z", fontsize=12)

# 设置角度：俯视角度，清晰对比 Z 值
ax.view_init(elev=30, azim=210)
fig.colorbar(surf1, ax=ax, shrink=0.5, aspect=10, label="Default Curve")
fig.colorbar(surf2, ax=ax, shrink=0.5, aspect=10, label="Fitted Curve")

plt.show()
