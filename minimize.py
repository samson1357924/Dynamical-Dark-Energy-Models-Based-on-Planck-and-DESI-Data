# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 21:35:24 2025

@author: samso
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import w_phi_a
from w_phi_a import NP_az
from w_phi_a import NP_az_rev
from sympy import *
from scipy.optimize import minimize
a = Symbol('a') 

h_0 = 73.48/100
cm = 1/(1.9733*10**(-14))
g = 1/(1.7827*10**(-24))
s = 1/(6.5822*10**(-25))
Mpc = 3.08567758e24*cm

rho_c0_1 = 1.8791*10**(-29)*h_0**2*g/cm**3
G = 6.673e-8*cm**3/g/s**2
H_0 = 100*h_0*1000/(3.08567758*10**(22))/s
M_p = (8*np.pi)**(-1/2)*2.177*10**(-5)*g
rho_c0 = 27000000
rho_m0 = 0.23*rho_c0
rho_phi0 = 0.002*rho_c0
rho_r0 = 0.768*rho_c0
rho_md = 1057035591.0486697
rho_rd = 415113098.8879818
rho_phid = 0.01435356935460154
Hd = 16277.371739511218

functions = []
for i in range(0, len(NP_az_rev[0])-1):
    f = lambdify(a, w_phi_a.function_results[f"function_{i}"])
    functions.append(f)
functions_rev = []
for i in range(0, len(NP_az_rev[0])-1):
    f = lambdify(a, w_phi_a.function_results_rev[f"function_{i}"])
    functions_rev.append(f)
    
    
# 微分方程模型（你提供的 model 函数保持不变）
def model(y, eta):
    rho_m, rho_r, rho_phi, H, a, t, r, D, d_m, v = y
    w_m, w_r, w_phi = 0.0, 1/3, -1.0
    if a < NP_az_rev[0, len(NP_az_rev[0])-1]:
        for x in range(0, len(NP_az_rev[0])-1):
            if a < NP_az_rev[0, x + 1]:
                w_phi = functions_rev[x](a)
                break
    else:
        w_phi = -1
    O_m = rho_m / (rho_m + rho_r + rho_phi)
    dydeta = [
        -3 * a * H * (1 + w_m) * rho_m,
        -3 * a * H * (1 + w_r) * rho_r,
        -3 * a * H * (1 + w_phi) * rho_phi,
        -1.5 * a * H**2 - 0.5 * a * (w_m * rho_m + w_r * rho_r + w_phi * rho_phi),
        a**2 * H,
        a / H_0,
        1 / (np.sqrt(3 * (1 + 3 * rho_m / 0.1424 * 0.02242 / (4 * rho_r)))),
        1,
        v,
        -a * H * v + 3 / 2 * a**2 * H**2 * O_m * d_m,
    ]
    return np.array(dydeta)

# 初始条件
y0 = np.array([rho_md, rho_rd, rho_phid, 
               Hd / h_0, 1/1061, 1.6890398288367673e+37, 
               10, 0, 0.002472775315401099, 0])

# 时间点
eta = np.linspace(0, 4, 1000000)

# 定义目标函数
def objective(y0_8):
    y0[8] = y0_8  # 修改 y0[8]
    sol = odeint(model, y0, eta)
    valid_indices = sol[:, 4] <= 1  # 过滤条件
    sol = sol[valid_indices]
    if len(sol) == 0:
        return np.inf  # 防止解不存在的情况
    return abs(sol[-1, 8] - 1)  # 返回最后一个值与 1 的差距

# 优化 y0[8]
result = minimize(objective, y0[8], bounds=[(1e-6, 1e-1)], method="L-BFGS-B")

# 最优值
optimal_y0_8 = result.x[0]
y0[8] = optimal_y0_8
print("最优 y0[8]:", optimal_y0_8)

# 用优化后的 y0[8] 解方程并查看结果
sol = odeint(model, y0, eta)
valid_indices = sol[:, 4] <= 1
sol = sol[valid_indices]
eta = eta[valid_indices]

print("sol[:, 8] 的最后一个值:", sol[-1, 8])
