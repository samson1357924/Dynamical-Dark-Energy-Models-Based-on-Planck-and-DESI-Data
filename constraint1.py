# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 02:08:24 2025

@author: samso
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
from sympy import *

a = Symbol('a') 
y = Symbol('y')

f = Function('f')
b = Symbol('b')
a1 = Symbol('a1')
h_0 = 73.48/100
cm = 1/(1.9733*10**(-14))
g = 1/(1.7827*10**(-24))
s = 1/(6.5822*10**(-25))
Mpc = 3.08567758e24*cm

rho_c0_1 = 1.8791*10**(-29)*h_0**2*g/cm**3
G = 6.673e-8*cm**3/g/s**2
H_0 = 100*h_0*1000/(3.08567758*10**(22))/s
M_p = (8*np.pi)**(-1/2)*2.177*10**(-5)*g
rho_c0 = 3
O_m = 0.295
O_r = 0.00010919
O_phi = 1 - O_r - O_m

rho_m0 = O_m*rho_c0
rho_phi0 = O_phi*rho_c0
rho_r0 = O_r*rho_c0
NP_az = np.array([[ 1.00000000e+00,  8.31053215e-01,  6.19908704e-01,  4.68416235e-01,  1.58079894e-04,  1.01624808e-04],
                  [-2.99980901e+00,  5.48678742e-01, -1.97731811e+00, -6.49292166e-01, -1.35553660e+00, -1.00000002e+00]])
NP_az_fixed = NP_az[0]

# 定義 w_phi_a 為 2x6 的變數 (可以初始化為隨機數或某個起始值)
def run_program_A(NP_az):
    function_results = {}

    for x in range(0, len(NP_az[0])-1):
        # 定義 F1_Ori 方程式
        F1_Ori = b*(a-NP_az[0,x])*(a-NP_az[0,x+1])
        
        # 進行積分
        result = integrate(F1_Ori, a)
        # 將 a 的值設定為 1，b 的值設定為 2
        a_value_1 = NP_az[0,x]
        a_value_2 = NP_az[0,x+1]
        b_value = 1
        
        # 求出b
        result_1 = result.subs({a: a_value_1, b: b_value})
        result_2 = result.subs({a: a_value_2, b: b_value})
        b_value = (NP_az[1,x]-NP_az[1,x+1])/(result_1-result_2)
        # 求出c
        result_1 = result.subs({ b: b_value}) - result.subs({a: a_value_1, b: b_value}) + NP_az[1,x]
        
        # 將結果存儲到字典中
        function_results[f"function_{x}"] = result_1
    functions = []
    for i in range(0, len(NP_az[0])-1):
        f = lambdify(a, function_results[f"function_{i}"])
        functions.append(f)
    print(NP_az)

    def model(y, eta):
        rho_m, rho_r, rho_phi, H, a, t, r, D = y
        w_m, w_r, w_phi = 0.0, 1/3, -1.0  # 這裡假設 w_i 為常數，您可以根據需要進行修改    
        # 根據a的值選擇適當的w_phi函數
        if a > NP_az[0, len(NP_az[0])-1]:
            for x in range(0, len(NP_az[0])-1):
                if  a > NP_az[0, x + 1]:
                    w_phi = functions[x](a)
                    break
        else:
            w_phi = -1
        dydeta = [3*a*H*(1+w_m)*rho_m,
                  3*a*H*(1+w_r)*rho_r,
                  3*a*H*(1+w_phi)*rho_phi,
                  1.5*a*H**2 + 0.5*a*(w_m*rho_m + w_r*rho_r + w_phi*rho_phi),
                  -a**2 * H,
                  -a/H_0,
                  -1 / ( np.sqrt(3 * (1 + 3 * rho_m/0.1424 * 0.02242 / (4 * rho_r)))),
                  -1 ]
        return np.array(dydeta)

    # 初始條件
    y0 = np.array([rho_m0, rho_r0, rho_phi0, 1, 1,1/H_0-1.8666203222880963e+40, 10, 0])  # 這裡假設初始值為0，您可以根據需要進行修改
    # 時間點
    eta = np.linspace(0, 4.1, 2000000)  # 這裡假設 eta 從 0 到 10，您可以根據需要進行修改

    # 解微分方程
    sol = odeint(model, y0, eta)

    # 過濾掉 sol[:, 4] < 0 的數據
    valid_indices = sol[:, 4] > 0
    sol = sol[valid_indices]
    eta = eta[valid_indices]

    z = 1/sol[:, 4] -1
    
    # 創建一個字典，其中 B 的元素是鍵，A 的元素是值
    dict_mz = dict(zip(z, sol[:, 0] ))
    dict_rz = dict(zip(z, sol[:, 1] ))
    dict_phiz = dict(zip(z, sol[:, 2] ))
    dict_Hz = dict(zip(z, sol[:, 3] ))
    dict_etaz = dict(zip(z, eta ))
    dict_tz = dict(zip(z, sol[:, 5] ))
    dict_rdz = dict(zip(z, sol[:, 6] ))
    dict_dz = dict(zip(z, sol[:, 7] ))

    # 輸入你想要查找的數字
    input_num = 1060
    #print("z = ",input_num)

    if input_num in dict_mz:
        print(dict_mz[input_num])
    else:
        input_num = min(1060, z[-1])  # 確保 input_num 不超過 z 的最大值
        # 如果在 B 陣列中沒有剛好的數字，找出最接近的兩個數字
        idx = np.searchsorted(z, input_num)
        B1, B2 = z[idx-1], z[idx]
        A1, A2 = sol[:, 0][idx-1], sol[:, 0][idx]

        # 透過前後兩個數字的差分法，計算出對應數值
        A_interp = A1 + (A2 - A1) * (input_num - B1) / (B2 - B1)
        m_1060 = A_interp
        
        A1, A2 = sol[:, 1][idx-1], sol[:, 1][idx]

        # 透過前後兩個數字的差分法，計算出對應數值
        A_interp = A1 + (A2 - A1) * (input_num - B1) / (B2 - B1)
        r_1060 = A_interp
        A1, A2 = sol[:, 2][idx-1], sol[:, 2][idx]

        # 透過前後兩個數字的差分法，計算出對應數值
        A_interp = A1 + (A2 - A1) * (input_num - B1) / (B2 - B1)
        phi_1060 = A_interp
        
        A1, A2 = sol[:, 3][idx-1], sol[:, 3][idx]

        # 透過前後兩個數字的差分法，計算出對應數值
        A_interp = A1 + (A2 - A1) * (input_num - B1) / (B2 - B1)
        H_1060 = A_interp
        #print(r'\rho_m =',A_interp)

        A1, A2 = eta[idx-1], eta[idx]

        # 透過前後兩個數字的差分法，計算出對應數值
        A_interp = A1 + (A2 - A1) * (input_num - B1) / (B2 - B1)
        
        A1, A2 = sol[:, 5][idx-1], sol[:, 5][idx]

        # 透過前後兩個數字的差分法，計算出對應數值
        A_interp = A1 + (A2 - A1) * (input_num - B1) / (B2 - B1)
        
    status_1060 = np.array([m_1060, r_1060, phi_1060, H_1060, 1/1061, A_interp, 0, 0])
    print("z=1060；", status_1060)
    
    if 1089 in dict_rdz:
        print(dict_rdz[1089])
    else:
        # 如果在 B 陣列中沒有剛好的數字，找出最接近的兩個數字
        idx = np.searchsorted(z, 1089)
        B1, B2 = z[idx-1], z[idx]
        A1, A2 = sol[:, 6][idx-1], sol[:, 6][idx]

        # 透過前後兩個數字的差分法，計算出對應數值
        r_d_1089 = A1 + (A2 - A1) * (1089 - B1) / (B2 - B1)
        #print(r'r_d@1060 =',r_d_interp)
        
    if 1060 in dict_rdz:
        print(dict_rdz[1060])
    else:
        # 如果在 B 陣列中沒有剛好的數字，找出最接近的兩個數字
        idx = np.searchsorted(z, 1060)
        B1, B2 = z[idx-1], z[idx]
        A1, A2 = sol[:, 6][idx-1], sol[:, 6][idx]

        # 透過前後兩個數字的差分法，計算出對應數值
        r_d_interp = A1 + (A2 - A1) * (1060 - B1) / (B2 - B1)
        #print(r'r_d@1060 =',r_d_interp)
    r_dinf = sol[:, 6][-1]
    r_d = r_d_interp-r_dinf
    r_d_1089 = r_d_1089-r_dinf
    #print(r'r_d@1e15 =',sol[:, 6][999999])
    #print(r'r_d =',r_d_interp-sol[:, 6][999999])

    # 要處理的 x 值列表
    z_input = [ 0.30, 0.51, 0.71, 0.93, 1.32, 1.49, 2.33,1060,1089]

    # 創建一個空的 NumPy 陣列來儲存結果
    D_A = np.empty(len(z_input))

    # 迴圈處理每個 x 值
    for i, x in enumerate(z_input):
        if x in dict_dz:
            D_A[i] = dict_dz[x]
        else:
            # 如果在 z 陣列中沒有剛好的數字，找出最接近的兩個數字
            idx = np.searchsorted(z, x)
            B1, B2 = z[idx-1], z[idx]
            A1, A2 = sol[:, 7][idx-1], sol[:, 7][idx]
            # 透過前後兩個數字的差分法，計算出對應數值
            D_A[i] = -(A1 + (A2 - A1) * (x - B1) / (B2 - B1))

    # 現在 results 陣列包含了所有計算結果
    D_Theta = r_d/D_A
    D_Theta_1089 = r_d_1089/D_A

    # 創建一個空的 NumPy 陣列來儲存結果
    D_H = np.empty(len(z_input))

    # 迴圈處理每個 x 值
    for i, x in enumerate(z_input):
        if x in dict_Hz:
            D_H[i] = 1/dict_Hz[x]
        else:
            # 如果在 z 陣列中沒有剛好的數字，找出最接近的兩個數字
            idx = np.searchsorted(z, x)
            B1, B2 = z[idx-1], z[idx]
            A1, A2 = sol[:, 3][idx-1], sol[:, 3][idx]
            # 透過前後兩個數字的差分法，計算出對應數值
            D_H[i] = 1/(A1 + (A2 - A1) * (x - B1) / (B2 - B1))

    # 現在 results 陣列包含了所有計算結果
    D_z = r_d/D_H

    D_V = (z_input*D_A**2*D_H)**(1/3)
    result = D_V/r_d

    D_A_Desi = np.array([[0, 13.62, 16.85, 21.71, 27.79, 0, 39.71],
                        [0, 0.25, 0.32, 0.28, 0.69, 0, 0.94]])
    D_H_Desi = np.array([[0, 20.98, 20.08, 17.88, 13.82, 0, 8.52],
                        [0, 0.61, 0.60, 0.35, 0.42, 0, 0.17]])
    D_V_Desi = np.array([[7.93, 0, 0, 0, 0, 26.07, 0 ],
                        [0.15, 0, 0, 0, 0, 0.67, 0]])
    Plank_2018 = np.array([0.0104109,0.000003])

    # 誤差計算

    D_A_error = np.array([])
    for i in range(0, len(D_A_Desi[0])):
        if D_A_Desi[0,i]==0:
            D_A_error = np.append(D_A_error,0)
        else:
            S = abs(1/D_Theta[i]-D_A_Desi[0,i])/D_A_Desi[1,i]
            D_A_error = np.append(D_A_error,S)
    D_H_error = np.array([])
    for i in range(0, len(D_H_Desi[0])):
        if D_H_Desi[0,i]==0:
            D_H_error = np.append(D_H_error,0)
        else:
            S = abs(1/D_z[i]-D_H_Desi[0,i])/D_H_Desi[1,i]
            D_H_error = np.append(D_H_error,S)
    D_V_error = np.array([])
    for i in range(0, len(D_V_Desi[0])):
        if D_V_Desi[0,i]==0:
            D_V_error = np.append(D_V_error,0)
        else:
            S = abs(result[i]-D_V_Desi[0,i])/D_V_Desi[1,i]
            D_V_error = np.append(D_V_error,S)
    result_P = (D_Theta_1089[8] - Plank_2018[0]) / Plank_2018[1]
    Plank_2018_error = np.array([result_P,0,0,0,0,0,0])
    #chi_squared
    D_A_error_sq = np.square(D_A_error)
    D_H_error_sq = np.square(D_H_error)
    D_V_error_sq = np.square(D_V_error)
    chi_squared_DESI = np.sum(D_A_error_sq)+np.sum(D_H_error_sq)+np.sum(D_V_error_sq)
    chi_squared_plank = np.sum(Plank_2018_error)
    chi_squared_SUM = chi_squared_DESI + (chi_squared_plank/10)**2
    return chi_squared_SUM, status_1060
def run_program_B(status, d_m1060):
    # 創建一個字典來存儲每個函數的結果
    function_results = {}

    for x in range(0, len(NP_az[0])-1):
        # 定義 F1_Ori 方程式
        F1_Ori = b*(a-NP_az[0,x])*(a-NP_az[0,x+1])
        
        # 進行積分
        result = integrate(F1_Ori, a)
        # 將 a 的值設定為 1，b 的值設定為 2
        a_value_1 = NP_az[0,x]
        a_value_2 = NP_az[0,x+1]
        b_value = 1
        
        # 求出b
        result_1 = result.subs({a: a_value_1, b: b_value})
        result_2 = result.subs({a: a_value_2, b: b_value})
        b_value = (NP_az[1,x]-NP_az[1,x+1])/(result_1-result_2)
        # 求出c
        result_1 = result.subs({ b: b_value}) - result.subs({a: a_value_1, b: b_value}) + NP_az[1,x]
        
        # 將結果存儲到字典中
        function_results[f"function_{x}"] = result_1
    # Reverse the order of NP_az 
    NP_az_rev = NP_az[:, ::-1]
    # 自定義反轉順序
    function_results_rev = {
        "function_0": function_results["function_4"],
        "function_1": function_results["function_3"],
        "function_2": function_results["function_2"],
        "function_3": function_results["function_1"],
        "function_4": function_results["function_0"]
    }
    functions_rev = []
    for i in range(0, len(NP_az_rev[0])-1):
        f = lambdify(a, function_results_rev[f"function_{i}"])
        functions_rev.append(f)
    # 定義微分方程
    def model(y, eta):
        rho_m, rho_r, rho_phi, H, a, t, r, D, d_m, v = y
        w_m, w_r, w_phi = 0.0, 1/3, -1.0  # 這裡假設 w_i 為常數，您可以根據需要進行修改    
        # 根據a的值選擇適當的w_phi函數
        if a < NP_az_rev[0, len(NP_az_rev[0])-1]:
            for x in range(0, len(NP_az_rev[0])-1):
                if  a < NP_az_rev[0, x + 1]:
                    w_phi = functions_rev[x](a)
                    break
        else:
            w_phi = -1
        O_m = rho_m/(rho_m+rho_r+rho_phi)
        dydeta = [-3*a*H*(1+w_m)*rho_m,
                  -3*a*H*(1+w_r)*rho_r,
                  -3*a*H*(1+w_phi)*rho_phi,
                  -1.5*a*H**2 - 0.5*a*(w_m*rho_m + w_r*rho_r + w_phi*rho_phi),
                  a**2 * H,
                  a/H_0,
                  1 / ( np.sqrt(3 * (1 + 3 * rho_m/0.1424 * 0.02242 / (4 * rho_r)))),
                  1, 
                  v, 
                  -a*H*v + 3/2*a**2 * H**2 * O_m * d_m]
        return np.array(dydeta)

    # 初始條件
    y0 = np.concatenate([status, np.array([d_m1060, 0])])    # 時間點
    eta = np.linspace(0, 4, 1000000)  # 這裡假設 eta 從 0 到 10，您可以根據需要進行修改

    # 解微分方程
    sol_B = odeint(model, y0, eta)

    # 過濾掉 sol_B[:, 4] < 0 的數據
    valid_indices = sol_B[:, 4] <= 1
    sol_B = sol_B[valid_indices]
    eta = eta[valid_indices]
    
    #print("sol_B[:, 8] 的最后一个值:", sol_B[-1, 8])
    z = 1/sol_B[:, 4] -1
    rho_c = 3*sol_B[:, 3]**2

    Omega_m = sol_B[:, 0]/rho_c
    Omega_r = sol_B[:, 1]/rho_c
    Omega_phi = sol_B[:, 2]/rho_c
    Omega_b = sol_B[:, 0]/rho_c/0.1424 * 0.02242
    Omega_b_0 = rho_m0/rho_c0/0.1424 * 0.02242

    # 計算gamma
    gamma = (np.log(sol_B[-1, 8]) - np.log(sol_B[-2, 8]))/(np.log(sol_B[-1, 4]) - np.log(sol_B[-2, 4]))
    gamma = np.log(gamma)/np.log(Omega_m[-1])
    chi_squared_gamma = ((0.633-gamma)/0.024)**2
    d_m0 = sol_B[-1, 8]
    print("z =", z[-1])
    print("d_m0 =", d_m0,"a_0 = ",sol_B[-1, 4])
    return d_m0, chi_squared_gamma
# 內層計算函數：根據 NP_az 輸出 d_m0 和 chi_squared
def inner_optimization(NP_az):
    print(NP_az)
    # 恢復 NP_az_0
    first_element = np.array([1])  # 固定的第一個元素
    NP_az_0_recovered = np.hstack((first_element, NP_az[:5].flatten()))

    # 恢復 NP_az_1
    NP_az_1_recovered = NP_az[5:11]

    # 恢復 d_m1060
    d_m1060 = NP_az[11]
    print(d_m1060)

    # 確保陣列都是 2D
    NP_az_0_recovered = NP_az_0_recovered.reshape(1, -1)
    NP_az_1_recovered = NP_az_1_recovered.reshape(1, -1)

    # 重建 NP_az
    NP_az = np.vstack((NP_az_0_recovered, NP_az_1_recovered))

    # 這裡呼叫程式A，獲取 chi_squared_SUM 和 status_1060
    chi_squared_SUM, status_1060 = run_program_A(NP_az)
    # 這裡呼叫程式B，獲取 d_m0 和 chi_squared_gamma
    d_m0, chi_squared_gamma = run_program_B(status_1060, d_m1060)
    
    return d_m0, chi_squared_SUM, chi_squared_gamma


# 目標函數：最小化 chi_squared_SUM + chi_squared_gamma
def objective(NP_az):


    # 獲取 chi_squared_SUM 和 chi_squared_gamma
    d_m0, chi_squared_SUM, chi_squared_gamma = inner_optimization(NP_az)
    chi_squared_SUM = chi_squared_SUM + chi_squared_gamma
    print("chi_squared_SUM + chi_squared_gamma = ", chi_squared_SUM)
    return chi_squared_SUM  # 目標：最小化這個總和


# 內層約束條件：確保 d_m0 = 1
def constraint_d_m0(NP_az):

    # 確保回傳值正確
    d_m0, _, _ = inner_optimization(NP_az)

    return d_m0 - 1  # 這個值要等於 0

# 遞減約束，確保 NP_az[0] 元素按遞減排列
def decreasing_constraint(NP_az_flat):
    NP_az_0 = np.hstack(([1], NP_az_flat[:5]))  # 第一個元素固定為 1
    return np.diff(NP_az_0)  # 返回每個相鄰元素的差值（應該是負數）

# 初始條件設置
NP_az_0_initial = NP_az[0][1:]  # 排除掉第一個固定的 1
NP_az_1_initial = NP_az[1]
d_m1060_initial = np.array([0.002472748450283206])
initial_guess = np.hstack((NP_az_0_initial, NP_az_1_initial, d_m1060_initial))


# 設置範圍
bounds_0 = [
    (1/1.6, 1),        # NP_az[0][1] 的範圍
    (1/2.1, 1/1.2),    # NP_az[0][2] 的範圍
    (1/9000, 1/1.6),   # NP_az[0][3] 的範圍
    (1/10000, 1/2.1),  # NP_az[0][4] 的範圍
    (1e-5, 1/3)        # NP_az[0][5] 的範圍
]
bounds_1 = [(-3, 2)] * len(NP_az_1_initial)  # NP_az[1] 的範圍 [-3, 2]
bounds_2 = [(0, 0.3)]
bounds = bounds_0 + bounds_1 + bounds_2

# 約束條件：確保 NP_az[0] 是遞減的
constraints = [{'type': 'ineq', 'fun': decreasing_constraint}]
constraints_1 =[{'type': 'eq', 'fun': constraint_d_m0}]
constraints = constraints + constraints_1

# 使用 SLSQP 優化
result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints, method='SLSQP')

# 優化結果
NP_az_0_optimized = np.hstack(([1], result.x[:5]))  # 第一個數字固定為 1
NP_az_1_optimized = result.x[5:11]
d_m1060_optimized = result.x[11]
NP_az_optimized = np.vstack((NP_az_0_optimized, NP_az_1_optimized))
# 最優化的 w_phi_a
optimal_w_phi_a = result.x.reshape(2, 6)

print("最佳 w_phi_a:", NP_az_optimized)
print("最佳 d_m1060:", d_m1060_optimized)

print("最小化的 chi_squared_SUM + chi_squared_gamma:", result.fun)
