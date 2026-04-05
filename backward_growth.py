# phys: backward integration from present-day (t=0) to the early universe
"""
Created on Fri Nov 10 01:59:31 2023

@author: Z13
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
import w_phi_a
from w_phi_a import NP_az
from sympy import *

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
rho_c0 = 3
O_m = 0.295
O_r = 0.00010919
O_phi = 1 - O_r - O_m
print(O_phi)

rho_m0 = O_m*rho_c0
rho_phi0 = O_phi*rho_c0
rho_r0 = O_r*rho_c0

functions = []
for i in range(0, len(NP_az[0])-1):
    f = lambdify(a, w_phi_a.function_results[f"function_{i}"])
    functions.append(f)
# 定義微分方程
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
eta_z13 = 3.28
# 時間點
eta = np.linspace(0, eta_z13, 2000000)  # 這裡假設 eta 從 0 到 10，您可以根據需要進行修改

# 解微分方程
sol = odeint(model, y0, eta)

# 過濾掉 sol[:, 4] < 0 的數據
valid_indices = sol[:, 4] > 0
sol = sol[valid_indices]
eta = eta[valid_indices]

z = 1/sol[:, 4] -1
rho_c = 3*sol[:, 3]**2


w_phi = []

for a in sol[:, 4]: # 對 A 中的每個元素 x
    if a > NP_az[0, len(NP_az[0])-1]:
        for x in range(0, len(NP_az[0])-1):
            if  a >= NP_az[0, x + 1]:
                w_phi.append(functions[x](a))
                break
    else:
        w_phi.append(-1)
w_phi_np = np.array(w_phi)
#print(w_phi_np)
q = sol[:, 1]/rho_c + 1/2*sol[:, 0]/rho_c + 1/2*sol[:, 2]/rho_c*(1 +3*w_phi_np)
Omega_m = sol[:, 0]/rho_c
Omega_r = sol[:, 1]/rho_c
Omega_phi = sol[:, 2]/rho_c
Omega_b = sol[:, 0]/rho_c/0.1424 * 0.02242
Omega_b_0 = rho_m0/rho_c/0.1424 * 0.02242
Theta_star_0 = 1.04119e-2

C_s = 1 / (3 * (1 + 3 * Omega_b / (4 * Omega_r)))**(1/2)

'''
# 創建一個空的陣列 C
w_phi_avg_reversed = np.zeros(1000000)
# 使用[::-1]索引來顛倒 A
Omega_phi_reversed = Omega_phi[::-1]
w_phi_np_reversed = w_phi_np[::-1]

# 對於每一個索引 i，計算 A 和 B 的前 i 個元素的內積，然後除以 B 的前 i 個元素的總和
for i in range(1, 1000000):
    w_phi_avg_reversed[i] = np.dot(w_phi_np_reversed[:i],Omega_phi_reversed[:i]) / np.sum(Omega_phi_reversed[:i])
w_phi_avg = w_phi_avg_reversed[::-1]
'''

# 創建一個字典，其中 B 的元素是鍵，A 的元素是值
dict_mz = dict(zip(z, sol[:, 0] ))
dict_rz = dict(zip(z, sol[:, 1] ))
dict_phiz = dict(zip(z, sol[:, 2] ))
dict_Hz = dict(zip(z, sol[:, 3] ))
dict_etaz = dict(zip(z, eta ))
dict_tz = dict(zip(z, sol[:, 5] ))
dict_rdz = dict(zip(z, sol[:, 6] ))
dict_dz = dict(zip(z, sol[:, 7] ))
final_z = z[-1]
print('z = ',final_z)
print(r'\rho_m = ',sol[-1, 0])
print(r'\rho_r = ',sol[-1, 1])
print(r'\rho_phi = ',sol[-1, 2])
print('H = ',sol[-1, 3])
print('eta = ',eta[-1])
print('t = ',sol[-1, 5])
print('t_0 = ',1/H_0)


'''
# 輸入你想要查找的數字
input_num = 1e13
print("z = ",input_num)

if input_num in dict_mz:
    print(dict_mz[input_num])
else:
    # 如果在 B 陣列中沒有剛好的數字，找出最接近的兩個數字
    idx = np.searchsorted(z, input_num)
    B1, B2 = z[idx-1], z[idx]
    A1, A2 = sol[:, 0][idx-1], sol[:, 0][idx]

    # 透過前後兩個數字的差分法，計算出對應數值
    A_interp = A1 + (A2 - A1) * (input_num - B1) / (B2 - B1)
    print(r'\rho_m =',A_interp)

if input_num in dict_rz:
    print(dict_rz[input_num])
else:
    # 如果在 B 陣列中沒有剛好的數字，找出最接近的兩個數字
    idx = np.searchsorted(z, input_num)
    B1, B2 = z[idx-1], z[idx]
    A1, A2 = sol[:, 1][idx-1], sol[:, 1][idx]

    # 透過前後兩個數字的差分法，計算出對應數值
    A_interp = A1 + (A2 - A1) * (input_num - B1) / (B2 - B1)
    print(r'\rho_r =',A_interp)

if input_num in dict_phiz:
    print(dict_phiz[input_num])
else:
    # 如果在 B 陣列中沒有剛好的數字，找出最接近的兩個數字
    idx = np.searchsorted(z, input_num)
    B1, B2 = z[idx-1], z[idx]
    A1, A2 = sol[:, 2][idx-1], sol[:, 2][idx]

    # 透過前後兩個數字的差分法，計算出對應數值
    A_interp = A1 + (A2 - A1) * (input_num - B1) / (B2 - B1)
    print(r'\rho_phi =',A_interp)

if input_num in dict_Hz:
    print(dict_Hz[input_num]*h_0*100)
else:
    # 如果在 B 陣列中沒有剛好的數字，找出最接近的兩個數字
    idx = np.searchsorted(z, input_num)
    B1, B2 = z[idx-1], z[idx]
    A1, A2 = sol[:, 3][idx-1], sol[:, 3][idx]

    # 透過前後兩個數字的差分法，計算出對應數值
    A_interp = A1 + (A2 - A1) * (input_num - B1) / (B2 - B1)
    print(r'H =',A_interp*h_0*100)

if input_num in dict_etaz:
    print(dict_etaz[input_num])
else:
    # 如果在 B 陣列中沒有剛好的數字，找出最接近的兩個數字
    idx = np.searchsorted(z, input_num)
    B1, B2 = z[idx-1], z[idx]
    A1, A2 = eta[idx-1], eta[idx]

    # 透過前後兩個數字的差分法，計算出對應數值
    A_interp = A1 + (A2 - A1) * (input_num - B1) / (B2 - B1)
    print(r'eta =',A_interp)

if input_num in dict_tz:
    print(dict_tz[input_num])
else:
    # 如果在 B 陣列中沒有剛好的數字，找出最接近的兩個數字
    idx = np.searchsorted(z, input_num)
    B1, B2 = z[idx-1], z[idx]
    A1, A2 = sol[:, 5][idx-1], sol[:, 5][idx]

    # 透過前後兩個數字的差分法，計算出對應數值
    A_interp = A1 + (A2 - A1) * (input_num - B1) / (B2 - B1)
    print(r't =',A_interp)
'''    
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
# 將陣列轉換成 DataFrame
df = pd.DataFrame({
    'z': z_input,
    '\Delta_Theta':D_Theta,
    'D_A/r_d': 1/D_Theta,
    'D_H/r_d': 1/D_z,
    'D_V/r_d': result,
    'D_Theta_1089': D_Theta_1089
})

df_error = pd.DataFrame({
    'D_A_error': D_A_error,
    'D_H_error':D_H_error,
    'D_V_error': D_V_error,
    'P_2018':Plank_2018_error
})

#chi_squared
D_A_error_sq = np.square(D_A_error)
D_H_error_sq = np.square(D_H_error)
D_V_error_sq = np.square(D_V_error)
chi_squared_DESI = np.sum(D_A_error_sq)+np.sum(D_H_error_sq)+np.sum(D_V_error_sq)
chi_squared_plank = np.sum(Plank_2018_error)
chi_squared_SUM = chi_squared_DESI + (chi_squared_plank/30)**2
print('chi_squared_DESI = ',chi_squared_DESI)
print('chi_plank = ',chi_squared_plank)
print('chi_squared_SUM = ',chi_squared_SUM)
# 顯示 DataFrame
print(df)
print(df_error)

with pd.ExcelWriter('output.xlsx') as writer:
    df.to_excel(writer, index=False, sheet_name='number')
    df_error.to_excel(writer, index=False, sheet_name='error')

f, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w',figsize=(10, 8))
# plot the same data on both axes
ax.plot(1+z, sol[:, 0]/rho_c, label=r'$\Omega_m$')
ax.plot(1+z, sol[:, 1]/rho_c, label=r'$\Omega_r$')
ax.plot(1+z, sol[:, 2]/rho_c, label=r'$\Omega_\phi$')
#plt.plot(z, sol[:, 3], label=r'$H$')
#plt.plot(z, sol[:, 4], label='a')
ax.plot(1+z, w_phi, label=r'$w_\phi$')
#ax.plot(1/w_phi_a.all_a_values, w_phi_a.all_y_values, label=r'$w_\phi$')
#ax.plot(1+z, q, label='q')
ax.plot(1+z, sol[:, 5]*H_0, label=r'$H_0t$')
#ax.plot(1+z, w_phi_avg, label=r'$\langle w_\phi \rangle$')
ax2.plot(1+z, sol[:, 0]/rho_c, label=r'$\Omega_m$')
ax2.plot(1+z, sol[:, 1]/rho_c, label=r'$\Omega_r$')
ax2.plot(1+z, sol[:, 2]/rho_c, label=r'$\Omega_\phi$')
#plt.plot(z, sol[:, 3], label=r'$H$')
#plt.plot(z, sol[:, 4], label='a')
ax2.plot(1+z, w_phi, label=r'$w_\phi$')
#ax2.plot(1/w_phi_a.all_a_values, w_phi_a.all_y_values, label=r'$w_\phi$')
#ax2.plot(1+z, q, label='q')
ax2.plot(1+z, sol[:, 5]*H_0, label=r'$H_0t$')
#ax2.plot(1+z, w_phi_avg, label=r'$\langle w_\phi \rangle$')


ax.set_xlim(1, 10)
ax2.set_xlim(100000, 1e15)

# hide the spines between ax and ax2
ax.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax.yaxis.tick_left()
ax.tick_params(labelleft='off')
ax2.yaxis.tick_right()

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((1-d, 1+d), (-d, +d), **kwargs)
ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1-d, 1+d), **kwargs)
ax2.plot((-d, +d), (-d, +d), **kwargs)
ax.set_xscale('log')
ax2.set_xscale('log')
ax.grid()
ax2.grid()
plt.legend(loc='best')
plt.xlabel(r'$1+z$')
f.subplots_adjust(hspace=0.01)
# What's cool about this is that now if we vary the distance between
# ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
# the diagonal lines will move accordingly, and stay right at the tips
# of the spines they are 'breaking'
plt.savefig('ax.png')
plt.show()


plt.figure(figsize=(10, 8))
plt.plot(1+z, sol[:, 0]/rho_c, label=r'$\Omega_m$')
plt.plot(1+z, sol[:, 1]/rho_c, label=r'$\Omega_r$')
plt.plot(1+z, sol[:, 2]/rho_c, label=r'$\Omega_\phi$')
#plt.plot(z, sol[:, 3], label=r'$H$')
#plt.plot(z, sol[:, 4], label='a')
#plt.plot(1+z, w_phi, label=r'$w_\phi$')
plt.plot(1+z, q, label='q')
plt.plot(1+z, sol[:, 5]*H_0, label=r'$H_0t$')
plt.plot(1+z,w_phi, label=r'$w_\phi$')
#plt.plot(1+z, w_phi_avg, label=r'$\langle w_\phi \rangle$')
plt.legend(loc='best')
plt.xlabel(r'$1+z$')
plt.xlim( 1,1e15) # 限制 x 軸的範圍從 0 到 10
plt.xscale('log')

plt.grid()
plt.savefig('1~1e15.png')
plt.show()

array_str = np.array2string(NP_az)

z_input = np.array([0.30, 0.51, 0.71, 0.93, 1.32, 1.49, 2.33])

plt.figure(figsize=(10, 8))
plt.plot(1+z, -sol[:, 7]/r_d, label=r'$D_A$')
plt.plot(1+z, 1/sol[:, 3]/r_d, label=r'$D_H$')
plt.plot(1+z, (z*sol[:, 7]**2/sol[:, 3])**(1/3)/r_d, label=r'$D_V$')
plt.errorbar(1+z_input, D_A_Desi[0], yerr=D_A_Desi[1], fmt='o')
plt.errorbar(1+z_input, D_H_Desi[0], yerr=D_H_Desi[1], fmt='o')
plt.errorbar(1+z_input, D_V_Desi[0], yerr=D_V_Desi[1], fmt='o')
plt.legend(loc='best')
plt.xlabel(r'$1+z$')
plt.xlim( 1.2,3.4) # 限制 x 軸的範圍從 0 到 10
plt.ylim( 1,42)
plt.xscale('log')
#plt.title(f'{array_str}')
plt.grid()
plt.savefig('error.png')
plt.show()
