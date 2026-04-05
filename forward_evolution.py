# phys: forward evolution from decoupling (z~1060) to the present day
"""
Created on Fri Jan  3 04:09:22 2025

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

functions = []
for i in range(0, len(NP_az_rev[0])-1):
    f = lambdify(a, w_phi_a.function_results[f"function_{i}"])
    functions.append(f)
functions_rev = []
for i in range(0, len(NP_az_rev[0])-1):
    f = lambdify(a, w_phi_a.function_results_rev[f"function_{i}"])
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
y0 = np.array([1057035591.0248578, 415113098.8697977, 0.014353569403219258, 16277.37173930573/h_0, 1/1061,1.6890398288367673e+37, 10, 0, 0.0024727484505241547, 0])  # 這裡假設初始值為0，您可以根據需要進行修改

# 時間點
eta = np.linspace(0, 3.5, 1000000)  # 這裡假設 eta 從 0 到 10，您可以根據需要進行修改

# 解微分方程
sol = odeint(model, y0, eta)

# 過濾掉 sol[:, 4] < 0 的數據
valid_indices = sol[:, 4] <= 1
sol = sol[valid_indices]
eta = eta[valid_indices]
#print("sol[:, 8] 的最后一个值:", sol[-1, 8])



z = 1/sol[:, 4] -1
rho_c = 3*sol[:, 3]**2
w_phi = []

for a in sol[:, 4]: # 對 A 中的每個元素 x
    if a > NP_az[0, len(NP_az_rev[0])-1]:
        for x in range(0, len(NP_az_rev[0])-1):
            if  a >= NP_az[0, x + 1]:
                w_phi.append(functions[x](a))
                break
    else:
        w_phi.append(-1)
w_phi_np = np.array(w_phi)

q = sol[:, 1]/rho_c + 1/2*sol[:, 0]/rho_c + 1/2*sol[:, 2]/rho_c*(1 +3*w_phi_np)
Omega_m = sol[:, 0]/rho_c
Omega_r = sol[:, 1]/rho_c
Omega_phi = sol[:, 2]/rho_c
Omega_b = sol[:, 0]/rho_c/0.1424 * 0.02242
Omega_b_0 = rho_m0/rho_c0/0.1424 * 0.02242

print(sol[9][-1])
# 計算gamma
gamma = (np.log(sol[-1, 8]) - np.log(sol[-2, 8]))/(np.log(sol[-1, 4]) - np.log(sol[-2, 4]))
gamma = np.log(gamma)/np.log(Omega_m[-1])
chi_squared_gamma = ((0.633-gamma)/0.024)**2
print("gamma = ",gamma,chi_squared_gamma)


# 創建一個空的陣列 C
w_phi_avg_reversed = np.zeros(10000)
# 使用[::-1]索引來顛倒 A
Omega_phi_reversed = Omega_phi[::-1]
w_phi_np_reversed = w_phi_np[::-1]


# 對於每一個索引 i，計算 A 和 B 的前 i 個元素的內積，然後除以 B 的前 i 個元素的總和
for i in range(1, 10000):
    w_phi_avg_reversed[i] = np.dot(w_phi_np_reversed[:i],Omega_phi_reversed[:i]) / np.sum(Omega_phi_reversed[:i])
w_phi_avg = w_phi_avg_reversed[::-1]

# 對於每一個索引 i，計算 A 和 B 的前 i 個元素的內積，然後除以 B 的前 i 個元素的總和
for i in range(1, 10000):
    w_phi_avg_reversed[i] = np.dot(w_phi_np_reversed[:i],Omega_phi_reversed[:i]) / np.sum(Omega_phi_reversed[:i])
w_phi_avg = w_phi_avg_reversed[::-1]
# 繪製結果
plt.figure(figsize=(10, 8))
plt.plot(1+z, sol[:, 0]/rho_c, label=r'$\Omega_m$')
plt.plot(1+z, sol[:, 1]/rho_c, label=r'$\Omega_r$')
plt.plot(1+z, sol[:, 2]/rho_c, label=r'$\Omega_\phi$')
#plt.plot(z, sol[:, 3], label=r'$H$')
#plt.plot(z, sol[:, 4], label='a')
plt.plot(1+z, w_phi, label=r'$w_\phi$')
plt.plot(1+z, q, label='q')
#plt.plot(1+z, sol[:, 5]*H_0, label=r'$H_0t$')
plt.plot(1+z, sol[:, 8], label=r'$\delta$')
plt.plot(1+z, sol[:, 9], label=r'$d\delta$')
#plt.plot(1+z, w_phi_avg, label=r'$\langle w_\phi \rangle$')
#plt.plot(eta, rho_c/rho_c0, label='rho_c')
plt.legend(loc='best')
plt.xlabel(r'$1+z$')
plt.ylim(-3, 2)
plt.xlim(1061, 1) # 限制 x 軸的範圍從 0 到 10
plt.xscale('log')
plt.grid()
plt.show()

plt.figure(figsize=(10, 8))
plt.plot(1+z, sol[:, 8], label=r'$\delta$')
plt.plot(1+z, sol[:, 9], label=r'$d\delta$')
#plt.plot(1+z, w_phi_avg, label=r'$\langle w_\phi \rangle$')
#plt.plot(eta, rho_c/rho_c0, label='rho_c')
plt.legend(loc='best')
plt.xlabel(r'$1+z$')
#plt.ylim(-3, 2)
plt.xlim(1061, 1) # 限制 x 軸的範圍從 0 到 10
plt.xscale('log')
plt.grid()
plt.show()