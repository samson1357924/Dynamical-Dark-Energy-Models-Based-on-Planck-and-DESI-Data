# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:09:06 2024

@author: samso
"""

import matplotlib.pyplot as plt
import numpy as np
from sympy import *
y = Symbol('y')

f = Function('f')
a = Symbol('a') 
b = Symbol('b')
a1 = Symbol('a1')

NP_az = np.array([[ 1.00000000e+00,  8.31053215e-01,  6.19908704e-01,  4.68416235e-01,  1.58079894e-04,  1.01624808e-04],
                  [-2.99980901e+00,  5.48678742e-01, -1.97731811e+00, -6.49292166e-01, -1.35553660e+00, -1.00000002e+00]])

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

