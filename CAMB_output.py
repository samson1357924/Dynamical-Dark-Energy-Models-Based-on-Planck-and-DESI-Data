#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 06:02:21 2025

@author: hsiaohsuanchen
"""

import numpy as np
import matplotlib.pyplot as plt

# --- 讀取物質功率譜 P(k) ---
# 假設檔名是 output/camb_model_py_matterpower.dat
# CAMB 檔案通常有註解行，skiprows=1 可能不夠，需要檢查
# 或者使用更穩健的方式，例如讀取所有行，找到第一個非 '#' 的行開始
try:
    # 嘗試自動跳過註解，指定分隔符為空白
    data_pk = np.loadtxt('output/camb_model_py_3_matterpower.dat')
    k = data_pk[:, 0]  # 第一列是 k (h/Mpc)
    Pk = data_pk[:, 1]  # 第二列是 P(k) ((Mpc/h)^3)

    # 繪圖 (log-log scale)
    plt.figure(figsize=(8, 6))
    plt.loglog(k, Pk)
    plt.xlabel(r'$k \quad [h\,\mathrm{Mpc}^{-1}]$')
    plt.ylabel(r'$P(k) \quad [(h^{-1}\,\mathrm{Mpc})^3]$')
    plt.title('Matter Power Spectrum from CAMB')
    plt.grid(True, which='both', linestyle=':')
    plt.show()

except Exception as e:
    print(f"Error reading P(k) file: {e}")
    print("Please check the file path and format.")


# --- 讀取 CMB TT 功率譜 C_l ---
# 假設檔名是 output/camb_model_py_scalCls.dat
try:
    # 欄位通常是 L, TT, EE, BB, TE ... 檢查檔案頭部註解！
    data_cl = np.loadtxt('output/camb_model_py_3_scalCls.dat')
    ell = data_cl[:, 0]    # L
    cl_tt = data_cl[:, 1]  # TT C_l
    # cl_ee = data_cl[:, 2]  # EE C_l
    # cl_te = data_cl[:, 4]  # TE C_l

    # 計算並繪製 D_l = l(l+1)C_l / (2*pi)
    dl_tt = ell * (ell + 1) * cl_tt / (2 * np.pi)

    plt.figure(figsize=(8, 6))
    plt.semilogx(ell, dl_tt) # 通常 l 軸取對數
    # 或者 plt.loglog(ell, dl_tt) 如果 dl_tt 範圍也很大
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$D_\ell^{TT} = \ell(\ell+1)C_\ell^{TT}/(2\pi) \quad [\mu K^2]$')
    plt.title('CMB Temperature Power Spectrum (TT)')
    plt.grid(True, which='both', linestyle=':')
    # 可能需要設定 l 的範圍，例如從 l=2 開始
    plt.xlim(2, 2500)
    plt.show()

except Exception as e:
    print(f"Error reading C_l file: {e}")
    print("Please check the file path and format.")

# --- 讀取衍生參數 (例如 sigma8) ---
# 需要讀取 *_params.ini 檔案，它不是簡單的 .dat 格式
# 可以手動查看，或用 Python 的 configparser 模組，或簡單的文本處理
try:
    sigma8 = np.nan # Placeholder
    with open('output/camb_model_py_3_params.ini', 'r') as f:
        for line in f:
            # 移除註解和空白
            line = line.split('#')[0].strip()
            if line.startswith('sigma8'):
                parts = line.split('=')
                if len(parts) == 2:
                    sigma8 = float(parts[1].strip())
                    break # 找到就停止
    if not np.isnan(sigma8):
        print(f"Found sigma8 = {sigma8}")
    else:
        print("Could not find sigma8 in _params.ini file.")
except FileNotFoundError:
    print("Error: output/camb_model_py_3_params.ini not found.")
except Exception as e:
    print(f"Error reading params file: {e}")