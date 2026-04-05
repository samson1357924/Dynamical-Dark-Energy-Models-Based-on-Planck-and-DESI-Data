# Dynamical Dark Energy Models Based on Planck and DESI Data

This repository implements a numerical framework to simulate cosmological evolution and optimize Dark Energy equation-of-state parameters ($w_\phi$) using observational constraints from DESI (Dark Energy Spectroscopic Instrument) and Planck 2018 datasets.

## 🔬 Project Overview

The core of this project is a Python-based solver that explores the parameter space of dynamical dark energy. By integrating the Friedmann equations and density evolution, the system calculates $\chi^2$ statistics against observational data to find the most probable cosmological model.

### Key Logic: A+B Merged System
The primary analysis script (e.g., `A+B.py` or `merge.py`) represents the integration of two critical numerical phases:

1.  **Phase A (Forward Integration - `from_decouple`)**: 
    - Integrates the background evolution from the decoupling epoch ($z \approx 1060$) to the present day ($z=0$).
    - Computes observational quantities: Hubble parameter $H(z)$, Angular Diameter Distance $D_A(z)$, and comoving volume distance $D_V(z)$.
    - Compares results against **DESI BAO** measurements across 7 redshift bins and the **Planck 2018** acoustic angular scale $\theta_*$.

2.  **Phase B (Backward Integration - `reverse_compare`)**:
    - Performed to analyze the matter perturbation growth rate ($\delta_m$).
    - **$D_{m1060}$ Parameter**: This specific parameter is treated as a constraint. It is adjusted such that the final growth factor $\delta_m$ matches the current observed normalization (effectively ensuring the growth history is physically consistent with $\theta=1$ at $z=0$).

### Dynamic Dark Energy Model ($w_\phi$)
- Uses a piecewise quadratic functional approach defined by control points `NP_az`.
- Leverages `sympy` for symbolic integration and `lambdify` for high-performance numerical execution.

## 🛠 Required Environment

- **Python 3.8+**
- **Libraries**:
    - `numpy`: Numerical array operations.
    - `scipy`: ODE integration (`odeint`) and optimization (`minimize`).
    - `sympy`: Symbolic mathematics.
    - `pandas`: Data export to Excel.
    - `matplotlib`: Plotting cosmological evolution and $\chi^2$ contours.

Install all requirements via:
```bash
pip install numpy scipy sympy pandas matplotlib openpyxl
```

## 🚀 Execution Guide

### 1. Model Setup
The control points for the redshift bins and $w_\phi$ values are stored in `w_phi_a.py`. This file pre-generates the functional forms required for the ODE solver.

### 2. Running the Main Analysis
To execute the merged A+B logic and calculate the total $\chi^2$:
```bash
python A+B.py
```

### 3. Optimization
The `minimize.py` script utilizes the `SLSQP` algorithm to find the best-fit values for the dark energy EoS parameters while respecting the constraints defined in the A+B framework.

## 📊 Outputs
- **Plots**: Evolution of $\Omega_m, \Omega_r, \Omega_\phi$, and $w_\phi$ over time ($1+z$).
- **Excel**: Detailed $\chi^2$ breakdowns and distance measurements are exported to `output.xlsx`.

---
**Author**: Chen, Hsiao-Hsuan (Samson)
**Academic Affiliation**: M.S. in Physics, National Taiwan Normal University (NTNU).
