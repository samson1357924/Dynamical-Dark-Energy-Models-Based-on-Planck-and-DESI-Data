# Dynamical Dark Energy Models Based on Planck and DESI Data

This repository implements a numerical framework to simulate cosmological evolution and optimize Dark Energy equation-of-state parameters ($w_\phi$) using observational constraints from DESI (Dark Energy Spectroscopic Instrument) and Planck 2018 datasets.

## 🔬 Project Overview

The core of this project is a Python-based solver that explores the parameter space of dynamical dark energy. By integrating the Friedmann equations and density evolution, the system calculates $\chi^2$ statistics against observational data to find the most probable cosmological model.

### Key Logic: Main Execution (`Main.py`)
The `Main.py` script serves as the primary entry point, integrating two distinct numerical phases to ensure physical consistency from the early universe to the present day.

1.  **Phase A: Forward Evolution (`backward_growth`)**
    - **Physical Basis**: Logic corresponding to `forward_evolution.py`.
    - **Direction**: Integrates background cosmological evolution from the **decoupling epoch ($z \approx 1060$) forward to the present day ($t=0$)**.
    - **Goal**: Calculates Hubble parameter $H(z)$, Angular Diameter Distance $D_A(z)$, and comoving volume distance $D_V(z)$ to evaluate $\chi^2$ against DESI and Planck data.
    - *Note: While named `backward_growth` in the code, its mathematical direction is forward-in-time from z=1060.*

2.  **Phase B: Backward Integration & Growth Analysis (`forward_evolution`)**
    - **Physical Basis**: Logic corresponding to `backward_growth.py`.
    - **Direction**: Integrates matter perturbation growth ($\delta_m$) and the background state from the **present day ($t=0$) backward into the early universe**.
    - **Optimization Role (`optimizer_dm1060.py`)**: This specialized script is used to find the most suitable $D_{m1060}$ value. It treats $D_{m1060}$ as an optimization parameter to ensure that the backward integration results in a growth factor $\theta_0 = 1$ at the present epoch. Once optimized, this value is passed into Phase B.
    - *Note: While named `forward_evolution` in the code, its mathematical direction is backward-in-time from t=0.*

### Dynamic Dark Energy Model ($w_\phi$)
- Uses a piecewise quadratic functional approach defined by control points `NP_az` (configured in `w_phi_a.py`).
- Leverages `sympy` for symbolic integration and `lambdify` for numerical execution.

## 🛠 Required Environment

- **Python 3.8+**
- **Libraries**: `numpy`, `scipy`, `sympy`, `pandas`, `matplotlib`, `openpyxl`.

Install via pip:
```bash
pip install numpy scipy sympy pandas matplotlib openpyxl
```

## 🚀 Execution Guide

1. **`w_phi_a.py`**: Configuration for redshift bins and symbolic $w_\phi$ generation.
2. **`Main.py`**: Executes the full integrated analysis (Phase A + Phase B) using best-fit parameters.
3. **`optimizer_dm1060.py`**: Specialized search for the $D_{m1060}$ parameter to satisfy the $\theta_0 = 1$ condition.
4. **`minimize.py`**: Performs the primary multi-parameter optimization for $w_\phi$ parameters.

---
**Author**: Chen, Hsiao-Hsuan (Samson)
**Academic Affiliation**: M.S. in Physics, National Taiwan Normal University (NTNU).
