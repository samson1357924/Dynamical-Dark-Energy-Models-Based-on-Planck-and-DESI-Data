# Dynamical Dark Energy Models Based on Planck and DESI Data

This repository implements a numerical framework to simulate cosmological evolution and optimize Dark Energy equation-of-state parameters ($w_\phi$) using observational constraints from DESI (Dark Energy Spectroscopic Instrument) and Planck 2018 datasets.

## 🔬 Project Overview

The core of this project is a Python-based solver that explores the parameter space of dynamical dark energy. By integrating the Friedmann equations and density evolution, the system calculates $\chi^2$ statistics against observational data to find the most probable cosmological model.

### Key Logic: A+B Merged System
The primary analysis script (e.g., `A+B.py`) represents the integration of two critical numerical phases:

1.  **Phase A (Forward Evolution - `from_decouple.py`)**: 
    - **Logic**: Implemented in `run_program_A`.
    - **Direction**: Integrates the background cosmological evolution from the **decoupling epoch ($z \approx 1060$) forward to the present day ($t=0$)**.
    - **Functions**: Computes Hubble parameter $H(z)$, Angular Diameter Distance $D_A(z)$, and comoving volume distance $D_V(z)$.
    - **Data Comparison**: Evaluates $\chi^2$ against **DESI BAO** and **Planck 2018** acoustic angular scale $\theta_*$.

2.  **Phase B (Backward Evolution & Growth Analysis - `reverse_compare_...py`)**:
    - **Logic**: Implemented in `run_program_B`.
    - **Direction**: Integrates the matter perturbation growth ($\delta_m$) and background state from the **present day ($t=0$) backward to the early universe**.
    - **Constraint Logic (`constraint.py`)**: This script is used to find the most suitable $D_{m1060}$ value. It treats $D_{m1060}$ as an optimization parameter to ensure that the backward integration results in a growth factor $\theta_0 = 1$ at the present epoch.
    - **Integration**: The optimized $D_{m1060}$ is then used in Phase B to maintain physical consistency across the growth history.

### Dynamic Dark Energy Model ($w_\phi$)
- Uses a piecewise quadratic functional approach defined by control points `NP_az` (implemented in `w_phi_a.py`).
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
2. **`A+B.py`**: Executes the full analysis pipeline (Phase A + Phase B) using current best-fit parameters.
3. **`constraint.py`**: Specialized script for optimizing the $D_{m1060}$ parameter to satisfy the $\theta_0 = 1$ condition.
4. **`minimize.py`**: Performs multi-parameter optimization for $w_\phi$ using the `SLSQP` algorithm.

---
**Author**: Chen, Hsiao-Hsuan (Samson)
**Academic Affiliation**: M.S. in Physics, National Taiwan Normal University (NTNU).
