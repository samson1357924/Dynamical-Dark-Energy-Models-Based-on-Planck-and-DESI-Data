# Dynamical Dark Energy Models Based on Planck and DESI Data

This repository implements a numerical framework to simulate cosmological evolution and optimize Dark Energy equation-of-state parameters ($w_\phi$) using observational constraints from DESI (Dark Energy Spectroscopic Instrument) and Planck 2018 datasets.

## 🔬 Project Overview

The core of this project is a Python-based solver that explores the parameter space of dynamical dark energy. By integrating the Friedmann equations and density evolution, the system calculates $\chi^2$ statistics against observational data to find the most probable cosmological model.

### 🚀 Key Components & Simulation Logic

The analysis is built upon two complementary numerical approaches:

1.  **Backward Integration (`backward_growth.py`)**
    - **Logic**: Implemented in the `backward_growth` function.
    - **Direction**: Integrates from the **present day ($t=0$) backward to the early universe ($z \to \infty$)**.
    - **Initial Conditions**: Uses current density parameters ($\Omega_{m0}, \Omega_{r0}, \Omega_{\phi0}$) as $y_0$.
    - **Purpose**: Derives the past states of the universe from current observations.

2.  **Forward Evolution (`forward_evolution.py`)**
    - **Logic**: Implemented in the `forward_evolution` function.
    - **Direction**: Integrates from the **decoupling epoch ($z \approx 1060$) forward to the present day ($t=0, z=0$)**.
    - **Initial Conditions**: Starts from the state at $z=1060$ (`status_1060`).
    - **Purpose**: Simulates the growth and evolution history leading up to the current epoch.

3.  **Optimization & Constraints (`optimizer_dm1060.py`)**
    - **Constraint Logic**: Finds the most suitable $D_{m1060}$ value.
    - **Goal**: Ensures that the growth factor resulting from the evolution satisfies the physical observation $\theta_0 = 1$ at the present day.

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

---
**Author**: Chen, Hsiao-Hsuan (Samson)
**Academic Affiliation**: M.S. in Physics, National Taiwan Normal University (NTNU).
