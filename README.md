# DH-HeatExchanger
DH-HeatExchanger is a lightweight, reproducible toolkit for steady-state thermal prediction of plate heat exchangers (PHEs) operating with single-phase water.

It includes three complementary methods:
- ETL-CF (ε–θ) — A UA-independent model expressing effectiveness as a function of thermal length r=A/Cmin and capacity-rate ratio C*.
- ε–NTU — Classical effectiveness method (UA-based).
- LMTD — Log-mean temperature difference method (UA-based, robust implicit solver).

The library is designed for district heating and building heating applications: fast, deterministic, and transparent, suitable for selection tools, batch studies, and model-based control/digital-twin pipelines.

## Features
- UA-independent prediction (ETL-CF): return temperatures and effectiveness without proprietary UA data.
- Classical baselines (ε–NTU & LMTD): identical I/O conventions for apples-to-apples benchmarking.
- Coefficient I/O: load ε–θ coefficients α(C*), β(C*) from CSV.
- Pure SI units and constant-cp water (optional property hooks can be added).
- No heavy dependencies: NumPy/Pandas/Matplotlib only for your own analysis/plots.

## Repository Structure
```
.
├── etl_cf_model.py                 # ETL-CF (ε–θ) core: effectiveness, rating, sizing
├── etl_cf1_coeffs_by_cstar.csv     # Example ε–θ coefficients α(C*)[, β(C*)]
├── e_NTU.py                        # ε–NTU solver (UA-based), CSV workflow
├── lmtd.py                         # LMTD solver (UA-based), CSV workflow
└── example_script.py               # Minimal ETL-CF quickstart (extended below)
```

## Quickstart
The snippet below extends `example_script.py` to compute all three methods at a single operating point.
- ETL-CF is UA-independent.
- ε–NTU and LMTD require UA; here we use a simple guess U_guess*A_m2. Replace with your correlation or identified UA when available.

## License
You are free to use, modify and distribute the code as long as **authorship is properly acknowledged**. Please reference this repository in derivative works.

## Citing
Tol, Hİ. Development of a U-Independent Effectiveness–Thermal Length Modelling for Heat Exchanger Performance Prediction. International Journal of Energy Horizon. Submitted. 

## Acknowledgements
Above all, I give thanks to **Allah, The Creator (C.C.)**, and honor His name **Al-‘Alīm (The All-Knowing)**.

This repository is lovingly dedicated to my parents who have passed away, in remembrance of their guidance and support.

I would also like to thank **ChatGPT (by OpenAI)** for providing valuable support in updating and improving the Python implementation.
