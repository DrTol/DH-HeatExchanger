from __future__ import annotations

# -*- coding: utf-8 -*-
"""
Developed on Sat Sep 27 23:01:39 2025

@author: Dr. Hakan İbrahim Tol
"""

"""
ETL-CF heat exchanger rating (primary return temperature).

Physics:
    ε(r, C*) = (1 - exp[- α(C*) (1 - C*) r^β(C*)]) / (1 - C* exp[- α(C*) (1 - C*) r^β(C*)])
with r = A / Cmin,  C* = Cmin / Cmax,
Cmin = min(m_hot*cp_hot, m_cold*cp_cold), Cmax = max(...)

Units:
    A: m²
    m_hot, m_cold: kg/s
    cp: J/(kg·K)
    α: W/(m²·K)  (effective overall conductance scale)
    r = A/Cmin: m²·K/W (so α·r is dimensionless)
"""


from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------- Properties -------------------------

def cp_water_const(T_C: np.ndarray | float) -> np.ndarray:
    """Return constant cp for liquid water over 0–100 °C (J/kg-K)."""
    arr = np.asarray(T_C, dtype=float)
    return np.full_like(arr, 4180.0, dtype=float)

# (Optional) Hook for CoolProp if you decide to use it later.
def cp_water(T_C: np.ndarray | float, method: str = "constant") -> np.ndarray:
    """
    Heat capacity of liquid water (J/kg-K).
    method="constant"  : 4180 J/kg-K
    method="coolprop"  : use CoolProp if available (IAPWS-IF97), else fallback
    """
    if method.lower() != "coolprop":
        return cp_water_const(T_C)
    try:
        from CoolProp.CoolProp import PropsSI
        T_K = np.asarray(T_C, dtype=float) + 273.15
        cp = np.array([PropsSI("C", "T", Tk, "P", 1e5, "Water") for Tk in np.atleast_1d(T_K)], dtype=float)
        return cp.reshape(np.shape(T_K))
    except Exception:
        return cp_water_const(T_C)


# ------------------------- Coefficients -------------------------

@dataclass
class ETLCFCoeffs:
    """
    Storage + interpolation for ETL-CF coefficients vs C*.
    Accepts files created by previous fits:
      - etl_cf1_coeffs_by_cstar.csv  (columns: cstar_mid, alpha, ...)
      - etl_cfb_coeffs_by_cstar.csv  (columns: cstar_mid, alpha, beta, ...)
    """
    cstar_mid: np.ndarray  # shape (n,)
    alpha: np.ndarray      # shape (n,)
    beta: Optional[np.ndarray] = None  # shape (n,) or None (→ use β=1)

    @classmethod
    def from_csv(cls, path: str | Path) -> "ETLCFCoeffs":
        df = pd.read_csv(path)
        # tolerate variant column names
        cm = df.get("cstar_mid", df.get("cstar_bin_mid", None))
        if cm is None:
            raise ValueError("CSV must contain cstar_mid (or cstar_bin_mid).")
        a = df["alpha"].to_numpy(dtype=float)
        b = df["beta"].to_numpy(dtype=float) if "beta" in df.columns else None
        return cls(cstar_mid=np.asarray(cm, dtype=float), alpha=a, beta=b)

    def _interp(self, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
        xs = np.asarray(self.cstar_mid, dtype=float)
        ys = np.asarray(y, dtype=float)
        xq = np.asarray(xq, dtype=float)
        if xs.size == 0:
            return np.full_like(xq, np.nan)
        if xs.size == 1:
            return np.full_like(xq, ys[0])
        order = np.argsort(xs)
        return np.interp(xq, xs[order], ys[order], left=ys[order][0], right=ys[order][-1])

    def alpha_at(self, cstar: float | np.ndarray) -> np.ndarray:
        return self._interp(self.alpha, np.asarray(cstar, dtype=float))

    def beta_at(self, cstar: float | np.ndarray) -> np.ndarray:
        if self.beta is None:
            return np.ones_like(np.asarray(cstar, dtype=float))
        return self._interp(self.beta, np.asarray(cstar, dtype=float))


# ------------------------- Core ETL-CF relations -------------------------

def effectiveness_etl_cf(r: np.ndarray | float,
                         cstar: np.ndarray | float,
                         alpha: np.ndarray | float,
                         beta: np.ndarray | float = 1.0) -> np.ndarray:
    """
    ε = (1 - exp[- α (1-C*) r^β]) / (1 - C* exp[- α (1-C*) r^β])
    """
    r = np.asarray(r, dtype=float)
    cstar = np.asarray(cstar, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    t = alpha * (1.0 - cstar) * np.power(r, beta)
    e = np.exp(-t)
    return (1.0 - e) / (1.0 - cstar * e)


def invert_for_r_from_eps(eps: float,
                          cstar: float,
                          alpha: float,
                          beta: float = 1.0) -> float:
    """
    Analytic inverse for r (thermal length) given ε (0<ε<1), for ETL-CF(β).
    For β=1:
        r = [ ln((1 - ε C*)/(1 - ε)) ] / [ α (1 - C*) ]
    For β≠1:
        r = { [ ln((1 - ε C*)/(1 - ε)) ] / [ α (1 - C*) ] }^(1/β)
    """
    if not (0.0 < eps < 1.0):
        raise ValueError("eps must be in (0,1) for inversion.")
    num = np.log((1.0 - eps * cstar) / (1.0 - eps))
    denom = alpha * (1.0 - cstar)
    if denom <= 0:
        raise ValueError("alpha must be > 0 and C* < 1 for inversion.")
    if beta == 1.0:
        return float(num / denom)
    return float(np.power(num / denom, 1.0 / beta))


# ------------------------- Rating: outlet temperatures -------------------------

@dataclass
class RatingResult:
    Th_out: float
    Tc_out: float
    epsilon: float
    q_W: float
    Cmin: float
    Cmax: float
    Cstar: float
    r: float
    iterations: int


def primary_return_temperature(
    A_m2: float,
    Th_in_C: float,
    Tc_in_C: float,
    m_hot_kg_s: float,
    m_cold_kg_s: float,
    coeffs: ETLCFCoeffs,
    cp_method: str = "constant",
    tol_K: float = 1e-3,
    max_iter: int = 20,
) -> RatingResult:
    """
    Compute primary (hot) outlet temperature using ETL-CF model.

    Parameters
    ----------
    A_m2 : plate area [m²]
    Th_in_C, Tc_in_C : inlet temperatures [°C], assume Th_in > Tc_in
    m_hot_kg_s, m_cold_kg_s : mass flow rates [kg/s]
    coeffs : ETLCFCoeffs  (contains α(C*) and optionally β(C*))
    cp_method : "constant" | "coolprop"
    tol_K : convergence tolerance for temperatures [K]
    max_iter : maximum iterations for cp(T) update

    Returns
    -------
    RatingResult with Th_out, Tc_out, ε, q, Cmin, Cmax, C*, r, iterations
    """
    if Th_in_C <= Tc_in_C:
        raise ValueError("Expected Th_in_C > Tc_in_C for the primary (hot) side.")

    # Initial cp guess
    cp_h = float(cp_water(Th_in_C, method=cp_method))
    cp_c = float(cp_water(Tc_in_C, method=cp_method))

    Th_out = Th_in_C  # placeholders for iteration
    Tc_out = Tc_in_C

    for it in range(1, max_iter + 1):
        Ch = m_hot_kg_s * cp_h
        Cc = m_cold_kg_s * cp_c
        Cmin = min(Ch, Cc)
        Cmax = max(Ch, Cc)
        Cstar = Cmin / Cmax
        r = A_m2 / Cmin  # NTU/U

        # Interpolate α and β at this C*
        alpha = float(coeffs.alpha_at(Cstar))
        beta = float(coeffs.beta_at(Cstar))

        # Effectiveness and duty
        eps = float(effectiveness_etl_cf(r, Cstar, alpha, beta))
        Qmax = Cmin * (Th_in_C - Tc_in_C)
        qW = eps * Qmax  # W

        # New outlets
        Th_new = Th_in_C - qW / Ch
        Tc_new = Tc_in_C + qW / Cc

        # Update cp with mean-film temperatures
        Th_mean = 0.5 * (Th_in_C + Th_new)
        Tc_mean = 0.5 * (Tc_in_C + Tc_new)
        cp_h_new = float(cp_water(Th_mean, method=cp_method))
        cp_c_new = float(cp_water(Tc_mean, method=cp_method))

        # Convergence check on temperatures
        if max(abs(Th_new - Th_out), abs(Tc_new - Tc_out)) < tol_K:
            Th_out, Tc_out = Th_new, Tc_new
            return RatingResult(Th_out, Tc_out, eps, qW, Cmin, Cmax, Cstar, r, it)

        # Iterate
        Th_out, Tc_out = Th_new, Tc_new
        cp_h, cp_c = cp_h_new, cp_c_new

    # If not converged, still return last iterate
    return RatingResult(Th_out, Tc_out, eps, qW, Cmin, Cmax, Cstar, r, max_iter)


# ------------------------- Convenience: sizing inverse -------------------------

def area_for_target_effectiveness(
    eps_target: float,
    Th_in_C: float,
    Tc_in_C: float,
    m_hot_kg_s: float,
    m_cold_kg_s: float,
    coeffs: ETLCFCoeffs,
    cp_method: str = "constant",
) -> Tuple[float, float, float]:
    """
    Given a target effectiveness and operating point, compute required area A.

    Returns
    -------
    (A_m2, r, Cstar)
    """
    # single cp evaluate (small error; you can add an outer iteration if desired)
    cp_h = float(cp_water(0.5 * (Th_in_C + Tc_in_C), method=cp_method))
    cp_c = cp_h  # symmetric for water if band is small; adjust if needed
    Ch = m_hot_kg_s * cp_h
    Cc = m_cold_kg_s * cp_c
    Cmin = min(Ch, Cc)
    Cmax = max(Ch, Cc)
    Cstar = Cmin / Cmax
    alpha = float(coeffs.alpha_at(Cstar))
    beta = float(coeffs.beta_at(Cstar))
    r = invert_for_r_from_eps(eps_target, Cstar, alpha, beta)
    A = r * Cmin
    return A, r, Cstar
