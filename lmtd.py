import argparse
import math
import pandas as pd
from typing import Optional, Tuple

# --- Optional UA correlation hook (example: log-linear in m_h, m_c, A) ---
def ua_from_correlation(row, map_A, mh, mc, A_default=None, coeffs=None):
    """
    Predict UA [W/K] from a lightweight correlation.
    Example model: ln(UA) = b0 + b1 ln(m_h) + b2 ln(m_c) + b3 ln(A)
    Args:
        row: the pandas Series
        map_A: column name for area
        mh, mc: mass flow rates [kg/s]
        A_default: fallback area if map_A missing/NaN
        coeffs: dict with keys b0,b1,b2,b3 (floats). If None -> return None.
    """
    if coeffs is None:
        return None
    A = row_get_float(row, map_A) if map_A else None
    if A is None or not (A > 0):
        A = A_default
    if not (mh and mh > 0 and mc and mc > 0 and A and A > 0):
        return None
    import math
    b0 = coeffs.get("b0", 0.0); b1 = coeffs.get("b1", 1.0)
    b2 = coeffs.get("b2", 1.0); b3 = coeffs.get("b3", 1.0)
    return math.exp(b0 + b1*math.log(mh) + b2*math.log(mc) + b3*math.log(A))


def _to_float(val):
    """Robust numeric parser: handles None, '', whitespace, and comma decimals."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s == "":
        return None
    s = s.replace(",", ".")  # support European decimals
    try:
        return float(s)
    except Exception:
        return None

def coerce_numeric_columns(df: pd.DataFrame, cols):
    """In-place: coerce selected columns to numeric (with comma decimal support)."""
    for c in cols:
        if c is None or c not in df.columns:
            continue
        df[c] = df[c].map(_to_float)


# ---------- Thermo helpers ----------
def cp_water_constant(T_C: float) -> float:
    """
    Constant cp for water (J/kg-K). Use 4180 by default.
    Args:
        T_C: temperature in °C (ignored; kept for API consistency)
    """
    return 4180.0

def lmtd(deltaT1: float, deltaT2: float) -> float:
    """
    Log-mean temperature difference with safe limiting.
    Returns LMTD >= 0.
    """
    # Guard against tiny/negative differences due to rounding
    eps = 1e-9
    d1 = max(deltaT1, 0.0)
    d2 = max(deltaT2, 0.0)
    # If both zero -> no driving force
    if d1 <= eps and d2 <= eps:
        return 0.0
    # If nearly equal -> arithmetic mean
    if abs(d1 - d2) <= 1e-9:
        return 0.5 * (d1 + d2)
    # Standard LMTD
    # Ensure strictly positive to avoid log domain errors
    d1c = max(d1, eps)
    d2c = max(d2, eps)
    return (d1c - d2c) / math.log(d1c / d2c)

def lmtd_deltas(flow: str,
                Th_in: float, Th_out: float,
                Tc_in: float, Tc_out: float) -> Tuple[float, float]:
    """
    Compute the terminal temperature differences for given arrangement.
    """
    flow = flow.lower()
    if flow.startswith("counter"):
        # Countercurrent:
        # ΔT1 = Th_in - Tc_out; ΔT2 = Th_out - Tc_in
        return Th_in - Tc_out, Th_out - Tc_in
    elif flow.startswith("parallel"):
        # Parallel/cocurrent:
        # ΔT1 = Th_in - Tc_in; ΔT2 = Th_out - Tc_out
        return Th_in - Tc_in, Th_out - Tc_out
    else:
        raise ValueError(f"Unknown flow arrangement: {flow}")

def solve_outlets_lmtd(UA: float, Fcorr: float,
                       Th_in: float, Tc_in: float,
                       mh: float, mc: float,
                       cph: float, cpc: float,
                       flow: str) -> Tuple[float, float, float, float]:
    """
    Solve for Q, Th_out, Tc_out given UA, F, inlets, mass flow rates and cp's using LMTD.
    Returns: (Q [W], Th_out [°C], Tc_out [°C], eps_effectiveness [-])
    """
    Ch = mh * cph  # W/K
    Cc = mc * cpc  # W/K
    if Ch <= 0 or Cc <= 0 or UA <= 0:
        return (0.0, Th_in, Tc_in, 0.0)

    Cmin = min(Ch, Cc)

    # Theoretical upper bound for heat transfer (avoid pinch/log singularities)
    # Conservative margin ensures ΔT>0 in logs
    dT_in = max(Th_in - Tc_in, 0.0)
    if dT_in <= 0:
        return (0.0, Th_in, Tc_in, 0.0)

    Q_upper = 0.999999 * Cmin * dT_in  # W
    Q_lower = 0.0

    # Define residual f(Q) = Q - UA*F*LMTD(Th_in, Th_out(Q), Tc_in, Tc_out(Q))
    def residual(Q: float) -> float:
        Th_out = Th_in - Q / Ch
        Tc_out = Tc_in + Q / Cc
        dT1, dT2 = lmtd_deltas(flow, Th_in, Th_out, Tc_in, Tc_out)
        DTlm = lmtd(dT1, dT2)
        Q_pred = UA * Fcorr * DTlm
        return Q - Q_pred

    # If driving force is zero, quick-exit
    if UA * Fcorr <= 0:
        return (0.0, Th_in, Tc_in, 0.0)

    # Bisection on [Q_lower, Q_upper]
    fL = residual(Q_lower)
    fU = residual(Q_upper)

    # If residual already tiny at bounds, accept
    if abs(fL) < 1e-6:
        Q = Q_lower
    elif abs(fU) < 1e-6:
        Q = Q_upper
    else:
        # Ensure we have a bracket; if not, clamp to physically plausible limit
        # Typically f(0) = -UA*F*LMTD < 0; f(Q_upper) > 0, so bracket holds.
        max_iter = 80
        for _ in range(max_iter):
            Q_mid = 0.5 * (Q_lower + Q_upper)
            fM = residual(Q_mid)
            if abs(fM) < 1e-6 or (Q_upper - Q_lower) < 1e-6:
                Q = Q_mid
                break
            # Maintain bracket
            if fL * fM <= 0.0:
                Q_upper, fU = Q_mid, fM
            else:
                Q_lower, fL = Q_mid, fM
        else:
            Q = 0.5 * (Q_lower + Q_upper)

    # Compute outlets
    Th_out = Th_in - Q / Ch
    Tc_out = Tc_in + Q / Cc
    # Effectiveness (ε) for reference
    eps = Q / (Cmin * dT_in) if Cmin * dT_in > 0 else 0.0

    return Q, Th_out, Tc_out, eps

# ---------- I/O & CLI ----------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LMTD outlet temperature calculator (per-row).")
    p.add_argument("--data", required=True, help="Input CSV with performance data.")
    p.add_argument("--out", required=True, help="Output CSV path.")
    p.add_argument("--cp", choices=["constant", "coolprop"], default="constant",
                   help="How to compute cp (constant=water 4180 J/kg-K; coolprop hook if available).")
    p.add_argument("--cp-const-h", type=float, default=4180.0, help="Constant cp_hot [J/kg-K].")
    p.add_argument("--cp-const-c", type=float, default=4180.0, help="Constant cp_cold [J/kg-K].")
    p.add_argument("--flow", choices=["counter", "parallel"], default="counter",
                   help="Flow arrangement for LMTD.")
    # Column mappings
    p.add_argument("--map-A", default=None, help="Column name for area A [m2].")
    p.add_argument("--map-U", default=None, help="Column name for U [W/m2-K].")
    p.add_argument("--map-UA", default=None, help="Column name for UA [W/K].")
    p.add_argument("--map-F", default=None, help="Column name for LMTD correction factor F.")
    p.add_argument("--map-Thin", required=True, help="Column name for hot inlet temperature [°C].")
    p.add_argument("--map-Tcin", required=True, help="Column name for cold inlet temperature [°C].")
    p.add_argument("--map-mh", required=True, help="Column name for hot mass flow [kg/s].")
    p.add_argument("--map-mc", required=True, help="Column name for cold mass flow [kg/s].")
    # Optional measured outlets for comparison
    p.add_argument("--map-Thout", default=None, help="(Optional) Column for measured hot outlet [°C].")
    p.add_argument("--map-Tcout", default=None, help="(Optional) Column for measured cold outlet [°C].")
    # Fixed values if columns are absent
    p.add_argument("--U-fixed", type=float, default=None, help="Fixed U [W/m2-K] if no column.")
    p.add_argument("--UA-fixed", type=float, default=None, help="Fixed UA [W/K] if no column.")
    p.add_argument("--F-fixed", type=float, default=1.0, help="Fixed correction factor F (default 1.0).")
    return p.parse_args(argv)

def get_cp_func(mode: str, cp_const_h: float, cp_const_c: float):
    if mode == "coolprop":
        try:
            import CoolProp.CoolProp as CP  # noqa
            # You can customize here if you want temperature-dependent cp for water.
            # For now, just fall back to constant unless you implement full CoolProp calls.
            def cph(T_C: float) -> float:
                return cp_const_h
            def cpc(T_C: float) -> float:
                return cp_const_c
            return cph, cpc
        except Exception:
            # Fallback to constants
            pass

    def cph(T_C: float) -> float: return cp_const_h
    def cpc(T_C: float) -> float: return cp_const_c
    return cph, cpc

def row_get_float(row, col: Optional[str]) -> Optional[float]:
    if col is None:
        return None
    return _to_float(row.get(col, None))


def compute_UA_for_row(
    row,
    map_UA, map_U, map_A, UA_fixed, U_fixed,
    flow: str, Fcorr: float,
    Th_in: float, Tc_in: float, mh: float, mc: float,
    cph: float, cpc: float,
    map_Thout: Optional[str], map_Tcout: Optional[str]
) -> Tuple[Optional[float], str]:
    """
    Return (UA [W/K], UA_source).
    Priority:
      1) From measured outlets (if available): UA = Q / (F * ΔT_lm(measured))
      2) From explicit UA column
      3) From U and A columns (U*A)
      4) From --UA-fixed
      5) From --U-fixed and A column
    """
    # 1) Try to compute UA from measured outlets
    Th_meas = row_get_float(row, map_Thout) if map_Thout else None
    Tc_meas = row_get_float(row, map_Tcout) if map_Tcout else None

    if (Th_meas is not None) or (Tc_meas is not None):
        # Need both sides’ inlets plus at least one measured outlet.
        # Prefer averaging Q_h and Q_c when both are valid.
        Ch = mh * cph  # W/K
        Cc = mc * cpc  # W/K
        Qh = None
        Qc = None
        if (Th_meas is not None) and (Ch is not None) and (Ch > 0):
            Qh = Ch * (Th_in - Th_meas)
        if (Tc_meas is not None) and (Cc is not None) and (Cc > 0):
            Qc = Cc * (Tc_meas - Tc_in)

        # Decide Q (robust to small measurement noise)
        if (Qh is not None) and (Qc is not None):
            # If signs disagree badly, pick the one with larger magnitude; else average
            if Qh * Qc <= 0:
                Q = Qh if abs(Qh) >= abs(Qc) else Qc
            else:
                Q = 0.5 * (Qh + Qc)
        elif Qh is not None:
            Q = Qh
        elif Qc is not None:
            Q = Qc
        else:
            Q = None

        if (Q is not None) and (Fcorr is not None) and (Fcorr > 0.0):
            # Build ΔT1, ΔT2 using measured outlets (fill missing outlet with energy balance)
            if Th_meas is None:
                Th_meas = Th_in - Q / Ch
            if Tc_meas is None:
                Tc_meas = Tc_in + Q / Cc

            dT1, dT2 = lmtd_deltas(flow, Th_in, Th_meas, Tc_in, Tc_meas)
            DTlm = lmtd(dT1, dT2)
            if DTlm > 0:
                UA = Q / (Fcorr * DTlm)
                if (UA is not None) and (UA > 0) and math.isfinite(UA):
                    return UA, "from_measured"
            # If ΔTlm <= 0 or Q invalid: fall through to other sources

    # 2) Explicit UA column
    UA = row_get_float(row, map_UA)
    if UA is not None:
        return UA, "from_UA_col"

    # 3) U*A from columns
    U = row_get_float(row, map_U)
    A = row_get_float(row, map_A)
    if (U is not None) and (A is not None):
        return U * A, "from_U_times_A"

    # 4) Fixed UA
    if UA_fixed is not None:
        return UA_fixed, "from_UA_fixed"
    
    # 4b) From correlation (if provided via kwargs)
    coeffs = row.get("_UA_CORR_COEFFS_", None)  # set by driver script
    UA_corr = ua_from_correlation(
        row=row, map_A=map_A, mh=mh, mc=mc,
        A_default=None, coeffs=coeffs
    )
    if UA_corr is not None and UA_corr > 0:
        return UA_corr, "from_corr"


    # 5) U_fixed * A
    if (U_fixed is not None) and (A is not None):
        return U_fixed * A, "from_Ufixed_times_A"

    return None, "none"


def compute_F_for_row(row, map_F, F_fixed: float) -> float:
    F = row_get_float(row, map_F)
    if F is None:
        F = F_fixed
    # Safety: keep within (0,1]
    if not (0.0 < F <= 1.0):
        F = max(min(F, 1.0), 1e-6)
    return F

# ---------- Main ----------
def main(argv=None):
    # If run without args (e.g., Spyder Run button), supply defaults here:
    if argv is None:
        argv = [
            "--data",   "all_outdata.csv",
            "--cp",     "constant",
            "--out",    "results_lmtd.csv",
            "--map-A",   "area_m2",
            "--map-Thin","Tin1_C",
            "--map-Tcin","Tin2_C",
            "--map-mh",  "m1_kg_s",
            "--map-mc",  "m2_kg_s",
            "--map-Thout","Tout1_C",
            "--map-Tcout","Tout2_C",
            # Optional: if you have U or UA columns, add:
            # "--map-U",  "U_W_m2K",
            # "--map-UA", "UA_W_per_K",
            # "--map-F",  "F_corr",
            # Or supply fixed values:
            # "--U-fixed","2500",
            # "--UA-fixed","1200",
            # "--F-fixed","0.95",
            # Flow arrangement:
            # "--flow","counter",
        ]
    args = parse_args(argv)

    df = pd.read_csv(args.data)

    # Coerce all relevant columns to numeric (handles comma decimals & stray text)
    cols_to_numeric = [
        args.map_Thin, args.map_Tcin, args.map_mh, args.map_mc,
        args.map_A, args.map_U, args.map_UA, args.map_F,
        args.map_Thout, args.map_Tcout
    ]
    coerce_numeric_columns(df, cols_to_numeric)

    cph_func, cpc_func = get_cp_func(args.cp, args.cp_const_h, args.cp_const_c)

    results = []
    for idx, row in df.iterrows():
        Th_in = row_get_float(row, args.map_Thin)
        Tc_in = row_get_float(row, args.map_Tcin)
        mh    = row_get_float(row, args.map_mh)
        mc    = row_get_float(row, args.map_mc)
        Fcorr = compute_F_for_row(row, args.map_F, args.F_fixed)

        # cp (constant by default; hook ready for CoolProp)
        cph = cp_water_constant(Th_in) if args.cp == "constant" else get_cp_func(args.cp, args.cp_const_h, args.cp_const_c)[0](Th_in)
        cpc = cp_water_constant(Tc_in) if args.cp == "constant" else get_cp_func(args.cp, args.cp_const_h, args.cp_const_c)[1](Tc_in)

        UA, UA_source = compute_UA_for_row(
            row=row,
            map_UA=args.map_UA, map_U=args.map_U, map_A=args.map_A,
            UA_fixed=args.UA_fixed, U_fixed=args.U_fixed,
            flow=args.flow, Fcorr=Fcorr,
            Th_in=Th_in, Tc_in=Tc_in, mh=mh, mc=mc, cph=cph, cpc=cpc,
            map_Thout=args.map_Thout, map_Tcout=args.map_Tcout
        )

        # Basic input checks
        ok = all(v is not None for v in [Th_in, Tc_in, mh, mc, UA])
        if not ok:
            results.append({
                "index": idx,
                "status": "skip_missing_inputs",
                "UA_W_per_K": UA,
                "UA_source": UA_source,
                "Th_in_C": Th_in, "Tc_in_C": Tc_in,
                "mh_kg_s": mh, "mc_kg_s": mc
            })
            continue

        Q, Th_out_pred, Tc_out_pred, eps = solve_outlets_lmtd(
            UA=UA, Fcorr=Fcorr,
            Th_in=Th_in, Tc_in=Tc_in,
            mh=mh, mc=mc,
            cph=cph, cpc=cpc,
            flow=args.flow
        )

        rec = {
            "index": idx,
            "status": "ok",
            "UA_W_per_K": UA,
            "UA_source": UA_source,
            "F_corr": Fcorr,
            "Th_in_C": Th_in, "Tc_in_C": Tc_in,
            "mh_kg_s": mh, "mc_kg_s": mc,
            "cp_h_J_kgK": cph, "cp_c_J_kgK": cpc,
            "Q_W": Q,
            "Th_out_lmtd_C": Th_out_pred,
            "Tc_out_lmtd_C": Tc_out_pred,
            "epsilon": eps,
        }

        # If measured outlets exist, compute residuals
        if args.map_Thout:
            Th_meas = row_get_float(row, args.map_Thout)
            if Th_meas is not None:
                rec["Th_out_meas_C"] = Th_meas
                rec["dTh_out_C"] = Th_out_pred - Th_meas
        if args.map_Tcout:
            Tc_meas = row_get_float(row, args.map_Tcout)
            if Tc_meas is not None:
                rec["Tc_out_meas_C"] = Tc_meas
                rec["dTc_out_C"] = Tc_out_pred - Tc_meas

        results.append(rec)


    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out, index=False)
    print(f"[LMTD] Wrote {len(out_df)} rows to: {args.out}")
    
    # ---------- Verification of model performance ----------
    if args.map_Thout or args.map_Tcout:
        # Compute performance metrics if both measured and predicted exist
        hot_mask = out_df["status"] == "ok"
        cold_mask = out_df["status"] == "ok"
    
        # Initialize counters
        N_h = N_c = 0
        mae_h = mae_c = 0.0
        rmse_h = rmse_c = 0.0
    
        if "dTh_out_C" in out_df.columns:
            diffs_h = out_df.loc[hot_mask & out_df["dTh_out_C"].notna(), "dTh_out_C"]
            N_h = len(diffs_h)
            if N_h > 0:
                mae_h = (diffs_h.abs()).mean()
                rmse_h = (diffs_h ** 2).mean() ** 0.5
    
        if "dTc_out_C" in out_df.columns:
            diffs_c = out_df.loc[cold_mask & out_df["dTc_out_C"].notna(), "dTc_out_C"]
            N_c = len(diffs_c)
            if N_c > 0:
                mae_c = (diffs_c.abs()).mean()
                rmse_c = (diffs_c ** 2).mean() ** 0.5

        print(f"Hot outlet:  MAE={mae_h:.3f} K, RMSE={rmse_h:.3f} K, N={N_h}")
        print(f"Cold outlet: MAE={mae_c:.3f} K, RMSE={rmse_c:.3f} K, N={N_c}")


if __name__ == "__main__":
    main()
