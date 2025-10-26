# quickstart_all_methods.py
# Run: python quickstart_all_methods.py

from etl_cf_model import ETLCFCoeffs, primary_return_temperature
from e_NTU import solve_outlets_entu
from lmtd import solve_outlets_lmtd

# --- 1) Load ε–θ coefficients (α(C*)[, β(C*)]) ---
coeffs = ETLCFCoeffs.from_csv("etl_cf1_coeffs_by_cstar.csv")

# --- 2) Operating point (SI units) ---
A_m2   = 0.414       # thermal area [m²]
Th_in  = 70.0        # hot inlet [°C]
Tc_in  = 30.0        # cold inlet [°C]
m_hot  = 0.60        # hot mass flow [kg/s]
m_cold = 0.50        # cold mass flow [kg/s]
cph = cpc = 4180.0   # water cp [J/kg-K]
Fcorr  = 1.0         # correction factor (single-pass counter-current)
flow   = "counter"

# --- 3) ETL-CF (ε–θ): UA-independent prediction ---
res = primary_return_temperature(
    A_m2=A_m2, Th_in_C=Th_in, Tc_in_C=Tc_in,
    m_hot_kg_s=m_hot, m_cold_kg_s=m_cold,
    coeffs=coeffs, cp_method="constant"
)
print(f"[ETL-CF] Th_out={res.Th_out:.2f} °C, Tc_out={res.Tc_out:.2f} °C, "
      f"eps={res.epsilon:.3f}, q={res.q_W/1e3:.1f} kW, C*={res.Cstar:.3f}, r={res.r:.3f} m²K/W")

# --- 4) ε–NTU & LMTD: need UA (use your correlation or an engineering guess) ---
U_guess = 2500.0     # W/m²-K (illustrative; prefer a literature correlation)
UA = U_guess * A_m2  # W/K

# --- 5) ε–NTU ---
Qe, Th_e, Tc_e, eps_e = solve_outlets_entu(UA, Fcorr, Th_in, Tc_in, m_hot, m_cold, cph, cpc, flow)
print(f"[ε–NTU] Th_out={Th_e:.2f} °C, Tc_out={Tc_e:.2f} °C, eps={eps_e:.3f}, q={Qe/1e3:.1f} kW")

# --- 6) LMTD ---
Ql, Th_l, Tc_l, eps_l = solve_outlets_lmtd(UA, Fcorr, Th_in, Tc_in, m_hot, m_cold, cph, cpc, flow)
print(f"[LMTD ] Th_out={Th_l:.2f} °C, Tc_out={Tc_l:.2f} °C, eps={eps_l:.3f}, q={Ql/1e3:.1f} kW")
