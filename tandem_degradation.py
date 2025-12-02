"""
Colab-ready simulation of degradation in 2-terminal perovskite-silicon tandem modules.
Only numpy and matplotlib are required. Run top-to-bottom to reproduce figures.
"""

# =============================
# Imports
# =============================
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# =============================
# Helper constants and utilities
# =============================
K_BOLTZ = 1.380649e-23  # J/K
Q_E = 1.602176634e-19   # C


def thermal_voltage(T: float) -> float:
    """Return kT/q in volts at temperature T [K]."""
    return (K_BOLTZ * T) / Q_E


# =============================
# Data structures
# =============================
@dataclass
class SubcellParams:
    """Compact model parameters for a single-junction solar cell.

    All quantities are expressed per unit area (A/cm^2, Ohm*cm^2).
    """

    Jph: float              # Photocurrent density [A/cm^2]
    J01: float              # Diode 1 saturation current [A/cm^2]
    n1: float               # Ideality factor for diode 1
    J02: float = 0.0        # Diode 2 saturation current [A/cm^2]
    n2: float = 2.0         # Ideality factor for diode 2
    Rs: float = 0.0         # Series resistance [Ohm*cm^2]
    Rsh: float = np.inf     # Shunt resistance [Ohm*cm^2]
    T: float = 298.15       # Cell temperature [K]


# =============================
# Single-cell compact model helpers
# =============================
def diode_currents(Vb: float, params: SubcellParams) -> float:
    """Return total diode current density (Jd1 + Jd2) at branch voltage Vb.

    Parameters
    ----------
    Vb : float
        Voltage across the diode + shunt branch [V].
    params : SubcellParams
        Electrical parameters for the subcell.
    """
    vt = thermal_voltage(params.T)
    arg1 = np.clip(Vb / (params.n1 * vt), -100, 100)
    Jd1 = params.J01 * (np.exp(arg1) - 1.0)

    if params.J02 > 0:
        arg2 = np.clip(Vb / (params.n2 * vt), -100, 100)
        Jd2 = params.J02 * (np.exp(arg2) - 1.0)
    else:
        Jd2 = 0.0
    return Jd1 + Jd2


def branch_residual(Vb: float, Jtarget: float, params: SubcellParams) -> float:
    """Residual between calculated and target current for a solar cell branch.

    Parameters
    ----------
    Vb : float
        Branch voltage [V].
    Jtarget : float
        Target terminal current density [A/cm^2].
    params : SubcellParams
        Cell parameters.
    """
    Jd = diode_currents(Vb, params)
    Jsh = 0.0 if np.isinf(params.Rsh) else Vb / params.Rsh
    Jcalc = params.Jph - Jd - Jsh
    return Jcalc - Jtarget


def estimate_voc(params: SubcellParams) -> float:
    """Rough Voc estimate ignoring Rs/Rsh, using Jph and (J01 + J02)."""
    vt = thermal_voltage(params.T)
    J0_eff = params.J01 + params.J02
    if J0_eff <= 0:
        return 0.0
    # Add 1 inside the log to avoid negative values near Jph ~ J0.
    return params.n1 * vt * np.log(params.Jph / J0_eff + 1.0)


def solve_branch_voltage_for_current(
    Jtarget: float,
    params: SubcellParams,
    V_low: float = -0.5,
    V_high: float | None = None,
    max_iter: int = 60,
    tol: float = 1e-9,
) -> float:
    """Solve branch voltage Vb such that J(Vb) = Jtarget via bisection.

    Parameters
    ----------
    Jtarget : float
        Target terminal current density [A/cm^2].
    params : SubcellParams
        Cell parameters.
    V_low : float, optional
        Lower bound for Vb search [V]. Default -0.5 V.
    V_high : float or None, optional
        Upper bound for Vb search [V]. If None, estimated from Voc.
    max_iter : int, optional
        Maximum iterations for bisection.
    tol : float, optional
        Convergence tolerance on residual.
    """
    if V_high is None:
        Voc_est = estimate_voc(params)
        V_high = 1.5 * max(Voc_est, 0.1)

    f_low = branch_residual(V_low, Jtarget, params)
    f_high = branch_residual(V_high, Jtarget, params)

    # Expand upper bracket if needed to ensure a sign change.
    expand = 0
    while f_low * f_high > 0 and expand < 10:
        V_high *= 1.5
        f_high = branch_residual(V_high, Jtarget, params)
        expand += 1

    if f_low * f_high > 0:
        # Fallback: return high voltage (approaches Voc) if bracket failed.
        return V_high

    for _ in range(max_iter):
        V_mid = 0.5 * (V_low + V_high)
        f_mid = branch_residual(V_mid, Jtarget, params)
        if abs(f_mid) < tol:
            return V_mid
        if f_low * f_mid < 0:
            V_high, f_high = V_mid, f_mid
        else:
            V_low, f_low = V_mid, f_mid
    return 0.5 * (V_low + V_high)


# =============================
# Tandem JV (series coupling)
# =============================
def tandem_jv(
    p_top: SubcellParams,
    p_bot: SubcellParams,
    T: float = 298.15,
    Rs_extra: float = 0.0,
    num_points: int = 200,
):
    """Compute JV of a 2T perovskite/Si tandem under current matching.

    Parameters
    ----------
    p_top, p_bot : SubcellParams
        Parameters for top (perovskite) and bottom (Si) subcells.
    T : float, optional
        Operating temperature [K]. (Currently uses cell-specific T values.)
    Rs_extra : float, optional
        Additional series resistance (interconnects) [Ohm*cm^2].
    num_points : int, optional
        Number of current points in the sweep.

    Returns
    -------
    V_tandem : np.ndarray
        Tandem voltage array [V].
    J_grid : np.ndarray
        Current density sweep [A/cm^2].
    V_top_arr, V_bot_arr : np.ndarray
        Terminal voltages of top and bottom subcells [V].
    """
    # Limit current sweep to the smaller photocurrent to respect current matching.
    J_lim = min(p_top.Jph, p_bot.Jph) * 0.999
    J_grid = np.linspace(0.0, J_lim, num_points)

    V_top_arr = np.zeros_like(J_grid)
    V_bot_arr = np.zeros_like(J_grid)
    V_tandem = np.zeros_like(J_grid)

    for i, J in enumerate(J_grid):
        Vb_top = solve_branch_voltage_for_current(J, p_top)
        Vb_bot = solve_branch_voltage_for_current(J, p_bot)

        V_top = Vb_top - J * p_top.Rs
        V_bot = Vb_bot - J * p_bot.Rs
        V_tot = V_top + V_bot - J * Rs_extra

        V_top_arr[i] = V_top
        V_bot_arr[i] = V_bot
        V_tandem[i] = V_tot

    return V_tandem, J_grid, V_top_arr, V_bot_arr


# =============================
# Performance metrics
# =============================
def extract_performance(V: np.ndarray, J: np.ndarray) -> dict:
    """Extract Voc, Jsc, FF, and efficiency from JV data.

    Parameters
    ----------
    V : np.ndarray
        Voltage array [V].
    J : np.ndarray
        Current density array [A/cm^2].

    Returns
    -------
    dict
        Keys: Voc, Jsc, Vmpp, Jmpp, Pmpp (W/cm^2), FF, eta.
    """
    idx_voc = np.argmin(np.abs(J))
    Voc = V[idx_voc]

    idx_jsc = np.argmin(np.abs(V))
    Jsc = J[idx_jsc]

    P = V * J
    idx_mpp = np.argmax(P)
    Vmpp, Jmpp, Pmpp = V[idx_mpp], J[idx_mpp], P[idx_mpp]

    FF = 0.0 if (Voc * Jsc) == 0 else Pmpp / (Voc * Jsc)
    Pin = 0.1  # W/cm^2 (100 mW/cm^2)
    eta = Pmpp / Pin

    return {
        "Voc": Voc,
        "Jsc": Jsc,
        "Vmpp": Vmpp,
        "Jmpp": Jmpp,
        "Pmpp": Pmpp,
        "FF": FF,
        "eta": eta,
    }


# =============================
# Degradation models
# =============================
def degrade_params(
    top0: SubcellParams,
    bot0: SubcellParams,
    t_years: float,
    mode: str = "none",
    **kwargs,
) -> tuple[SubcellParams, SubcellParams]:
    """Return degraded (top, bottom) parameters at time t_years.

    This function implements simple knobs inspired by literature:
    - Jph declines (e.g., bleaching or collection efficiency loss)
    - J0 increases (e.g., interface recombination, passivation loss)
    - Rs increases (contact corrosion)
    - Rsh decreases (shunting)
    - Qian bleaching: Jph loss in top transmits to bottom via alpha_coup.
    """

    # Copy baseline values
    top = SubcellParams(**{f: getattr(top0, f) for f in top0.__dataclass_fields__})
    bot = SubcellParams(**{f: getattr(bot0, f) for f in bot0.__dataclass_fields__})

    # Default rates
    r_Jph_top = kwargs.get("r_Jph_top_per_year", 0.0)
    r_Jph_bot = kwargs.get("r_Jph_bot_per_year", 0.0)
    decades_J0_top = kwargs.get("decades_J0_top_per_year", 0.0)
    decades_J0_bot = kwargs.get("decades_J0_bot_per_year", 0.0)
    r_Rsh_top = kwargs.get("r_Rsh_top_per_year", 0.0)
    r_Rsh_bot = kwargs.get("r_Rsh_bot_per_year", 0.0)
    r_Rs_top = kwargs.get("r_Rs_top_per_year", 0.0)
    r_Rs_bot = kwargs.get("r_Rs_bot_per_year", 0.0)
    alpha_coup = kwargs.get("alpha_coup", 0.89)

    if mode in ("top_Jph_linear", "qian_bleach"):
        top.Jph = max(0.0, top0.Jph * (1.0 - r_Jph_top * t_years))
        if mode == "qian_bleach":
            delta = top0.Jph - top.Jph
            bot.Jph = max(0.0, bot0.Jph + alpha_coup * delta)
    if mode == "bottom_Jph_linear":
        bot.Jph = max(0.0, bot0.Jph * (1.0 - r_Jph_bot * t_years))

    if mode == "top_J0_exp":
        top.J01 = top0.J01 * (10 ** (decades_J0_top * t_years))
    if mode == "bottom_J0_exp":
        bot.J01 = bot0.J01 * (10 ** (decades_J0_bot * t_years))

    if mode == "top_Rsh_exp_decay":
        top.Rsh = top0.Rsh * np.exp(-r_Rsh_top * t_years)
    if mode == "bottom_Rsh_exp_decay":
        bot.Rsh = bot0.Rsh * np.exp(-r_Rsh_bot * t_years)

    if mode == "top_Rs_linear":
        top.Rs = top0.Rs * (1.0 + r_Rs_top * t_years)
    if mode == "bottom_Rs_linear":
        bot.Rs = bot0.Rs * (1.0 + r_Rs_bot * t_years)

    # Combined modes can be encoded by calling with multiple steps externally
    # or by passing mode="none" and custom kwargs; here we keep it simple.
    return top, bot


# =============================
# Lifetime simulation
# =============================
def simulate_lifetime(
    top0: SubcellParams,
    bot0: SubcellParams,
    years: float = 30.0,
    dt_years: float = 0.5,
    mode: str = "qian_bleach",
    **kwargs,
):
    """Simulate tandem degradation over time horizon.

    Parameters
    ----------
    top0, bot0 : SubcellParams
        Baseline top and bottom cell parameters.
    years : float, optional
        Total simulation time [years].
    dt_years : float, optional
        Time step [years].
    mode : str, optional
        Degradation mode passed to ``degrade_params``.

    Returns
    -------
    t_grid : np.ndarray
        Time points [years].
    metrics : dict of np.ndarray
        Each key contains an array vs time: Voc, Jsc, FF, eta, EY_norm.
    """
    t_grid = np.arange(0.0, years + dt_years, dt_years)
    n = len(t_grid)

    Voc_arr = np.zeros(n)
    Jsc_arr = np.zeros(n)
    FF_arr = np.zeros(n)
    eta_arr = np.zeros(n)

    for i, t in enumerate(t_grid):
        top_t, bot_t = degrade_params(top0, bot0, t, mode=mode, **kwargs)
        V, J, _, _ = tandem_jv(top_t, bot_t)
        perf = extract_performance(V, J)
        Voc_arr[i] = perf["Voc"]
        Jsc_arr[i] = perf["Jsc"]
        FF_arr[i] = perf["FF"]
        eta_arr[i] = perf["eta"]

    # Simple normalized energy yield: cumulative efficiency normalized to ideal.
    eta_ref = eta_arr[0] if eta_arr[0] > 0 else 1.0
    EY_norm = np.cumsum(eta_arr) / (eta_ref * np.arange(1, n + 1))

    metrics = {
        "Voc": Voc_arr,
        "Jsc": Jsc_arr,
        "FF": FF_arr,
        "eta": eta_arr,
        "EY_norm": EY_norm,
    }
    return t_grid, metrics


# =============================
# Baseline parameters
# =============================
top_base = SubcellParams(
    Jph=0.0199,
    J01=1.5e-18,
    n1=1.6,
    J02=0.0,
    n2=2.0,
    Rs=2.5,
    Rsh=1e5,
)

bot_base = SubcellParams(
    Jph=0.0199,
    J01=5.0e-13,
    n1=1.1,
    J02=0.0,
    n2=2.0,
    Rs=0.5,
    Rsh=1e4,
)


# =============================
# Plotting utilities
# =============================
def plot_jv_over_time(times, top0, bot0, mode, kwargs, title):
    """Plot JV curves at selected times for a scenario."""
    plt.figure(figsize=(6, 5))
    for t in times:
        top_t, bot_t = degrade_params(top0, bot0, t, mode=mode, **kwargs)
        V, J, _, _ = tandem_jv(top_t, bot_t)
        plt.plot(V, J * 1e3, label=f"t={t:.0f} y")
    plt.xlabel("Voltage [V]")
    plt.ylabel("Current density [mA/cm$^2$]")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_metrics(t, metrics, title):
    """Plot Voc, Jsc, FF, and efficiency vs time."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()

    axes[0].plot(t, metrics["Voc"])
    axes[0].set_ylabel("Voc [V]")
    axes[0].grid(True)

    axes[1].plot(t, metrics["Jsc"] * 1e3)
    axes[1].set_ylabel("Jsc [mA/cm$^2$]")
    axes[1].grid(True)

    axes[2].plot(t, metrics["FF"])
    axes[2].set_ylabel("Fill factor")
    axes[2].set_xlabel("Time [years]")
    axes[2].grid(True)

    axes[3].plot(t, metrics["eta"] * 100)
    axes[3].set_ylabel("Efficiency [%]")
    axes[3].set_xlabel("Time [years]")
    axes[3].grid(True)

    fig.suptitle(title)
    plt.tight_layout()

    # Energy yield plot
    plt.figure(figsize=(6, 4))
    plt.plot(t, metrics["EY_norm"])
    plt.xlabel("Time [years]")
    plt.ylabel("Normalized EY")
    plt.title(f"Energy yield trend: {title}")
    plt.grid(True)
    plt.tight_layout()


# =============================
# Main execution with scenarios
# =============================
if __name__ == "__main__":
    # Scenario 1: baseline (no degradation)
    t1, m1 = simulate_lifetime(top_base, bot_base, mode="none")
    plot_jv_over_time([0, 10, 20, 30], top_base, bot_base, "none", {}, "JV: Baseline")
    plot_metrics(t1, m1, "Baseline (no degradation)")

    # Scenario 2: top Jph loss with Qian bleaching
    qian_kwargs = {"r_Jph_top_per_year": 0.02, "alpha_coup": 0.89}
    t2, m2 = simulate_lifetime(top_base, bot_base, mode="qian_bleach", **qian_kwargs)
    plot_jv_over_time([0, 10, 20, 30], top_base, bot_base, "qian_bleach", qian_kwargs, "JV: Qian bleaching")
    plot_metrics(t2, m2, "Top Jph loss with Qian bleaching (2%/yr)")

    # Scenario 3: strong shunt degradation in top cell
    rsh_kwargs = {"r_Rsh_top_per_year": 0.25}
    t3, m3 = simulate_lifetime(top_base, bot_base, mode="top_Rsh_exp_decay", **rsh_kwargs)
    plot_jv_over_time([0, 10, 20, 30], top_base, bot_base, "top_Rsh_exp_decay", rsh_kwargs, "JV: Top shunting")
    plot_metrics(t3, m3, "Top Rsh exponential decay (25%/yr)")

    plt.show()
