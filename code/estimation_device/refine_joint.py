"""refine_joint.py — 从抛光点暖启动的联合精化（块坐标精化的第二步）。

流程：读 results/fit_{name}_pertaxon.csv（已含抛光后 (logit_pi, log_theta)）
与 summary 的 φ̂ → 以此为单起点暖启动联合 L-BFGS-B（多方向联合，含 γ）
→ 再 profile 抛光 → 在最终点重算 Godambe 与收敛审计 → 更新 CSV
（refine_* 列记录精化位移；fit_success 按投影梯度 <1e-3 重判）。

用法：python3 refine_joint.py agp
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

sys.path.insert(0, "/mnt/agents/output/code/estimation")
from composite_likelihood import (detection_indicators, godambe_covariance,
                                  _neg_loglik_grad, fit_composite)
from model import effective_detection_strength
from scipy.special import logit as _logit

ROOT = Path("/mnt/agents/output/realdata")
DATA, RES = ROOT / "data", ROOT / "results"


def run(name, maxiter=800):
    z = np.load(DATA / f"{name}_genus.npz", allow_pickle=True)
    Y, depths, taxa = z["Y"], z["depths"].astype(float), z["taxa"].astype(str)
    n, p = Y.shape
    D = detection_indicators(Y)
    df = pd.read_csv(RES / f"fit_{name}_pertaxon.csv")
    summ = pd.read_csv(RES / f"fit_{name}_summary.csv").iloc[0]

    psi0 = np.concatenate([[np.log(float(summ["phi_hat"]))],
                           df["logit_pi"].to_numpy(),
                           df["log_theta"].to_numpy()])
    bounds = ([(np.log(0.05), np.log(1e5))]
              + [(_logit(1e-4), _logit(0.9999))] * p
              + [(np.log(1e-7), np.log(0.9))] * p)
    res = minimize(_neg_loglik_grad, psi0, args=(D, depths, None),
                   method="L-BFGS-B", jac=True, bounds=bounds,
                   options={"maxiter": maxiter, "ftol": 1e-13, "gtol": 1e-9})
    phi_ref = float(np.exp(res.x[0]))
    print(f"[{name}] warm joint: nit={res.nit} dloglik={-res.fun - float(summ['loglik']):.2f} "
          f"phi {summ['phi_hat']:.4g} -> {phi_ref:.4g}", flush=True)

    # 再抛光（α,β | φ_ref）
    prof = fit_composite(D, depths, phi_known=phi_ref)
    alpha = np.log(prof["pi"] / (1 - prof["pi"]))
    beta = np.log(prof["theta"])
    psi = np.concatenate([[np.log(phi_ref)], alpha, beta])

    V_god, V_naive, A, B = godambe_covariance(psi, D, depths)
    ia, ib = 1 + np.arange(p), 1 + p + np.arange(p)
    se_alpha = np.sqrt(np.maximum(np.diag(V_god)[ia], 0.0))
    se_beta = np.sqrt(np.maximum(np.diag(V_god)[ib], 0.0))
    cab = V_god[ia, ib] / np.sqrt(np.maximum(V_god[ia, ia], 1e-300)
                                  * np.maximum(V_god[ib, ib], 1e-300))
    se_gamma = float(np.sqrt(max(V_god[0, 0], 0.0)))
    _, grad = _neg_loglik_grad(psi, D, depths, None)
    sc_a, sc_b = np.abs(grad[ia]), np.abs(grad[ib])
    unconverged = (sc_a > 1e-3) | (sc_b > 1e-3)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    on_b = np.isclose(psi, lo, atol=1e-6) | np.isclose(psi, hi, atol=1e-6)

    e_j = effective_detection_strength(prof["theta"], phi_ref,
                                       float(depths.min()))
    df["refine_shift_logit_pi"] = alpha - df["logit_pi"]
    df["refine_shift_log_theta"] = beta - df["log_theta"]
    df["pi_hat"] = prof["pi"]; df["theta_hat"] = prof["theta"]
    df["logit_pi"] = alpha; df["log_theta"] = beta
    df["se_logit_pi"] = se_alpha; df["se_log_theta"] = se_beta
    df["godambe_corr_alpha_beta"] = cab
    df["e_j"] = e_j
    df["on_boundary_pi"] = on_b[ia]; df["on_boundary_theta"] = on_b[ib]
    df["score_alpha"] = sc_a; df["score_beta"] = sc_b
    df["unconverged"] = unconverged
    df.to_csv(RES / f"fit_{name}_pertaxon.csv", index=False)

    s = summ.to_dict()
    s.update({
        "phi_hat": phi_ref, "se_gamma_logphi": se_gamma,
        "phi_on_boundary": bool(on_b[0]),
        "loglik": float(-res.fun), "success": bool(
            np.max(np.abs(grad)) < 1e-3 or res.success),
        "refine_applied": True, "refine_n_iter": int(res.nit),
        "n_unconverged_after_polish": int(unconverged.sum()),
        "n_boundary_pi": int(on_b[ia].sum()),
        "n_boundary_theta": int(on_b[ib].sum()),
        "boundary_counts_stage": "refine",
        "max_abs_score_gamma_final": float(abs(grad[0])),
        "cond_A": float(np.linalg.cond(A)),
    })
    pd.DataFrame([s]).to_csv(RES / f"fit_{name}_summary.csv", index=False)
    print(f"[{name}] refined: phi={phi_ref:.4g} se_gamma={se_gamma:.3f} "
          f"unconverged={unconverged.sum()}/{p} "
          f"|score_gamma|={abs(grad[0]):.2e} maxshift_a="
          f"{np.max(np.abs(df['refine_shift_logit_pi'])):.3f}", flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        run(name)
