"""det_pertaxon_phi.py — 检出侧逐类群 (π_j, θ_j, φ_j) 三参数独立拟合（B2 桥接）。

现有管线的检出侧 φ 是**跨类群共享**参数（mbqc φ̂=1454）；计数侧 ZIBB 给出
逐类群 φ_j。为在同一把尺上对比两侧，本脚本对检出指示数据逐类群独立拟合
(π_j, θ_j, φ_j)（q = π(1−g(N;θ,φ))，3 维），仅保留检出曲线在该类群深度
支撑内提供 φ 信息的类群（流行率 ∈ [0.05, 0.9]：过低/过高时 φ_j 不可识别）。
输出逐类群 φ̂_det,j 与 φ̂_count,j 的配对 CSV（在 run_count_phi.py 完成后
由 compare_phi.py 合并）。
"""

from __future__ import annotations

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from scipy.optimize import minimize
from scipy.special import expit, logit

sys.path.insert(0, "/mnt/agents/output/code/estimation")
from model import log_g  # noqa: E402

RES = Path("/mnt/agents/output/realdata/phi_count_check/results")
PI_B = (logit(1e-6), logit(0.999999))
TH_B = (np.log(1e-8), np.log(0.9))
PHI_B = (np.log(0.05), np.log(1e7))

D = N = None


def _init():
    global D, N
    z = np.load("/mnt/agents/output/realdata/data/mbqc_genus.npz",
                allow_pickle=True)
    D = (z["Y"] > 0).astype(np.float64)
    N = z["depths"].astype(np.float64)


def _nll(psi, d, N):
    pi, th, phi = expit(psi[0]), np.exp(psi[1]), np.exp(psi[2])
    g = np.exp(log_g(N, th, phi))
    q = np.clip(pi * (1.0 - g), 1e-12, 1 - 1e-12)
    return -(d * np.log(q) + (1 - d) * np.log1p(-q)).sum()


def _one(j):
    d = D[:, j]
    prev = float(d.mean())
    if not (0.05 <= prev <= 0.9):
        return {"j": int(j), "prevalence": prev, "skipped": True}
    best = None
    for phi0 in (100.0, 3000.0, 1e5):
        for th0 in (3e-4, 3e-3, 3e-2):
            r = minimize(_nll, [logit(min(prev * 1.2, 0.99)), np.log(th0),
                                np.log(phi0)], args=(d, N),
                         method="L-BFGS-B", bounds=[PI_B, TH_B, PHI_B],
                         options={"maxiter": 500, "ftol": 1e-12})
            if best is None or r.fun < best.fun:
                best = r
    x = best.x
    # 数值 Hessian → SE(log φ)
    h = 1e-4
    H = np.zeros((3, 3))
    f0 = best.fun
    for a in range(3):
        for b in range(a, 3):
            ea = np.zeros(3); eb = np.zeros(3)
            ea[a] = h; eb[b] = h
            if a == b:
                H[a, a] = (_nll(x + ea, d, N) - 2 * f0 + _nll(x - ea, d, N)) / h**2
            else:
                H[a, b] = (_nll(x + ea + eb, d, N) - _nll(x + ea - eb, d, N)
                           - _nll(x - ea + eb, d, N) + _nll(x - ea - eb, d, N)
                           ) / (4 * h**2)
                H[b, a] = H[a, b]
    try:
        cov = np.linalg.inv(H + 1e-10 * np.eye(3))
        se = float(np.sqrt(max(cov[2, 2], 0.0)))
    except np.linalg.LinAlgError:
        se = np.nan
    return {"j": int(j), "prevalence": prev, "skipped": False,
            "pi_det": float(expit(x[0])), "theta_det": float(np.exp(x[1])),
            "phi_det": float(np.exp(x[2])), "se_logphi_det": se,
            "phi_det_on_boundary": bool(np.isclose(x[2], PHI_B, atol=1e-3).any()),
            "success": bool(best.success), "loglik": float(-best.fun)}


def main():
    _init()
    p = D.shape[1]
    t0 = time.time()
    with Pool(2, initializer=_init) as pool:
        rows = pool.map(_one, range(p))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "det_pertaxon_phi_mbqc.csv", index=False)
    ok = df[(~df["skipped"]) & df["success"].fillna(False)
            & (~df["phi_det_on_boundary"].fillna(False))
            & (df["se_logphi_det"] <= 1.0)]
    print(f"fitted {int((~df['skipped']).sum())} taxa, "
          f"{len(ok)} interior with se<=1; "
          f"median phi_det = {ok['phi_det'].median():.3g} "
          f"(IQR {ok['phi_det'].quantile(.25):.3g}–"
          f"{ok['phi_det'].quantile(.75):.3g}); "
          f"runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
