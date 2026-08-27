"""sweep.py — M4 校准扫描：θ̄ 网格 × 深度情景 × φ 已知/联合。

设计（复用 code/estimation 的生成器与估计器，与 Exp 1–3 同规格）：
  - φ=3000（主校准精度），深度窗口 [1e3,1e5]（Exp 1 规格）；
  - 8 个类群一行网格：φθ̄ ∈ geomspace(0.05, 4, 8)（跨脊区→可识别区，
    e_j(N_min) 从 ~0.014 到 ~1.2），π 交替 0.8/0.95；
  - 情景 clean：深度 log-U[1e3,1e5]（Exp 1）；
  - 情景 spike：95% log-U[1e3,1e5] + 5% U{1..10}（mbqc 式 N_min=1 离群，
    检验 N_min 版 e_j 的退化与分位版的稳健性）；
  - 臂：φ 已知（profile）与 φ 未知（joint）。

每重复每类群记录：π̂、logit 误差、Godambe SE(logit π̂)、撞界标志、
以及逐重复的候选判据得分：e_j(N_min)、e_j(N_q10)、ē_j、n·ē_j、I_j（剖面
Fisher 信息，criteria.fisher_per_taxon）。

运行：python3 sweep.py  → results/sweep_perrep.csv, results/sweep_cell.csv
"""

from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

sys.path.insert(0, "/mnt/agents/output/code/estimation")
sys.path.insert(0, "/mnt/agents/output/analysis/ej_criterion")
import model                      # noqa: E402
import composite_likelihood as cl  # noqa: E402
import criteria                   # noqa: E402

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

PHI = 3000.0
N_SAMP = 200
Z95 = 1.959964
PHT_GRID = np.geomspace(0.05, 4.0, 8)        # φθ̄ 网格
TH_GRID = PHT_GRID / PHI
PI_GRID = np.where(np.arange(8) % 2 == 0, 0.8, 0.95)


def _depths(rng, scenario):
    if scenario == "clean":
        return np.exp(rng.uniform(np.log(1e3), np.log(1e5), N_SAMP)).astype(int)
    if scenario == "spike":
        N = np.exp(rng.uniform(np.log(1e3), np.log(1e5), N_SAMP)).astype(int)
        k = int(round(0.05 * N_SAMP))
        idx = rng.choice(N_SAMP, k, replace=False)
        N[idx] = rng.integers(1, 11, k)
        return N
    raise ValueError(scenario)


def _rep(job):
    seed, scenario, arm = job
    rng = np.random.default_rng(seed)
    N = _depths(rng, scenario)
    theta_bar = np.concatenate([TH_GRID, [1.0 - TH_GRID.sum()]])
    pi_all = np.concatenate([PI_GRID, [1.0]])
    Y, _ = model.simulate_three_layer(pi_all, theta_bar, PHI, N, rng)
    D = cl.detection_indicators(Y)[:, :8]
    f = cl.fit_composite(D, N, phi_known=PHI if arm == "phiknown" else None)
    # 判据得分（真参数 + 本重复实现深度）
    ej0 = np.array([criteria.ej_min(t, PHI, N) for t in TH_GRID])
    ejq = np.array([criteria.ej_quantile(t, PHI, N, 0.10) for t in TH_GRID])
    ejm = np.array([criteria.ej_mean(t, PHI, N) for t in TH_GRID])
    Ij = np.array([criteria.fisher_per_taxon(p, t, PHI, N)[1]
                   for p, t in zip(PI_GRID, TH_GRID)])
    rows = []
    for j in range(8):
        ah = logit(np.clip(f["pi"][j], 1e-9, 1 - 1e-9))
        la = logit(PI_GRID[j])
        rows.append({
            "seed": seed, "scenario": scenario, "arm": arm, "taxon": j,
            "pi_true": PI_GRID[j], "theta_true": TH_GRID[j],
            "phitheta": PHT_GRID[j],
            "pi_hat": f["pi"][j], "logit_err": ah - la,
            "se_alpha": f["se_alpha"][j],
            "cov95": float(abs(ah - la) <= Z95 * f["se_alpha"][j]),
            "on_bnd_pi": bool(f["pi"][j] > 0.999 or f["pi"][j] < 1.1e-4),
            "on_bnd_th": bool(f["theta"][j] > 0.8),
            "e_j_min": ej0[j], "e_j_q10": ejq[j], "e_j_mean": ejm[j],
            "n_e_j_mean": N_SAMP * ejm[j], "I_j": Ij[j],
            "N_min": int(N.min()), "N_q10": float(np.quantile(N, 0.1)),
        })
    return rows


def main(R=400, cores=2):
    jobs = []
    for scenario in ("clean", "spike"):
        for r in range(R):
            jobs.append((10_000 + r, scenario, "phiknown"))
    for r in range(R):
        jobs.append((20_000 + r, "clean", "joint"))
    t0 = time.time()
    with Pool(cores) as pool:
        out = pool.map(_rep, jobs, chunksize=4)
    df = pd.DataFrame([r for rep in out for r in rep])
    df.to_csv(OUT / "sweep_perrep.csv", index=False)
    print("wrote %d rows, %.0fs" % (len(df), time.time() - t0))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=400)
    ap.add_argument("--cores", type=int, default=2)
    a = ap.parse_args()
    main(a.R, a.cores)
