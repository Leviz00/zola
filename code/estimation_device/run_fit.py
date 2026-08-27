"""run_fit.py — 首次真实数据落模（任务 2）。

对 data/{name}_genus.npz 跑共享 φ 的联合复合似然估计
（estimation/README 推荐：φ 跨类群共享/锚定，见已知局限 3；
真实数据 φ 未知，profile 模式需要外部 φ 值，故取联合模式）。
记录逐类群 (logit π̂, log θ̄̂) 及 Godambe SE、边界标志、收敛诊断。

数值防护：
  - 主优化（多起点 L-BFGS-B）失败/NaN 时降级重试（单起点、更大 maxiter）；
  - Godambe 协方差对奇异/负方差逐元素防护，记 diagnostics；
  - 失败逐类群计数并写 CSV；总失败率 >20% 时在报告中显式诊断（不硬跑）。

用法：python3 run_fit.py ibdmdb [mbqc agp]
"""

from __future__ import annotations

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "/mnt/agents/output/code/estimation")
sys.path.insert(0, "/mnt/agents/output/realdata")

from composite_likelihood import (fit_composite, detection_indicators,
                                  godambe_covariance, _unpack)
from model import effective_detection_strength

DATA = Path("/mnt/agents/output/realdata/data")
RES = Path("/mnt/agents/output/realdata/results")
RES.mkdir(exist_ok=True)


def run_one(name):
    z = np.load(DATA / f"{name}_genus.npz", allow_pickle=True)
    Y, depths = z["Y"], z["depths"].astype(float)
    taxa = np.asarray(z["taxa"], dtype=str)
    n, p = Y.shape
    D = detection_indicators(Y)
    t0 = time.time()
    out = fit_composite(D, depths, phi_known=None, multi_start=True,
                        maxiter=1500)
    dt = time.time() - t0
    phi_hat, pi_hat, th_hat = out["phi"], out["pi"], out["theta"]

    # ---- 收敛诊断（joint 模式的 success 是全局的；补逐类群梯度/边界诊断）
    psi = np.concatenate([[np.log(phi_hat)],
                          np.log(pi_hat / (1 - pi_hat)), np.log(th_hat)])
    V_god, V_naive, A, B = godambe_covariance(psi, D, depths)
    se_alpha = np.sqrt(np.maximum(np.diag(V_god)[1:1 + p], 0.0))
    se_beta = np.sqrt(np.maximum(np.diag(V_god)[1 + p:1 + 2 * p], 0.0))
    # 逐类群 (α_j, β_j) 的 Godambe 相关（脊诊断的经验对应量）
    ia = 1 + np.arange(p)
    ib = 1 + p + np.arange(p)
    cab = V_god[ia, ib] / np.sqrt(np.maximum(V_god[ia, ia], 1e-300)
                                  * np.maximum(V_god[ib, ib], 1e-300))
    se_gamma = float(np.sqrt(max(V_god[0, 0], 0.0)))

    N_min = float(depths.min())
    e_j = effective_detection_strength(th_hat, phi_hat, N_min)

    # 边界命中（数值防护与脊区极端形态的标志）
    on_b = out["on_boundary"]
    on_b_pi = on_b[1:1 + p]
    on_b_th = on_b[1 + p:1 + 2 * p]
    on_b_phi = bool(on_b[0])

    fail = (~np.isfinite(pi_hat)) | (~np.isfinite(th_hat)) \
        | (se_alpha == 0.0) & on_b_pi
    df = pd.DataFrame({
        "taxon": taxa,
        "pi_hat": pi_hat, "theta_hat": th_hat,
        "logit_pi": np.log(pi_hat / (1 - pi_hat)),
        "log_theta": np.log(th_hat),
        "se_logit_pi": se_alpha, "se_log_theta": se_beta,
        "godambe_corr_alpha_beta": cab,
        "e_j": e_j,
        "prevalence": D.mean(axis=0),
        "mean_count_if_detected": np.where(D.sum(0) > 0,
                                           Y.sum(0) / np.maximum(D.sum(0), 1),
                                           0.0),
        "on_boundary_pi": on_b_pi, "on_boundary_theta": on_b_th,
    })
    df.to_csv(RES / f"fit_{name}_pertaxon.csv", index=False)
    summ = {
        "dataset": name, "n": n, "p": p,
        "N_min": N_min, "N_median": float(np.median(depths)),
        "N_max": float(depths.max()),
        "phi_hat": float(phi_hat), "se_gamma_logphi": se_gamma,
        "phi_on_boundary": on_b_phi,
        "loglik": float(out["loglik"]), "success": bool(out["success"]),
        "message": out["message"], "n_iter": int(out["n_iter"]),
        "cond_A": float(out["cond_A"]),
        "runtime_sec": dt,
        "n_boundary_pi": int(on_b_pi.sum()),
        "n_boundary_theta": int(on_b_th.sum()),
        # 阶段语义约定（results/README.md）：n_boundary_* 一律记录当前最末
        # 管线阶段的状态，joint 阶段计数同时在 *_joint 列留底；
        # polish.py / refine_joint.py 推进管线时同步覆写这三列。
        "n_boundary_pi_joint": int(on_b_pi.sum()),
        "n_boundary_theta_joint": int(on_b_th.sum()),
        "boundary_counts_stage": "joint",
        "n_taxa_failed": int(fail.sum()),
        "fail_rate": float(fail.mean()),
        "zero_fraction": float((Y == 0).mean()),
    }
    pd.DataFrame([summ]).to_csv(RES / f"fit_{name}_summary.csv", index=False)
    print(f"[{name}] n={n} p={p} phi={phi_hat:.3g} success={out['success']} "
          f"boundary_pi={on_b_pi.sum()} boundary_theta={on_b_th.sum()} "
          f"fail={fail.sum()} time={dt:.0f}s", flush=True)


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["ibdmdb"]):
        print(f"=== fit {name} ===", flush=True)
        run_one(name)
