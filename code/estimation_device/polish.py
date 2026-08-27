"""polish.py — 联合拟合后的 profile 抛光与最终结果落盘（收敛修正）。

动机：联合 L-BFGS-B 受迭代上限约束时，个别类群的 (logit π_j, log θ̄_j)
坐标未走到最优点（AGP 联合拟合 5/200 类群移动 >2SE，含流行率 0.39 的
Adlercreutzia 卡在 π–θ 脊中段、SE 虚高 167）。复合似然在共享 φ 下对
(α,β) 为逐类群 2 维可分解问题，故：

  1. 以联合拟合的 φ̂ 为固定值做 profile 抛光（逐类群 2 维 L-BFGS-B，
     estimation/README 的 profile 路径，快速且稳健）；
  2. 在抛光后的完整参数向量 ψ* = (log φ̂, α*, β*) 上重算 Godambe 三明治
     （φ 自由方向保留在 A/B 中，φ 不确定性照常传播）；
  3. 逐类群收敛审计：最终点的得分绝对值（|score_α|, |score_β|）与
     抛光位移量写入 CSV；|score|>1e-3 的类群计入 unconverged 数。

原始联合拟合结果保留为 fit_{name}_pertaxon_joint.csv /
fit_{name}_summary.csv 不动（summary 中追加抛光审计行另存
fit_{name}_summary.csv 的 polish_* 列）。

用法：python3 polish.py ibdmdb [agp mbqc]
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "/mnt/agents/output/code/estimation")
from composite_likelihood import (fit_composite, detection_indicators,
                                  godambe_covariance, _neg_loglik_grad)
from model import effective_detection_strength

ROOT = Path("/mnt/agents/output/realdata")
DATA, RES = ROOT / "data", ROOT / "results"


def run(name):
    z = np.load(DATA / f"{name}_genus.npz", allow_pickle=True)
    Y, depths, taxa = z["Y"], z["depths"].astype(float), z["taxa"].astype(str)
    n, p = Y.shape
    D = detection_indicators(Y)
    df = pd.read_csv(RES / f"fit_{name}_pertaxon.csv")
    summ = pd.read_csv(RES / f"fit_{name}_summary.csv").iloc[0]
    phi_hat = float(summ["phi_hat"])

    # ---- 1) profile 抛光
    prof = fit_composite(D, depths, phi_known=phi_hat)
    alpha = np.log(prof["pi"] / (1 - prof["pi"]))
    beta = np.log(prof["theta"])
    # 边界（profile 模式的 2p 边界向量）
    on_b = prof["on_boundary"]
    on_b_pi, on_b_th = on_b[:p], on_b[p:]

    # ---- 2) 在抛光点重算 Godambe（φ 自由）
    psi = np.concatenate([[np.log(phi_hat)], alpha, beta])
    V_god, V_naive, A, B = godambe_covariance(psi, D, depths)
    ia, ib = 1 + np.arange(p), 1 + p + np.arange(p)
    se_alpha = np.sqrt(np.maximum(np.diag(V_god)[ia], 0.0))
    se_beta = np.sqrt(np.maximum(np.diag(V_god)[ib], 0.0))
    cab = V_god[ia, ib] / np.sqrt(np.maximum(V_god[ia, ia], 1e-300)
                                  * np.maximum(V_god[ib, ib], 1e-300))
    se_gamma = float(np.sqrt(max(V_god[0, 0], 0.0)))

    # ---- 3) 收敛审计（最终点得分）
    _, grad = _neg_loglik_grad(psi, D, depths, None)
    sc_a, sc_b = np.abs(grad[ia]), np.abs(grad[ib])
    unconverged = (sc_a > 1e-3) | (sc_b > 1e-3)

    N_min = float(depths.min())
    e_j = effective_detection_strength(prof["theta"], phi_hat, N_min)

    # 保留联合拟合原值（首次运行时重命名；重复运行时保留既有 *_joint 列，
    # 位移相对当前值计算，保证幂等）
    first_run = "logit_pi_joint" not in df.columns
    if first_run:
        df = df.rename(columns={
            "pi_hat": "pi_joint", "theta_hat": "theta_joint",
            "logit_pi": "logit_pi_joint", "log_theta": "log_theta_joint",
            "se_logit_pi": "se_logit_pi_joint",
            "se_log_theta": "se_log_theta_joint",
            "godambe_corr_alpha_beta": "godambe_corr_joint",
            "e_j": "e_j_joint",
            "on_boundary_pi": "on_boundary_pi_joint",
            "on_boundary_theta": "on_boundary_theta_joint"})
    prev_alpha = df["logit_pi_joint"] if first_run else df["logit_pi"]
    prev_beta = df["log_theta_joint"] if first_run else df["log_theta"]
    df["pi_hat"] = prof["pi"]
    df["theta_hat"] = prof["theta"]
    df["logit_pi"] = alpha
    df["log_theta"] = beta
    df["se_logit_pi"] = se_alpha
    df["se_log_theta"] = se_beta
    df["godambe_corr_alpha_beta"] = cab
    df["e_j"] = e_j
    df["on_boundary_pi"] = on_b_pi
    df["on_boundary_theta"] = on_b_th
    df["score_alpha"] = sc_a
    df["score_beta"] = sc_b
    df["unconverged"] = unconverged
    df["polish_shift_logit_pi"] = alpha - prev_alpha
    df["polish_shift_log_theta"] = beta - prev_beta
    df.to_csv(RES / f"fit_{name}_pertaxon.csv", index=False)

    # summary 追加抛光审计列
    s = summ.to_dict()
    s.update({
        "se_gamma_logphi": se_gamma,
        "polish_applied": True,
        "polish_success": bool(prof["success"]),
        "n_unconverged_after_polish": int(unconverged.sum()),
        # n_boundary_* 约定为最终管线状态（见 results/README.md）：
        # 抛光后边界集合已变，必须同步覆写，joint 值保留在 *_joint 列。
        "n_boundary_pi": int(on_b_pi.sum()),
        "n_boundary_theta": int(on_b_th.sum()),
        "boundary_counts_stage": "polish",
        "n_moved_gt2SE_by_polish": int((
            (np.abs(df["polish_shift_logit_pi"])
             / np.maximum(df["se_logit_pi_joint"], 1e-9) > 2)
            | (np.abs(df["polish_shift_log_theta"])
               / np.maximum(df["se_log_theta_joint"], 1e-9) > 2)).sum()),
        "max_abs_score_gamma_final": float(abs(grad[0])),
    })
    pd.DataFrame([s]).to_csv(RES / f"fit_{name}_summary.csv", index=False)
    print(f"[{name}] polished: moved>2SE={s['n_moved_gt2SE_by_polish']}, "
          f"unconverged={unconverged.sum()}/{p}, "
          f"|score_gamma|={abs(grad[0]):.2e}", flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        run(name)
