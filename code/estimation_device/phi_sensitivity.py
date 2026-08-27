"""phi_sensitivity.py — φ 撞边界时的敏感性分析（辅助诊断）。

ibdmdb 联合估计 φ̂ 撞上界 1e5（深度支撑 N_max≈3.2e4 ≪ 3φ̂，形状信息不足，
prop (iii) 的真实数据演示）。为检验"可识别区/脊区"划分对该 φ 选择的
稳健性，在固定 φ ∈ {10, 100, 1000, 1e4, 1e5} 的 profile 模式下重估
（φ 固定时目标按类群分解为 2 维问题，README 推荐路径），比较：
  - 逐类群乘积泛函 π̂_j θ̄_j（稀有区唯一可识别对象）的跨 φ 稳定性；
  - 基于 SE(logit π̂)<1 的区域划分一致性（与联合拟合的 zone 的吻合率）。

用法：python3 phi_sensitivity.py ibdmdb
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "/mnt/agents/output/code/estimation")
from composite_likelihood import fit_composite, detection_indicators  # noqa

ROOT = Path("/mnt/agents/output/realdata")
DATA, RES = ROOT / "data", ROOT / "results"
PHI_GRID = [10.0, 100.0, 1000.0, 1e4, 1e5]


def run(name):
    z = np.load(DATA / f"{name}_genus.npz", allow_pickle=True)
    Y, depths, taxa = z["Y"], z["depths"].astype(float), z["taxa"].astype(str)
    D = detection_indicators(Y)
    base = pd.read_csv(RES / f"identifiability_{name}.csv")
    rows = []
    for phi in PHI_GRID:
        out = fit_composite(D, depths, phi_known=phi)
        prod = out["pi"] * out["theta"]
        se_a = out["se_alpha"]
        zone = (se_a < 1.0) & ~out["on_boundary"][0:len(taxa)]
        agree = float((zone == (base["zone"] == "identifiable")).mean())
        # 乘积泛函跨 φ 的相关（log 尺度，对照联合拟合 φ=1e5 臂）
        lp = np.log(np.maximum(prod, 1e-300))
        lp0 = np.log(np.maximum(base["pi_hat"] * base["theta_hat"], 1e-300))
        rows.append({
            "phi_fixed": phi,
            "fit_success": bool(out["success"]),
            "n_boundary_pi": int(out["on_boundary"][:len(taxa)].sum()),
            "logprod_corr_vs_joint": float(np.corrcoef(lp, lp0)[0, 1]),
            "logprod_mae_vs_joint": float(np.mean(np.abs(lp - lp0))),
            "zone_agreement_vs_joint": agree,
            "frac_identifiable": float(zone.mean()),
        })
        print(f"  phi={phi:g} agree={agree:.3f} "
              f"logprod corr={rows[-1]['logprod_corr_vs_joint']:.4f}",
              flush=True)
    pd.DataFrame(rows).to_csv(RES / f"phi_sensitivity_{name}.csv", index=False)
    print(f"[{name}] phi sensitivity done")


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["ibdmdb"]):
        run(name)
