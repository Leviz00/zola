"""compare_phi.py — 计数侧 vs 检出侧 φ 的定量对比（任务 B2）。

输入：
  results/count_phi_mbqc_pertaxon.csv   计数侧逐类群 ZIBB φ_j（run_count_phi.py）
  results/det_pertaxon_phi_mbqc.csv     检出侧逐类群 (π,θ,φ_j)（det_pertaxon_phi.py，
                                        可选——存在则做配对分析）
输出：
  results/compare_phi_summary.csv       汇总对比（中位数/IQR/精度加权 vs 1454）
  results/compare_phi_paired.csv        逐类群配对 φ_count vs φ_det
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

RES = Path("/mnt/agents/output/realdata/phi_count_check/results")
PHI_DET_SHARED = 1454.0
SE_GAMMA_DET = 0.058


def main():
    df = pd.read_csv(RES / "count_phi_mbqc_pertaxon.csv")
    # success=False 的 85/223 个类群全部为 L-BFGS-B 线搜索良性退出（估计
    # README 已知局限 6 同型）：全部落在内点、se_logphi ≤ 0.36。故可识别
    # 判据用 se/边界/正计数数，不以 success 为条件（已在报告声明）。
    inf = df[(df["n_pos"] >= 1000) & (df["se_logphi"] <= 0.5)
             & (~df["phi_on_boundary"])]
    rows = [{
        "quantity": "detection_side_shared_phi", "value": PHI_DET_SHARED,
        "note": "run_fit+refine, SE(log phi)=0.058 -> 95% CI "
                f"[{PHI_DET_SHARED*np.exp(-1.96*SE_GAMMA_DET):.0f},"
                f"{PHI_DET_SHARED*np.exp(1.96*SE_GAMMA_DET):.0f}]"},
        {"quantity": "count_median_phi_all", "value": float(df["phi"].median()),
         "note": f"n={len(df)} taxa"},
        {"quantity": "count_median_phi_informative",
         "value": float(inf["phi"].median()),
         "note": f"n={len(inf)} taxa (n_pos>=1000, se<=0.5, interior; "
                 "success flag relaxed, see header comment)"},
        {"quantity": "count_q25_phi_informative",
         "value": float(inf["phi"].quantile(.25)), "note": ""},
        {"quantity": "count_q75_phi_informative",
         "value": float(inf["phi"].quantile(.75)), "note": ""},
        {"quantity": "count_precision_weighted_mean_logphi",
         "value": float(np.average(np.log(inf["phi"]),
                                   weights=1/inf["se_logphi"]**2)),
         "note": "exp of this = "
                 f"{float(np.exp(np.average(np.log(inf['phi']), weights=1/inf['se_logphi']**2))):.3g}"},
        {"quantity": "ratio_det_shared_over_count_median_informative",
         "value": PHI_DET_SHARED / float(inf["phi"].median()), "note": ""},
    ]

    det_file = RES / "det_pertaxon_phi_mbqc.csv"
    if det_file.exists():
        dd = pd.read_csv(det_file)
        dd = dd[(~dd["skipped"]) & (~dd["phi_det_on_boundary"].fillna(False))
                & (dd["se_logphi_det"] <= 1.0)]
        m = inf.merge(dd[["j", "phi_det", "se_logphi_det"]], on="j")
        sp = spearmanr(np.log(m["phi"]), np.log(m["phi_det"]))
        m["log10_ratio_count_over_det"] = np.log10(m["phi"] / m["phi_det"])
        m.to_csv(RES / "compare_phi_paired.csv", index=False)
        rows += [
            {"quantity": "paired_n_taxa", "value": len(m),
             "note": "both sides interior & precise"},
            {"quantity": "paired_spearman_logphi",
             "value": float(sp.statistic), "note": f"p={sp.pvalue:.3g}"},
            {"quantity": "paired_median_phi_det",
             "value": float(m["phi_det"].median()), "note": ""},
            {"quantity": "paired_median_phi_count",
             "value": float(m["phi"].median()), "note": ""},
            {"quantity": "paired_median_log10_ratio_count_over_det",
             "value": float(m["log10_ratio_count_over_det"].median()),
             "note": "median log10(phi_count/phi_det)"},
            {"quantity": "paired_frac_count_lt_det",
             "value": float((m["phi"] < m["phi_det"]).mean()), "note": ""},
        ]
    out = pd.DataFrame(rows)
    out.to_csv(RES / "compare_phi_summary.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
