"""batch_decomposition.py — 批次对 φ 贡献的稳健方差分解（B3 收尾，替代
run_batch_phi.py 内平衡子集 ANOVA——该法因"全部层均可识别"要求只剩 1/5
个类群而失效）。

方法（不平衡，逐层 × 逐类群计数侧 log φ_j，类群须在 ≥60% 层内可识别）：
  - sd_layer_medians      ：层中位数的跨层 sd（log φ）；
  - median_within_IQR     ：层内类群间 IQR 的中位（log φ）——类群身份效应；
  - cross_layer_sd_excess ：逐类群跨层 sd 扣除抽样噪声（√max(sd²−se²,0)）
                            的中位——纯粹的批次间过度离散成分；
  - detection 侧：逐层 φ̂ 表 + 深度跨度（prop (iii) 可识别窗口：φ 仅在
    N_max ≳ 3φ 时可识别；撞界 φ̂ 只给下界 ≈ N_max/3 的语境）。
输出 results/batch_decomposition.csv（覆盖 run_batch_phi.py 的弱版本）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RES = Path("/mnt/agents/output/realdata/phi_count_check/results")
DATA = Path("/mnt/agents/output/realdata/data/mbqc_genus.npz")
META = Path("/mnt/agents/output/datasets/mbqc/mbqc_sample_metadata.csv")


def main():
    a = pd.read_csv(RES / "batch_count_phi_pertaxon.csv")
    a["logphi"] = np.log(a["phi"])
    rows = []
    for layer, lname in (("HL", "handling_lab"), ("BL", "bioinformatics")):
        sub = a[a["stratum"].str.startswith(f"{layer}:") & a["informative"]]
        nstr = sub["stratum"].nunique()
        vc = sub.groupby("taxon")["stratum"].nunique()
        ok = vc[vc >= np.ceil(0.6 * nstr)].index
        b = sub[sub["taxon"].isin(ok)]
        lm = b.groupby("stratum")["logphi"].median()
        iqr = b.groupby("stratum")["logphi"].apply(
            lambda s: s.quantile(.75) - s.quantile(.25))
        sd_t = b.groupby("taxon")["logphi"].std()
        se_t = b.groupby("taxon")["se_logphi"].median()
        excess = np.sqrt(np.maximum(sd_t**2 - se_t**2, 0)).median()
        rows.append({
            "side": "count", "layer": lname, "k_strata": nstr,
            "n_taxa_used": len(ok),
            "sd_layer_medians_logphi": float(lm.std()),
            "median_within_layer_IQR_logphi": float(iqr.median()),
            "median_per_taxon_cross_layer_sd": float(sd_t.median()),
            "median_sampling_se_logphi": float(se_t.median()),
            "excess_cross_layer_sd_logphi": float(excess),
        })
    # 检出侧：逐层 φ̂ + 深度跨度语境
    sdf = pd.read_csv(RES / "batch_strata_summary.csv")
    z = np.load(DATA, allow_pickle=True)
    samples = z["samples"].astype(str)
    depths = z["depths"].astype(float)
    m = pd.read_csv(META, usecols=["Unnamed: 0", "HL_lab", "BL_lab"],
                    low_memory=False).rename(columns={"Unnamed: 0": "sample"})
    mm = pd.Series(samples).to_frame("sample").merge(m, on="sample", how="left")
    mm["depth"] = depths
    det_rows = []
    for layer, col in (("HL", "HL_lab"), ("BL", "BL_lab")):
        sub = sdf[sdf["layer"] == layer]
        interior = sub[sub["phi_det"] < 1e5 - 1]
        det_rows.append({
            "side": "detection", "layer": f"{layer}_interior_only",
            "k_strata": len(sub),
            "n_taxa_used": int((sub["phi_det"] >= 1e5 - 1).sum()),
            "sd_layer_medians_logphi": float(np.log(interior["phi_det"]).std())
            if len(interior) > 1 else np.nan,
            "median_within_layer_IQR_logphi": np.nan,
            "median_per_taxon_cross_layer_sd": float(
                np.log(sub["phi_det"]).std()),
            "median_sampling_se_logphi": float(
                sub["se_gamma_det"].median()),
            "excess_cross_layer_sd_logphi": np.nan,
        })
    out = pd.DataFrame(rows + det_rows)
    out.to_csv(RES / "batch_decomposition.csv", index=False)
    print(out.to_string(index=False))

    # 深度跨度表（可识别窗口语境）
    spans = []
    for col in ("HL_lab", "BL_lab"):
        for lab, g in mm.groupby(col)["depth"]:
            spans.append({"layer": col, "stratum": lab, "n": len(g),
                          "depth_min": g.min(), "depth_median": g.median(),
                          "depth_max": g.max(),
                          "span_decades": float(np.log10(g.max() / g.min())),
                          "phi_identifiable_up_to_approx": g.max() / 3.0})
    pd.DataFrame(spans).to_csv(RES / "batch_depth_spans.csv", index=False)


if __name__ == "__main__":
    main()
