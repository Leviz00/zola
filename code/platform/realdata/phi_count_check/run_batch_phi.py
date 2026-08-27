"""run_batch_phi.py — 按实验室（HL）/生信流程（BL）分层重估 φ（任务 B3）。

检验"φ̂ 承载的是生物异质还是批次异质"：
  (a) 检出侧：每个层（14 HL 实验室 + 9 BL 流程）子集上跑现有管线
      fit_composite 联合估计（单起点，maxiter=1000；与 run_fit.py 第一阶段
      同口径），得层内共享 φ̂_det 及 Godambe SE(log φ)；
  (b) 计数侧：每个层子集上逐类群 ZIBB MLE（count_phi.py，2 起点提速），
      层内中位 φ（可识别子集：n_pos ≥ max(30, 10%·n_layer)，SE(logφ) ≤ 0.5，
      内点，收敛），并保留逐类群值用于跨层方差分解；
  (c) 方差分解：逐类群 log φ_j（计数侧）按层做单因素 ANOVA（between/within
      均方、组内相关 ICC）；层中位数的标准差 vs 层内中位 IQR；
      检出侧层 φ̂ 的离散度 vs 全局 φ̂=1454 的 SE(0.058)。

输出：
  results/batch_strata_summary.csv      逐层 n、检出侧 φ̂、计数侧中位 φ
  results/batch_count_phi_pertaxon.csv  层 × 类群计数侧 φ_j（方差分解用）
  results/batch_decomposition.csv       ANOVA/ICC/离散度汇总
"""

from __future__ import annotations

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool

sys.path.insert(0, "/mnt/agents/output/code/estimation")
sys.path.insert(0, "/mnt/agents/output/realdata/phi_count_check")
from composite_likelihood import fit_composite, detection_indicators  # noqa: E402
from count_phi import fit_taxon_zibb  # noqa: E402

DATA = Path("/mnt/agents/output/realdata/data/mbqc_genus.npz")
META = Path("/mnt/agents/output/datasets/mbqc/mbqc_sample_metadata.csv")
RES = Path("/mnt/agents/output/realdata/phi_count_check/results")
RES.mkdir(parents=True, exist_ok=True)

Y = DEPTHS = None


def _init():
    global Y, DEPTHS
    z = np.load(DATA, allow_pickle=True)
    Y = z["Y"].astype(np.float64)
    DEPTHS = z["depths"].astype(np.float64)


def _detection_fit(idx):
    """层子集联合检出曲线拟合 → (phi, se_gamma, success, loglik)。"""
    D = detection_indicators(Y[idx])
    N = DEPTHS[idx]
    out = fit_composite(D, N, phi_known=None, multi_start=False, maxiter=1000)
    return (float(out["phi"]), float(out.get("se_gamma", np.nan)),
            bool(out["success"]), float(out["loglik"]), str(out["message"]))


def _count_fit(idx, taxa_idx, n_layer):
    """层子集逐类群 ZIBB（2 起点提速）。"""
    rows = []
    thr = max(30, int(0.1 * n_layer))
    for j in taxa_idx:
        y, N = Y[idx, j], DEPTHS[idx]
        n_pos = int((y > 0).sum())
        r = fit_taxon_zibb(y, N, phi0_grid=(300.0, 3e4))
        r["j"] = int(j)
        r["n_pos"] = n_pos
        r["informative"] = bool(n_pos >= thr and r["se_logphi"] <= 0.5
                                and not r["phi_on_boundary"] and r["success"])
        rows.append(r)
    return rows


def _stratum(args):
    name, idx, taxa_idx = args
    idx = np.asarray(idx)
    t0 = time.time()
    phi_d, se_g, succ, ll, msg = _detection_fit(idx)
    crows = _count_fit(idx, taxa_idx, len(idx))
    cdf = pd.DataFrame(crows)
    cdf["stratum"] = name
    inf = cdf[cdf["informative"]]
    summ = {
        "stratum": name, "n": len(idx),
        "phi_det": phi_d, "se_gamma_det": se_g, "det_success": succ,
        "det_loglik": ll, "det_message": msg,
        "count_n_informative": len(inf),
        "count_median_phi": float(inf["phi"].median()) if len(inf) else np.nan,
        "count_q25_phi": float(inf["phi"].quantile(.25)) if len(inf) else np.nan,
        "count_q75_phi": float(inf["phi"].quantile(.75)) if len(inf) else np.nan,
        "runtime_sec": time.time() - t0,
    }
    return summ, cdf


def main():
    _init()
    z = np.load(DATA, allow_pickle=True)
    taxa = z["taxa"].astype(str)
    samples = z["samples"].astype(str)
    m = pd.read_csv(META, usecols=["Unnamed: 0", "HL_lab", "BL_lab"],
                    low_memory=False).rename(columns={"Unnamed: 0": "sample"})
    mm = pd.Series(samples).to_frame("sample").merge(m, on="sample", how="left")
    n, p = Y.shape

    # 计数侧分层仅在全球流行率 ≥0.15 的类群上做（提速；稀有类群层内不可识别）
    prev = (Y > 0).mean(axis=0)
    taxa_idx = np.where(prev >= 0.15)[0]
    print(f"stratified count-side on {len(taxa_idx)}/{p} taxa "
          f"(global prevalence >= 0.15)", flush=True)

    jobs = []
    for lab, grp in mm.groupby("HL_lab"):
        jobs.append((f"HL:{lab}", grp.index.to_numpy(), taxa_idx))
    for bl, grp in mm.groupby("BL_lab"):
        jobs.append((f"BL:{bl}", grp.index.to_numpy(), taxa_idx))

    t0 = time.time()
    with Pool(2, initializer=_init) as pool:
        results = pool.map(_stratum, jobs)
    sums, cdfs = zip(*results)
    sdf = pd.DataFrame(sums)
    sdf["layer"] = sdf["stratum"].str.split(":").str[0]
    sdf.to_csv(RES / "batch_strata_summary.csv", index=False)
    alldf = pd.concat(cdfs, ignore_index=True)
    alldf["taxon"] = taxa[alldf["j"].to_numpy()]
    alldf.to_csv(RES / "batch_count_phi_pertaxon.csv", index=False)
    print(sdf.to_string(index=False))

    # ---- 方差分解（计数侧 log φ_j，逐层 × 逐类群，仅双侧均可识别格）-------
    dec = []
    for layer, lname in (("HL", "handling lab"), ("BL", "bioinformatics")):
        sub = alldf[alldf["stratum"].str.startswith(f"{layer}:")
                    & alldf["informative"]].copy()
        sub["logphi"] = np.log(sub["phi"])
        # 类群 × 层 平衡子集（该类群在所有层均可识别）
        vc = sub.groupby("taxon")["stratum"].nunique()
        balanced = vc[vc == sub["stratum"].nunique()].index
        b = sub[sub["taxon"].isin(balanced)]
        grp_layer = b.groupby("stratum")["logphi"]
        grand = b["logphi"].mean()
        k = b["stratum"].nunique()
        n_per = b.groupby("stratum").size().mean()
        ssb = ((grp_layer.mean() - grand) ** 2 * b.groupby("stratum").size()
               ).sum()
        ssw = ((b["logphi"] - b["stratum"].map(grp_layer.mean())) ** 2).sum()
        dfb, dfw = k - 1, len(b) - k
        msb, msw = ssb / dfb, ssw / dfw
        icc = (msb - msw) / (msb + (n_per - 1) * msw) if msb > 0 else np.nan
        layer_medians = b.groupby("stratum")["logphi"].median()
        within_iqr = b.groupby("stratum")["logphi"].apply(
            lambda s: s.quantile(.75) - s.quantile(.25))
        dec.append({
            "layer": lname, "k_strata": k,
            "n_cells_used": len(b), "n_taxa_balanced": len(balanced),
            "anova_F": (msb / msw) if msw > 0 else np.nan,
            "df_between": dfb, "df_within": dfw,
            "MS_between": msb, "MS_within": msw,
            "ICC_logphi": icc,
            "sd_of_layer_medians_logphi": float(layer_medians.std()),
            "median_within_layer_IQR_logphi": float(within_iqr.median()),
        })
    # 检出侧层 φ̂ 离散度
    for layer in ("HL", "BL"):
        sub = sdf[sdf["layer"] == layer]
        dec.append({
            "layer": f"detection-side {layer}",
            "k_strata": len(sub),
            "n_cells_used": np.nan, "n_taxa_balanced": np.nan,
            "anova_F": np.nan, "df_between": np.nan, "df_within": np.nan,
            "MS_between": np.nan, "MS_within": np.nan, "ICC_logphi": np.nan,
            "sd_of_layer_medians_logphi": float(np.log(sub["phi_det"]).std()),
            "median_within_layer_IQR_logphi": np.nan,
        })
    pd.DataFrame(dec).to_csv(RES / "batch_decomposition.csv", index=False)
    print(pd.DataFrame(dec).to_string(index=False))
    print(f"total runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
