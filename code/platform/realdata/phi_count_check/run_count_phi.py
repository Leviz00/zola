"""run_count_phi.py — mbqc 生物样本计数幅度侧逐类群 ZIBB φ 估计（任务 B1）。

对 data/mbqc_genus.npz（13,562 × 223）逐属 ZIBB MLE（count_phi.py），
输出逐类群 CSV 与汇总 CSV（中位数/IQR，全体与"可识别"子集）。
可识别子集判据：正计数样本数 ≥ 1000、SE(log φ) ≤ 0.5、未撞 φ 边界。
"""

from __future__ import annotations

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool

sys.path.insert(0, "/mnt/agents/output/realdata/phi_count_check")
from count_phi import fit_taxon_zibb  # noqa: E402

OUT = Path("/mnt/agents/output/realdata/phi_count_check")
RES = OUT / "results"
RES.mkdir(parents=True, exist_ok=True)

Y = DEPTHS = None


def _init():
    global Y, DEPTHS
    z = np.load("/mnt/agents/output/realdata/data/mbqc_genus.npz",
                allow_pickle=True)
    Y = z["Y"].astype(np.float64)
    DEPTHS = z["depths"].astype(np.float64)


def _one(j):
    r = fit_taxon_zibb(Y[:, j], DEPTHS)
    r["j"] = j
    return r


def summarize(df, tag):
    inf = df[(df["n_pos"] >= 1000) & (df["se_logphi"] <= 0.5)
             & (~df["phi_on_boundary"]) & df["success"]]
    return {
        "subset": tag,
        "n_taxa": len(df), "n_informative": len(inf),
        "median_phi_all": float(df["phi"].median()),
        "q25_phi_all": float(df["phi"].quantile(0.25)),
        "q75_phi_all": float(df["phi"].quantile(0.75)),
        "median_phi_informative": float(inf["phi"].median()),
        "q25_phi_informative": float(inf["phi"].quantile(0.25)),
        "q75_phi_informative": float(inf["phi"].quantile(0.75)),
        "median_logphi_informative": float(np.log(inf["phi"]).median()),
        "sd_logphi_informative": float(np.log(inf["phi"]).std()),
        "iw_mean_logphi": float(  # 精度加权（1/SE²）均值 log φ
            np.average(np.log(inf["phi"]),
                       weights=1.0 / inf["se_logphi"] ** 2)),
        "n_phi_boundary": int(df["phi_on_boundary"].sum()),
        "n_fail": int((~df["success"]).sum()),
    }


def main():
    _init()
    z = np.load("/mnt/agents/output/realdata/data/mbqc_genus.npz",
                allow_pickle=True)
    taxa = z["taxa"].astype(str)
    n, p = Y.shape
    t0 = time.time()
    with Pool(2, initializer=_init) as pool:
        rows = pool.map(_one, range(p))
    df = pd.DataFrame(rows).sort_values("j").reset_index(drop=True)
    df["taxon"] = taxa[df["j"]]
    df["n_pos"] = (Y > 0).sum(axis=0)[df["j"]]
    df["prevalence"] = df["n_pos"] / n
    df.to_csv(RES / "count_phi_mbqc_pertaxon.csv", index=False)
    s = summarize(df, "mbqc_bio_global")
    s["runtime_sec"] = time.time() - t0
    pd.DataFrame([s]).to_csv(RES / "count_phi_mbqc_summary.csv", index=False)
    print(pd.DataFrame([s]).T.to_string())


if __name__ == "__main__":
    main()
