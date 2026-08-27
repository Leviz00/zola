"""summarize_v34p.py — v3.4-PILOT 汇总：detail CSV + summary CSV。

主终点（预注册口径）：paired rep 级 FDR 差 estimated − placeholder，
均值与中位数同报。次终点：estimated − oracle 配对差；TPR 按格分层。
机制指标：Ŵ ROC-AUC/Brier。效率：分阶段 wall time + 全量外推。
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

BASE = "/mnt/agents/output/analysis/method_fix/v3/v34_pilot"


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(BASE, "npz", "*.npz"))):
        z = np.load(path, allow_pickle=False)
        r = dict(file=os.path.basename(path))
        for k in ("cell_id", "rep", "mechanism", "grid_group", "n",
                  "depth_cfg", "phi_true", "phi_hat", "cnt_veto", "cnt_lam",
                  "t_gen", "t_fit", "t_glm", "w_auc", "w_brier", "n_da",
                  "struct_frac"):
            v = z[k]
            r[k] = v.item() if hasattr(v, "item") else v
        r["t_total"] = r["t_gen"] + r["t_fit"] + r["t_glm"]
        for arm in ("oracle", "estimated", "placeholder"):
            for k in ("fdp", "tpr", "n_rej", "n_fp", "n_tp", "n_fallback"):
                r[f"{arm}_{k}"] = float(z[f"{arm}_{k}"])
        rows.append(r)
    df = pd.DataFrame(rows).sort_values(["cell_id", "rep"])
    df.to_csv(os.path.join(BASE, "v34p_detail.csv"), index=False)

    # 配对差
    df["d_fdp_est_plac"] = df["estimated_fdp"] - df["placeholder_fdp"]
    df["d_fdp_est_orac"] = df["estimated_fdp"] - df["oracle_fdp"]
    df["d_tpr_est_plac"] = df["estimated_tpr"] - df["placeholder_tpr"]
    df["d_tpr_est_orac"] = df["estimated_tpr"] - df["oracle_tpr"]

    summ = []
    for cid, g in df.groupby("cell_id"):
        s = dict(cell_id=cid, mechanism=g.mechanism.iloc[0],
                 grid_group=g.grid_group.iloc[0], R=len(g),
                 struct_frac=g.struct_frac.mean(),
                 phi_true=g.phi_true.iloc[0],
                 phi_hat_med=g.phi_hat.median(),
                 veto_rate=g.cnt_veto.mean(),
                 w_auc=g.w_auc.mean(), w_brier=g.w_brier.mean())
        for arm in ("oracle", "estimated", "placeholder"):
            s[f"{arm}_fdr"] = g[f"{arm}_fdp"].mean()
            s[f"{arm}_tpr"] = g[f"{arm}_tpr"].mean()
            s[f"{arm}_n_rej"] = g[f"{arm}_n_rej"].mean()
            s[f"{arm}_n_fallback"] = g[f"{arm}_n_fallback"].mean()
        for m in ("d_fdp_est_plac", "d_fdp_est_orac", "d_tpr_est_plac",
                  "d_tpr_est_orac"):
            s[f"{m}_mean"] = g[m].mean()
            s[f"{m}_med"] = g[m].median()
        for m in ("t_gen", "t_fit", "t_glm", "t_total"):
            s[f"{m}_mean"] = g[m].mean()
        summ.append(s)
    sdf = pd.DataFrame(summ)
    sdf.to_csv(os.path.join(BASE, "v34p_summary.csv"), index=False)

    cols = ["cell_id", "grid_group", "struct_frac", "phi_hat_med",
            "veto_rate", "w_auc", "w_brier",
            "oracle_fdr", "estimated_fdr", "placeholder_fdr",
            "oracle_tpr", "estimated_tpr", "placeholder_tpr",
            "d_fdp_est_plac_mean", "d_fdp_est_plac_med",
            "d_fdp_est_orac_mean", "d_fdp_est_orac_med",
            "t_fit_mean", "t_total_mean"]
    print(sdf[cols].to_string(index=False,
                              float_format=lambda x: f"{x:.3f}"))
    # 全量外推（2 worker）
    tot = df.t_total.mean()
    fit = df.t_fit.mean()
    print(f"\nper cell-rep: fit={fit:.0f}s total={tot:.0f}s (2-worker 墙钟)")
    for ncells in (8, 12):
        runs = ncells * 20
        hrs = runs * tot / 2 / 3600
        print(f"extrapolate {ncells} cells x R=20 = {runs} runs: "
              f"~{hrs:.1f} h (2 workers)")


if __name__ == "__main__":
    main()
