"""summarize_v34f.py — v3.4 全量确认汇总（预注册分析逻辑）。

预注册判定线（本脚本实现，MEMO 逐条引用）：
  共同主终点 (a)：est 臂全体平均 FDR ≤ 0.05 + MC-SE（受控）；
                  配对 FDR 差 est−plac 均值+中位数同报。
  共同主终点 (b)：可检层（oracle 设计期 TPR≥0.5：1000/1002/1005/1009）
                  内配对 TPR 差 est−plac 单侧 Wilcoxon 符号秩显著 >0。
  次终点：est−oracle FDR 配对差 |均值| ≤ 0.02；Ŵ AUC/Brier 仅 0<share<1
          格（1004-1007）；n_tested = 100 − n_fallback。
  敏感性臂：cell1002s（η₀=log 500）单独报告。
可检性在设计期数据上预先写死（v323 冒烟 rep0 + B2 R=20 + v34 试点）：
  detectable = {1000, 1002, 1005, 1009}；not = {1004, 1006, 1007, 1008}。
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASE = "/mnt/agents/output/analysis/method_fix/v3/v34_full"
DETECTABLE = {1000, 1002, 1005, 1009}
MAIN_CELLS = [1000, 1002, 1004, 1005, 1006, 1007, 1008, 1009]


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(BASE, "npz", "*.npz"))):
        z = np.load(path, allow_pickle=False)
        r = dict(file=os.path.basename(path))
        for k in ("cell_id", "rep", "mechanism", "grid_group", "n",
                  "depth_cfg", "phi_true", "phi_hat", "cnt_veto", "cnt_lam",
                  "t_gen", "t_fit", "t_glm", "w_auc", "w_brier", "n_da",
                  "struct_frac", "prior_eta0"):
            v = z[k]
            r[k] = v.item() if hasattr(v, "item") else v
        r["t_total"] = r["t_gen"] + r["t_fit"] + r["t_glm"]
        for arm in ("oracle", "estimated", "placeholder"):
            for k in ("fdp", "tpr", "n_rej", "n_fp", "n_tp", "n_fallback"):
                r[f"{arm}_{k}"] = float(z[f"{arm}_{k}"])
            r[f"{arm}_n_tested"] = 100 - r[f"{arm}_n_fallback"]
        rows.append(r)
    df = pd.DataFrame(rows)
    df["sensitivity"] = df.file.str.contains("cell1002s")
    df.to_csv(os.path.join(BASE, "v34f_detail.csv"), index=False)

    main_df = df[~df.sensitivity].copy()
    sens_df = df[df.sensitivity].copy()
    for d in (main_df, sens_df):
        d["d_fdp_ep"] = d["estimated_fdp"] - d["placeholder_fdp"]
        d["d_fdp_eo"] = d["estimated_fdp"] - d["oracle_fdp"]
        d["d_tpr_ep"] = d["estimated_tpr"] - d["placeholder_tpr"]
        d["d_tpr_eo"] = d["estimated_tpr"] - d["oracle_tpr"]

    # ---- 共同主终点 (a) ----------------------------------------------------
    est_fdr = main_df["estimated_fdp"]
    se = est_fdr.std(ddof=1) / np.sqrt(len(est_fdr))
    print("=== co-primary (a): FDR ===")
    print(f"est arm mean FDR = {est_fdr.mean():.4f}  MC-SE = {se:.4f}  "
          f"line 0.05+SE = {0.05 + se:.4f}  "
          f"CONTROLLED = {est_fdr.mean() <= 0.05 + se}")
    print(f"paired FDR diff est-plac: mean={main_df.d_fdp_ep.mean():.4f} "
          f"median={main_df.d_fdp_ep.median():.4f}")

    # ---- 共同主终点 (b) ----------------------------------------------------
    det = main_df[main_df.cell_id.isin(DETECTABLE)]
    ndet = main_df[~main_df.cell_id.isin(DETECTABLE)]
    stat, pv = wilcoxon(det.d_tpr_ep, alternative="greater",
                        zero_method="wilcox")
    print("=== co-primary (b): TPR (detectable layer) ===")
    print(f"detectable cells {sorted(DETECTABLE)} n_pairs={len(det)}")
    print(f"paired TPR diff est-plac: mean={det.d_tpr_ep.mean():.4f} "
          f"median={det.d_tpr_ep.median():.4f}  wilcoxon(greater) "
          f"stat={stat:.0f} p={pv:.3g}")
    print(f"non-detectable layer: mean={ndet.d_tpr_ep.mean():.4f} "
          f"median={ndet.d_tpr_ep.median():.4f}")

    # ---- 次终点 ------------------------------------------------------------
    print("=== secondary ===")
    print(f"paired FDR diff est-oracle: mean={main_df.d_fdp_eo.mean():.4f} "
          f"median={main_df.d_fdp_eo.median():.4f} "
          f"(line |mean|<=0.02: {abs(main_df.d_fdp_eo.mean()) <= 0.02})")
    print(f"paired TPR diff est-oracle: mean={main_df.d_tpr_eo.mean():.4f} "
          f"median={main_df.d_tpr_eo.median():.4f}")
    rf = main_df[main_df.grid_group == "realfrac"]
    print(f"W-hat (realfrac, n={len(rf)}): AUC mean="
          f"{rf.w_auc.dropna().mean():.4f} median={rf.w_auc.dropna().median():.4f}"
          f"  Brier mean={rf.w_brier.mean():.4f}")

    # ---- 格 × 臂表 ----------------------------------------------------------
    summ = (main_df.groupby(["cell_id", "grid_group"])
            .agg(struct_frac=("struct_frac", "mean"),
                 detectable=("cell_id", lambda s: s.iloc[0] in DETECTABLE),
                 phi_hat_med=("phi_hat", "median"),
                 veto=("cnt_veto", "mean"),
                 oracle_fdr=("oracle_fdp", "mean"),
                 est_fdr=("estimated_fdp", "mean"),
                 plac_fdr=("placeholder_fdp", "mean"),
                 oracle_tpr=("oracle_tpr", "mean"),
                 est_tpr=("estimated_tpr", "mean"),
                 plac_tpr=("placeholder_tpr", "mean"),
                 d_fdp_ep=("d_fdp_ep", "mean"),
                 d_tpr_ep=("d_tpr_ep", "mean"),
                 est_tested=("estimated_n_tested", "mean"),
                 orac_tested=("oracle_n_tested", "mean"),
                 plac_tested=("placeholder_n_tested", "mean"),
                 w_auc=("w_auc", "mean"), w_brier=("w_brier", "mean"),
                 t_fit=("t_fit", "mean"))
            .reset_index())
    summ.to_csv(os.path.join(BASE, "v34f_summary.csv"), index=False)
    print(summ.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- 敏感性臂 -----------------------------------------------------------
    if len(sens_df):
        print("=== sensitivity: cell 1002, eta0=log(500) ===")
        m1002 = main_df[main_df.cell_id == 1002]
        for tag, d in (("main(eta0=log50)", m1002), ("sens(eta0=log500)", sens_df)):
            print(f"{tag}: phi_hat med={d.phi_hat.median():.3g} "
                  f"veto={d.cnt_veto.mean():.2f} "
                  f"est_fdr={d.estimated_fdp.mean():.4f} "
                  f"est_tpr={d.estimated_tpr.mean():.4f} "
                  f"d_fdp_ep={d.d_fdp_ep.mean():.4f} "
                  f"d_tpr_ep={d.d_tpr_ep.mean():.4f} "
                  f"w_brier={d.w_brier.mean():.4f}")
        sens_df.to_csv(os.path.join(BASE, "v34f_sensitivity.csv"), index=False)

    tot = main_df.t_total.sum() + sens_df.t_total.sum()
    print(f"\nruns: main={len(main_df)} sens={len(sens_df)} "
          f"total compute={tot/3600:.1f}h (2-worker 墙钟合计)")


if __name__ == "__main__":
    main()
