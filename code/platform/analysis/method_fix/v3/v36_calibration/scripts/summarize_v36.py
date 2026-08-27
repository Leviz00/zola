"""summarize_v36.py — v3.6 A1/A3/A4 汇总（预注册判定）。

A1：校准后（est 臂）——
  * 不可检层 FDR ≤ 0.05+MC-SE（渐近版 0.249）；
  * 可检层 TPR ≥ 渐近版 90%（渐近 0.770 → 线 0.693）；
  * 共同主终点重评：est 全体 FDR ≤ 0.05+SE；可检层 est−plac TPR 差
    单侧 Wilcoxon 显著 >0。
A3：门控重评——R1 换 calibrated FWER（≤0.25 开），R2/R3 沿用 v35；
  开/关集变化 + 门控后全局 FDR。
A4：分臂墙钟实测。
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

BASE = "/mnt/agents/output/analysis/method_fix/v3/v36_calibration"
V35 = "/mnt/agents/output/analysis/method_fix/v3/v35_gating/v35_validation.csv"
DETECTABLE = {1000, 1002, 1005, 1009}
ASYM_TPR_DET = 0.770  # v34_full 渐近版可检层 est TPR（对照基准）


def main():
    d = pd.concat([pd.read_csv(p) for p in
                   glob.glob(os.path.join(BASE, "rows", "*.csv"))],
                  ignore_index=True)
    d["detectable"] = d.cell_id.isin(DETECTABLE)
    d.to_csv(os.path.join(BASE, "v36_detail.csv"), index=False)

    det = d[d.detectable]
    nd = d[~d.detectable]

    print("=== A1 calibrated retest (est arm) ===")
    f_nd, se_nd = nd.est_fdp.mean(), nd.est_fdp.std() / np.sqrt(len(nd))
    print(f"non-det layer FDR = {f_nd:.4f}  SE={se_nd:.4f}  "
          f"line={0.05 + se_nd:.4f}  PASS={f_nd <= 0.05 + se_nd}")
    tpr_det = det.est_tpr.mean()
    print(f"det layer est TPR = {tpr_det:.4f}  (asymptotic {ASYM_TPR_DET}, "
          f"retention={tpr_det / ASYM_TPR_DET:.3f}, "
          f"line 0.693, PASS={tpr_det >= 0.9 * ASYM_TPR_DET})")
    f_all, se_all = d.est_fdp.mean(), d.est_fdp.std() / np.sqrt(len(d))
    print(f"global est FDR = {f_all:.4f} SE={se_all:.4f} "
          f"line={0.05 + se_all:.4f} PASS={f_all <= 0.05 + se_all}")
    dd = det.est_tpr - det.plac_tpr
    stat, pv = wilcoxon(dd, alternative="greater", zero_method="wilcox")
    print(f"co-primary TPR diff est-plac (det layer): mean={dd.mean():.4f} "
          f"median={dd.median():.4f} wilcoxon p={pv:.3g}")
    dfdr = d.est_fdp - d.plac_fdp
    print(f"paired FDR diff est-plac (all): mean={dfdr.mean():.4f} "
          f"median={dfdr.median():.4f}")
    print(f"det layer: est FDR={det.est_fdp.mean():.4f} "
          f"plac FDR={det.plac_fdp.mean():.4f} "
          f"est TPR={det.est_tpr.mean():.3f} plac TPR={det.plac_tpr.mean():.3f}")
    print(f"non-det   : est FDR={nd.est_fdp.mean():.4f} "
          f"plac FDR={nd.plac_fdp.mean():.4f} "
          f"est TPR={nd.est_tpr.mean():.3f} plac TPR={nd.plac_tpr.mean():.3f}")
    print("per-cell:",
          d.groupby("cell_id").agg(fdp=("est_fdp", "mean"),
                                   tpr=("est_tpr", "mean"),
                                   fwer=("est_fwer", "mean"),
                                   heavy=("est_n_heavy", "mean")).round(3)
          .to_string())

    print("\n=== A3 gate re-eval (R1 <- calibrated FWER<=0.25) ===")
    v35 = pd.read_csv(V35)[["file", "est_tested_frac", "phi_hat"]]
    m = d.merge(v35, on="file", how="left")
    m["gate_on"] = ((m.est_fwer <= 0.25)
                    & (m.est_tested_frac >= 0.50)
                    & (m.phi_hat > 0.06))
    fdp_g = np.where(m.gate_on, m.est_fdp, 0.0)
    se_g = fdp_g.std() / np.sqrt(len(fdp_g))
    print(f"on_rate by cell:",
          m.groupby("cell_id").gate_on.mean().round(2).to_dict())
    print(f"on-set={m.gate_on.sum()}/{len(m)}  "
          f"gated global FDR={fdp_g.mean():.4f} SE={se_g:.4f} "
          f"line={0.05 + se_g:.4f} PASS={fdp_g.mean() <= 0.05 + se_g}")
    print(f"on-set est TPR={m.loc[m.gate_on, 'est_tpr'].mean():.4f}")
    print(f"agreement with detectable: "
          f"{(m.gate_on == m.detectable).mean():.3f}")
    print("confusion:\n", pd.crosstab(m.detectable, m.gate_on))

    print("\n=== A4 timing ===")
    print(f"per run: est {d.est_t.mean():.0f}s plac {d.plac_t.mean():.0f}s "
          f"total {(d.est_t + d.plac_t).mean():.0f}s; "
          f"sum={(d.est_t + d.plac_t).sum() / 3600:.1f}h compute")

    print("\n=== pooling diagnostics (est arm, all runs) ===")
    print(f"pooling_ok rate={d.diag_pooling_ok.mean():.3f}  "
          f"n_heavy mean={d.diag_n_heavy.mean():.2f}  "
          f"sd/mc mean={d.diag_sd_over_mc.mean():.2f}  "
          f"null_q99 mean={d.diag_null_q99.mean():.2f} (chi2 6.63)")


if __name__ == "__main__":
    main()
