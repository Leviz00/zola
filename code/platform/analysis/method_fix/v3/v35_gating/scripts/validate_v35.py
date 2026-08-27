"""validate_v35.py — v3.5 门控验证：LOCO 规则导出 + 预注册 5 条验收。

门控特征：rows/*.csv（run_gate_diag.py，data-only）。
结局（仅验证用，真值派生）：v34_full/v34f_detail.csv 的 est/oracle/plac FDP/TPR。
防过拟合：leave-one-cell-out（8 折）——门控面向未见格，rep 级切分会泄漏
格身份，故选 LOCO；规则类事先限定为 R1(perm_fdr_max) + R2(tested_min)
+ R3(φ̂ 撞下界, 固定) 的小网格。
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

BASE = "/mnt/agents/output/analysis/method_fix/v3/v35_gating"
DETAIL = "/mnt/agents/output/analysis/method_fix/v3/v34_full/v34f_detail.csv"
DETECTABLE = {1000, 1002, 1005, 1009}
MAIN_CELLS = [1000, 1002, 1004, 1005, 1006, 1007, 1008, 1009]

PERM_GRID = [0.10, 0.15, 0.20, 0.25]
TESTED_GRID = [0.50, 0.60, 0.70]


def decide(df, perm_fdr_max, tested_min, phi_lower=0.06):
    r1 = df["perm_fdr_est"] > perm_fdr_max
    r2 = df["est_tested_frac"] < tested_min
    r3 = df["phi_hat"] <= phi_lower
    on = ~(r1 | r2 | r3)
    return on, r1, r2, r3


def gated_metrics(df, on):
    fdp = np.where(on, df["estimated_fdp"], 0.0)
    se = fdp.std(ddof=1) / np.sqrt(len(fdp))
    tpr_on = df.loc[on, "estimated_tpr"].mean() if on.any() else 0.0
    tpr_all = df["estimated_tpr"].mean()
    return dict(fdr=fdp.mean(), se=se, tpr_on=tpr_on, tpr_all=tpr_all,
                on_rate=on.mean())


def select_rule(train):
    """训练格上选规则：可行（gated FDR ≤ 0.05+SE 且 TPR 保留 ≥90%）中
    on_rate 最高者；平局取 perm_fdr_max 大、tested_min 小。"""
    best = None
    for pq in PERM_GRID:
        for tm in TESTED_GRID:
            on, *_ = decide(train, pq, tm)
            m = gated_metrics(train, on)
            feasible = (m["fdr"] <= 0.05 + m["se"]
                        and m["tpr_on"] >= 0.9 * m["tpr_all"])
            key = (feasible, m["on_rate"], pq, -tm)
            if best is None or key > best[0]:
                best = (key, pq, tm, m)
    return best[1], best[2]


def main():
    rows = [pd.read_csv(p) for p in glob.glob(os.path.join(BASE, "rows", "*.csv"))]
    d = pd.concat(rows, ignore_index=True)
    det = pd.read_csv(DETAIL)
    det_main = det[~det.file.str.contains("1002s")].copy()
    d = d.merge(det_main[["file", "estimated_fdp", "estimated_tpr",
                          "oracle_fdp", "oracle_tpr", "placeholder_fdp",
                          "placeholder_tpr"]], on="file", how="inner")
    d["sensitivity"] = d.file.str.contains("1002s")
    m = d[~d.sensitivity].copy()
    m["detectable"] = m.cell_id.isin(DETECTABLE)
    m["leak_fail"] = (m.estimated_fdp - m.oracle_fdp) > 0.10

    # ---- LOCO ---------------------------------------------------------------
    print("=== LOCO (8 folds) ===")
    m["gate_on_loco"] = False
    m["loco_rule"] = ""
    for cell in MAIN_CELLS:
        train = m[m.cell_id != cell]
        test = m[m.cell_id == cell]
        pq, tm = select_rule(train)
        on, r1, r2, r3 = decide(test, pq, tm)
        m.loc[test.index, "gate_on_loco"] = on.values
        m.loc[test.index, "loco_rule"] = f"pq={pq},tm={tm}"
        print(f"cell {cell}: rule pq={pq} tm={tm} -> on_rate={on.mean():.2f} "
              f"(detectable={test.detectable.iloc[0]}) "
              f"r1={r1.mean():.2f} r2={r2.mean():.2f} r3={r3.mean():.2f}")

    # ---- 最终固化规则（全 8 格上重选一次，用于报告） -------------------------
    pq, tm = select_rule(m)
    on_all, r1a, r2a, r3a = decide(m, pq, tm)
    m["gate_on"] = on_all.values
    print(f"\nfinal rule: perm_fdr_max={pq} tested_min={tm} phi_lower=0.06")
    print(f"final on_rate={on_all.mean():.2f}  R1={r1a.mean():.2f} "
          f"R2={r2a.mean():.2f} R3={r3a.mean():.2f}")
    for col, r in (("R1", r1a), ("R2", r2a), ("R3", r3a)):
        print(col, "fires by cell:",
              m.assign(f=r.values).groupby("cell_id").f.mean().round(2).to_dict())

    # ---- 验收 2：门控全局 FDR ----------------------------------------------
    gm = gated_metrics(m, m.gate_on_loco.values)
    print("\n=== A2 gated global FDR (LOCO decisions) ===")
    print(f"gated FDR mean={gm['fdr']:.4f} SE={gm['se']:.4f} "
          f"line={0.05 + gm['se']:.4f} PASS={gm['fdr'] <= 0.05 + gm['se']}")
    gmF = gated_metrics(m, on_all.values)
    print(f"(final rule: FDR={gmF['fdr']:.4f} SE={gmF['se']:.4f})")

    # ---- 验收 3：TPR 保留 + 与可检层一致性 ---------------------------------
    print("=== A3 TPR retention & layer agreement ===")
    print(f"on-set est TPR={gm['tpr_on']:.4f}  global est TPR={gm['tpr_all']:.4f}"
          f"  retention={gm['tpr_on']/gm['tpr_all']:.3f} (need >=0.9)")
    agree = (m.gate_on_loco == m.detectable).mean()
    both = ((m.gate_on_loco == m.detectable).astype(int)
            - (~m.detectable).astype(int)).mean()
    print(f"rep-level agreement with detectable layer = {agree:.3f}")

    # ---- 验收 4：1008 关闭 ---------------------------------------------------
    off1008 = 1 - m.loc[m.cell_id == 1008, "gate_on_loco"].mean()
    print(f"=== A4 cell 1008 off-rate = {off1008:.2f} (need 1.00) ===")

    # ---- 验收 5：泄漏预检 vs 实际泄漏失败 ------------------------------------
    lf = m.dropna(subset=["w_group_r"])
    rb = np.corrcoef(lf.w_group_r.abs(), lf.leak_fail.astype(int))[0, 1]
    # 简单 AUC
    from scipy.stats import rankdata
    r_ = rankdata(lf.w_group_r.abs())
    pos = lf.leak_fail.values
    auc = ((r_[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
           / (pos.sum() * (~pos).sum())) if 0 < pos.sum() < len(pos) else np.nan
    print(f"=== A5 leakage precheck ===")
    print(f"leak_fail reps={lf.leak_fail.sum()}/{len(lf)}  "
          f"|r(W,group)| point-biserial r={rb:.3f}  AUC={auc:.3f}")
    print("w_group_r by cell:",
          m.groupby("cell_id").w_group_r.apply(lambda s: s.abs().mean())
          .round(3).to_dict())

    # ---- 混淆矩阵（rep 级） ---------------------------------------------------
    print("=== confusion: gate_on vs detectable (rep level, LOCO) ===")
    ct = pd.crosstab(m.detectable, m.gate_on_loco)
    print(ct)
    sens = ct.loc[True, True] / ct.loc[True].sum()
    spec = ct.loc[False, False] / ct.loc[False].sum()
    print(f"sensitivity(on|detectable)={sens:.3f} "
          f"specificity(off|non-detectable)={spec:.3f}")

    # ---- 副臂：est@α=0.01 在关闭集上的表现 ------------------------------------
    off = m[~m.gate_on_loco]
    print("=== fallback arm (declared fallback vs est@0.01 on gated-OFF reps) ===")
    print(f"off-set n={len(off)}  est01 FDR={off.est01_fdp.mean():.4f} "
          f"TPR={off.est01_tpr.mean():.4f}  "
          f"(est@0.05 on same set: FDR={off.estimated_fdp.mean():.4f} "
          f"TPR={off.estimated_tpr.mean():.4f}; "
          f"placeholder FDR={off.placeholder_fdp.mean():.4f})")

    # ---- 保存 ----------------------------------------------------------------
    keep_cols = ["file", "cell_id", "rep", "mechanism", "grid_group",
                 "detectable", "phi_hat", "cnt_veto", "phi_on_lower",
                 "depth_over_3phi", "zero_rate", "prev_low_frac",
                 "est_tested_frac", "perm_fdr_est", "perm_fdr_plac",
                 "perm_rej_rate_est", "w_group_r", "w_group_p",
                 "w_depth_r", "gate_on_loco", "gate_on", "loco_rule",
                 "estimated_fdp", "estimated_tpr", "oracle_fdp",
                 "placeholder_fdp", "est01_fdp", "est01_tpr", "leak_fail"]
    m[keep_cols].to_csv(os.path.join(BASE, "v35_validation.csv"), index=False)
    print("saved v35_validation.csv", len(m))


if __name__ == "__main__":
    main()
