"""score_baselines.py — v3_baselines 计分器（SPEC §3 + Amendment 1 双口径）。

主口径 emp_fdr_uncond = 全部有效 rep 的 FDP 均值（静默 rep 记 0）；
辅助 emp_fdr_cond = 仅有 ≥1 拒绝 rep 的 FDP 均值（n_rej_reps 同报，
n_rej_reps<5 不可释读）；emp_fdr_mcse / tpr_mcse = 跨 rep 标准误。
参考臂（oracle/est/plac）从 v34f_detail.csv 逐 rep 原始输出按无条件口径
重算（其 fdp 定义本就是静默=0，行 175 确认）；校准口径对照取 v36_detail。
ANCOM-BC2 保留 q_bh 敏感性行（双口径）。
LinDA prev.filter 披露：逐格被过滤的 DA 类群均值（Amendment 1 §6(ii)）。
产出：scores_per_cell.csv / scores_overall.csv / scores_reference_arms.csv。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

BASE = "/mnt/agents/output/analysis/v3_baselines"
V34_DET = "/mnt/agents/output/analysis/method_fix/v3/v34_full/v34f_detail.csv"
V36 = "/mnt/agents/output/analysis/method_fix/v3/v36_calibration/v36_detail.csv"
CELLS = [1000, 1002, 1004, 1005, 1006, 1007, 1008, 1009]
DETECTABLE = {1000, 1002, 1005, 1009}
METHODS = ["linda", "ancombc2", "deseq2", "tss_wilcoxon"]
R_REPS = 20


def _mcse(x):
    x = np.asarray(x, dtype=float)
    return x.std() / np.sqrt(len(x)) if len(x) > 1 else np.nan


def _agg(method, cell, df, filt_counts, filt_da, n_failed):
    fdp0 = df.fdp.fillna(0.0).values  # 无条件：静默记 0
    rr = df[df.n_rej >= 1]
    return dict(method=method, cell_id=cell,
                n_valid=len(df), n_failed=n_failed, n_rej_reps=len(rr),
                emp_fdr_uncond=fdp0.mean(), emp_fdr_mcse=_mcse(fdp0),
                emp_fdr_cond=rr.fdp.mean() if len(rr) else np.nan,
                emp_fdr_cond_med=rr.fdp.median() if len(rr) else np.nan,
                tpr=df.tpr.mean(), tpr_med=df.tpr.median(),
                tpr_mcse=_mcse(df.tpr.values),
                filtered_mean=np.mean(filt_counts) if filt_counts else 0.0,
                filtered_da_mean=np.mean(filt_da) if filt_da else 0.0)


def _rep_scores(method, cell, rep, rej_col, q_col=None):
    """逐 rep 读 raw + truth，返回 (df_row, n_filtered, n_da_filtered)。
    文件缺失/失败返回 None。"""
    err = f"{BASE}/raw/{method}/cell{cell}_rep{rep}.errors.csv"
    f = f"{BASE}/raw/{method}/cell{cell}_rep{rep}.csv"
    if os.path.exists(err) or not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    d = d.drop(columns=["abs_da_truth"], errors="ignore")
    t = pd.read_csv(f"{BASE}/exchange/truth/cell{cell}_rep{rep}.csv")
    d = d.merge(t, on="taxon", how="left")
    da = d["abs_da_truth"].astype(int) == 1
    n_filt = n_da_filt = 0
    if "filtered" in d.columns:
        fm = d["filtered"].fillna(False).astype(bool)
        n_filt, n_da_filt = int(fm.sum()), int((fm & da).sum())
    if rej_col == "q_bh":
        rej = d["q_bh"] < 0.05
    else:
        rej = d["rejected"].fillna(0).astype(int) == 1
    fp, tp = int((rej & ~da).sum()), int((rej & da).sum())
    return (dict(rep=rep, n_rej=fp + tp, fp=fp, tp=tp,
                 fdp=fp / (fp + tp) if fp + tp else np.nan,
                 tpr=tp / max(int(da.sum()), 1)), n_filt, n_da_filt)


def score_method_cell(method, cell):
    recs, fc, fd, n_failed = [], [], [], 0
    for rep in range(R_REPS):
        out = _rep_scores(method, cell, rep, "rejected")
        if out is None:
            n_failed += 1
            continue
        recs.append(out[0])
        fc.append(out[1])
        fd.append(out[2])
    df = pd.DataFrame(recs)
    if len(df) == 0:
        return dict(method=method, cell_id=cell, n_valid=0, n_failed=n_failed,
                    n_rej_reps=0, emp_fdr_uncond=np.nan, emp_fdr_mcse=np.nan,
                    emp_fdr_cond=np.nan, emp_fdr_cond_med=np.nan, tpr=np.nan,
                    tpr_med=np.nan, tpr_mcse=np.nan, filtered_mean=np.nan,
                    filtered_da_mean=np.nan)
    out = _agg(method, cell, df, fc, fd, n_failed)
    if method == "ancombc2":  # q_bh 敏感性（双口径）
        recs2 = []
        for rep in range(R_REPS):
            o = _rep_scores(method, cell, rep, "q_bh")
            if o is not None:
                recs2.append(o[0])
        if recs2:
            d2 = pd.DataFrame(recs2)
            fdp0 = d2.fdp.fillna(0.0).values
            rr2 = d2[d2.n_rej >= 1]
            out.update(bh_emp_fdr_uncond=fdp0.mean(),
                       bh_emp_fdr_cond=rr2.fdp.mean() if len(rr2) else np.nan,
                       bh_tpr=d2.tpr.mean(), bh_n_rej_reps=len(rr2))
    return out


def reference_arms():
    """oracle/est/plac 无条件口径（v34f_detail 逐 rep 原始输出重算）。"""
    v = pd.read_csv(V34_DET)
    if "sensitivity" in v.columns:  # 排除 1002 敏感性臂（η₀=log 500）
        v = v[~v["sensitivity"].astype(bool)]
    rows = []
    for arm, pre in (("oracle", "oracle"), ("est", "estimated"),
                     ("plac", "placeholder")):
        for cell in CELLS:
            g = v[v.cell_id == cell]
            fdp, tpr = g[f"{pre}_fdp"].values, g[f"{pre}_tpr"].values
            rows.append(dict(method=f"ref_{arm}_raw", cell_id=cell,
                             n_valid=len(g), n_rej_reps=int((g[f"{pre}_n_rej"] > 0).sum()),
                             emp_fdr_uncond=fdp.mean(), emp_fdr_mcse=_mcse(fdp),
                             tpr=tpr.mean(), tpr_mcse=_mcse(tpr)))
    return pd.DataFrame(rows)


def _overall(df, name_col="method"):
    ov = []
    for m in df[name_col].unique():
        sub0 = df[df[name_col] == m]
        for scope, sub in (("all", sub0),
                           ("detectable", sub0[sub0.detectable])):
            ov.append(dict(method=m, scope=scope,
                           emp_fdr_uncond=sub.emp_fdr_uncond.mean(),
                           emp_fdr_mcse=sub.emp_fdr_mcse.mean(),
                           emp_fdr_cond=sub.emp_fdr_cond.mean()
                           if "emp_fdr_cond" in sub else np.nan,
                           tpr=sub.tpr.mean(), tpr_med=sub.tpr_med.median()
                           if "tpr_med" in sub else np.nan,
                           tpr_mcse=sub.tpr_mcse.mean(),
                           rej_reps=f"{int(sub.n_rej_reps.sum())}/"
                                    f"{int(sub.n_valid.sum())}"))
    return pd.DataFrame(ov)


def main():
    rows = [score_method_cell(m, c) for m in METHODS for c in CELLS]
    df = pd.DataFrame(rows)
    df["detectable"] = df.cell_id.isin(DETECTABLE)
    df.to_csv(f"{BASE}/scores_per_cell.csv", index=False)

    ref = reference_arms()
    ref["detectable"] = ref.cell_id.isin(DETECTABLE)
    ref.to_csv(f"{BASE}/scores_reference_arms.csv", index=False)

    odf = _overall(df)
    # 校准口径对照列（v36_detail，无条件）
    v36 = pd.read_csv(V36)
    cal = []
    for scope, sub in (("all", v36), ("detectable",
                                      v36[v36.cell_id.isin(DETECTABLE)])):
        cal.append(dict(method="ref_est_cal", scope=scope,
                        emp_fdr_uncond=sub.est_fdp.mean(),
                        tpr=sub.est_tpr.mean()))
        cal.append(dict(method="ref_plac_cal", scope=scope,
                        emp_fdr_uncond=sub.plac_fdp.mean(),
                        tpr=sub.plac_tpr.mean()))
    odf = pd.concat([odf, _overall(ref), pd.DataFrame(cal)],
                    ignore_index=True)
    odf.to_csv(f"{BASE}/scores_overall.csv", index=False)
    print(df.round(3)[["method", "cell_id", "n_rej_reps", "emp_fdr_uncond",
                       "emp_fdr_mcse", "emp_fdr_cond", "tpr", "tpr_mcse",
                       "filtered_mean", "filtered_da_mean"]]
          .to_string(index=False))
    print(odf.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
