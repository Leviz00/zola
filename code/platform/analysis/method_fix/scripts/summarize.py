"""summarize.py — 汇总 method_fix npz → dev_eval_summary.csv + λ 选择表。

指标（逐 npz，即 cell × rep × arm(λ)）：
  ROC-AUC（Mann–Whitney）、PR-AUC（average precision）、Brier、ECE(10 等宽
  箱)、post_mean、struct_frac、分辨率（唯一后验值比例）、phi_hat/cov_phi、
  phi 撞界、converged、t_est、φ̂ 回收 log 偏差 log(phi_hat/phi_true)。
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

NPZ_DIR = "/mnt/agents/output/analysis/method_fix/npz"
OUT_CSV = "/mnt/agents/output/analysis/method_fix/dev_eval_summary.csv"
SEL_CSV = "/mnt/agents/output/analysis/method_fix/lambda_selection.csv"


def auc_mw(labels, scores):
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores, kind="mergesort")
    sr = scores[order]
    ranks = np.empty(scores.shape[0])
    i = 0
    while i < scores.shape[0]:
        j = i
        while j + 1 < scores.shape[0] and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def pr_auc(labels, scores):
    """Average precision（sklearn 定义：PR 曲线上 precision 的召回加权均值）。"""
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    order = np.argsort(-scores, kind="mergesort")
    lab = labels[order]
    tp = np.cumsum(lab)
    fp = np.cumsum(1 - lab)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(tp[-1], 1)
    # sklearn average_precision: sum (rec_n - rec_{n-1}) * prec_n
    return float(np.sum(np.diff(rec, prepend=0.0) * prec))


def ece10(labels, scores):
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    edges = np.linspace(0.0, 1.0, 11)
    idx = np.clip(np.digitize(scores, edges) - 1, 0, 9)
    e = 0.0
    for k in range(10):
        m = idx == k
        if m.any():
            e += m.mean() * abs(labels[m].mean() - scores[m].mean())
    return float(e)


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(NPZ_DIR, "*.npz"))):
        z = np.load(path, allow_pickle=False)
        s, lab = z["scores"], z["labels"].astype(int)
        phi_hat = float(z["phi_hat"])
        phi_true = float(z["phi_true"])
        arm = str(z["arm"])
        lam = float(z["prior_lam"])
        arm_tag = arm if arm == "v1" else f"v2_l{lam:g}"
        rows.append(dict(
            file=os.path.basename(path),
            cell_id=int(z["cell_id"]), rep=int(z["rep"]), arm=arm_tag,
            mechanism=str(z["mechanism"]), informative=bool(z["informative"]),
            sz=float(z["sz"]), n=int(z["n"]), depth=int(z["depth_cfg"]),
            phi_true=phi_true,
            n_zero=int(s.size),
            roc_auc=auc_mw(lab, s),
            pr_auc=pr_auc(lab, s),
            brier=float(np.mean((s - lab) ** 2)),
            ece10=ece10(lab, s),
            post_mean=float(s.mean()),
            struct_frac=float(lab.mean()),
            uniq_frac=float(np.unique(s).size / s.size),
            phi_hat=phi_hat,
            phi_log_bias=float(np.log(phi_hat / phi_true)),
            cov_phi=float(z["cov_phi"]),
            cov_phi_log_bias=float(np.log(float(z["cov_phi"]) / phi_true)),
            phi_on_boundary=bool(z["cnt_phi_on_boundary"]),
            cov_phi_on_boundary=bool(z["cov_phi_on_boundary"]),
            converged=bool(z["cov_converged"] and z["cnt_converged"]),
            cov_converged=bool(z["cov_converged"]),
            cnt_converged=bool(z["cnt_converged"]),
            cov_nit=int(z["cov_nit"]), cnt_nit=int(z["cnt_nit"]),
            t_cov=float(z["t_cov"]), t_count=float(z["t_count"]),
            t_est=float(z["t_est"]),
            prior_lam=lam, prior_eta0=float(z["prior_eta0"]),
            cov_multistart=int(z["cov_multistart"]),
        ))
    df = pd.DataFrame(rows).sort_values(["cell_id", "arm", "rep"])
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(df)} rows)")

    # λ 选择：rep0 的 v2 各 λ，按 PR-AUC 主选、Brier 辅选
    grid = df[(df.arm != "v1") & (df.rep == 0)].copy()
    sel = []
    for cid, g in grid.groupby("cell_id"):
        g = g.sort_values(["pr_auc", "brier"], ascending=[False, True])
        best = g.iloc[0]
        sel.append(dict(cell_id=cid, best_lam=best["prior_lam"],
                        pr_auc=best["pr_auc"], brier=best["brier"],
                        table=" | ".join(
                            f"l={r.prior_lam:g}:PR={r.pr_auc:.4f},B={r.brier:.4f}"
                            for r in g.itertuples())))
    sdf = pd.DataFrame(sel)
    sdf.to_csv(SEL_CSV, index=False)
    print(sdf.to_string(index=False))
    print(f"wrote {SEL_CSV}")


if __name__ == "__main__":
    main()
