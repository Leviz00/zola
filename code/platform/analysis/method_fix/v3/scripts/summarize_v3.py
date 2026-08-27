"""summarize_v3.py — v3.1 汇总：dev_eval_v3.csv + 连贯 vs 混装对比表。

逐 (cell, rep) 行：W0/W1/W2 三臂的逐零后验指标（ROC-AUC、PR-AUC、Brier、
ECE10、post_mean、uniq_frac）+ DA 结局（fdp/tpr/n_rej）+ 诊断（cnt_phi、
veto、耗时）。W1 = 连贯后验（主终点）；混装对照从 v2 npz 合并
（cell 6/2/11 用 v2 λ=1.0，cell 22 用 v2 λ=0.44 —— v2 开发级选定臂）。
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

V3 = "/mnt/agents/output/analysis/method_fix/v3"
V2_NPZ = "/mnt/agents/output/analysis/method_fix/npz"
OUT_CSV = os.path.join(V3, "dev_eval_v3.csv")
MIXED_LAM = {6: "1.0", 2: "1.0", 11: "1.0", 22: "0.44"}


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
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    order = np.argsort(-scores, kind="mergesort")
    lab = labels[order]
    tp = np.cumsum(lab)
    fp = np.cumsum(1 - lab)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / max(tp[-1], 1)
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


def metrics_row(lab, s):
    return dict(roc_auc=auc_mw(lab, s), pr_auc=pr_auc(lab, s),
                brier=float(np.mean((s - lab) ** 2)), ece10=ece10(lab, s),
                post_mean=float(s.mean()),
                uniq_frac=float(np.unique(s).size / s.size))


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(V3, "npz", "*.npz"))):
        z = np.load(path, allow_pickle=False)
        lab = z["labels"].astype(int)
        cid, rep = int(z["cell_id"]), int(z["rep"])
        r = dict(file=os.path.basename(path), cell_id=cid, rep=rep,
                 mechanism=str(z["mechanism"]),
                 informative=bool(z["informative"]), sz=float(z["sz"]),
                 n=int(z["n"]), depth=int(z["depth_cfg"]),
                 phi_true=float(z["phi_true"]),
                 n_zero=int(lab.size), struct_frac=float(lab.mean()),
                 cnt_phi=float(z["cnt_phi"]),
                 cnt_phi_log_bias=float(np.log(float(z["cnt_phi"])
                                               / float(z["phi_true"]))),
                 cnt_phi_on_boundary=bool(z["cnt_phi_on_boundary"]),
                 cnt_veto_retry=bool(z["cnt_veto_retry"]),
                 cnt_lam=float(z["cnt_lam"]), det_phi=float(z["det_phi"]),
                 cnt_success=bool(z["cnt_success"]),
                 w1_abs_gamma1_mean=float(z["w1_abs_gamma1_mean"]),
                 t_count=float(z["t_count"]), t_w0=float(z["t_w0"]),
                 t_w1=float(z["t_w1"]), t_w2=float(z["t_w2"]),
                 t_est=float(z["t_est"]))
        for arm in ("w0", "w1", "w2"):
            s = z[f"scores_{arm}"]
            for k, v in metrics_row(lab, s).items():
                r[f"{arm}_{k}"] = v
            for k in ("fdp", "tpr", "n_rej"):
                r[f"{arm}_{k}"] = (float(z[f"{arm}_{k}"]) if k != "n_rej"
                                   else int(z[f"{arm}_{k}"]))
        # 混装对照（v2 npz）
        mpath = os.path.join(
            V2_NPZ, f"cell{cid:02d}_rep{rep}_v2_l{MIXED_LAM[cid]}.npz")
        if os.path.exists(mpath):
            zm = np.load(mpath, allow_pickle=False)
            for k, v in metrics_row(zm["labels"].astype(int),
                                    zm["scores"]).items():
                r[f"mixed_{k}"] = v
            r["mixed_cov_phi"] = float(zm["cov_phi"])
            r["mixed_cnt_phi"] = float(zm["phi_hat"])
        rows.append(r)
    df = pd.DataFrame(rows).sort_values(["cell_id", "rep"])
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(df)} rows)")

    # 主终点：连贯(w1) vs 混装(mixed)，格均值
    cmp_rows = []
    for cid, g in df.groupby("cell_id"):
        row = dict(cell_id=cid)
        for m in ("roc_auc", "pr_auc", "brier", "ece10", "post_mean",
                  "uniq_frac"):
            row[f"w1_{m}"] = g[f"w1_{m}"].mean()
            row[f"mixed_{m}"] = g[f"mixed_{m}"].mean()
            row[f"d_{m}"] = row[f"w1_{m}"] - row[f"mixed_{m}"]
        cmp_rows.append(row)
    cdf = pd.DataFrame(cmp_rows)
    print(cdf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # DA 结局汇总（三臂，R=3 信号筛查）
    da = (df.groupby(["cell_id", "informative"])
          [["w0_fdp", "w1_fdp", "w2_fdp", "w0_tpr", "w1_tpr", "w2_tpr",
            "w0_n_rej", "w1_n_rej", "w2_n_rej"]].mean().reset_index())
    print(da.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
