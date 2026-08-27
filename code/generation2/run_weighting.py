"""run_weighting.py — upstream-weighted multiple testing smoke (Remark 8 /
Theorem E of theory_draft_v1).

Scenario: a mixed-testability family mimicking IBDMDB-type cohorts —
concatenate the 100 taxa of cell 1005 (testable regime) and the 100 taxa of
cell 1008 (hopeless regime) into ONE BH family of 200 hypotheses per rep
(each taxon keeps the permutation p-value from its own dataset's two-channel
test; upstream covariates are label-blind functions of each dataset).

Procedures compared at alpha = 0.05 (all valid; Theorem E):
  plain    : BH on all 200 p-values, unit weights.
  gated    : 0/1 weights — drop taxa whose upstream testability is in the
             bottom half (the gate as binary weighting; data-only rule
             w=0 iff info_j < median(info)), BH on the kept set.
  weighted : GRW weighted BH with w_j = info_j / mean(info_j), where
             info_j = z-scale proxy of channel information:
             info_det = sum_i qhat_j(N_i)(1-qhat_j(N_i)) (Fisher proxy),
             info_int = # positives; info_j = info_det + info_int, computed
             label-blind (Theorem E(i) conditional validity).

Prediction (pre-stated): TPR(weighted) >= TPR(gated) >= TPR(plain), FDP all
controlled. Dev level, R reps, K=999.
"""
from __future__ import annotations

import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude/ch_smoke/code/simulation_v3")
sys.path.insert(0, "/home/claude/ch_smoke")
import design, generators                               # noqa: E402
from twochannel import two_channel_test                 # noqa: E402

P_TAXA = 100
R = 10
K = 999
ALPHA = 0.05
CFG = pd.read_csv(
    "/home/claude/ch_smoke/code/simulation_v3/configs/config_supplementary.csv"
).set_index("cell_id")


def run_cell(cell, rep):
    row = CFG.loc[cell]
    prm = design.params_for_cell(row)
    n = int(row["n"])
    seeds = np.random.SeedSequence(int(row["seed"])).spawn(20)
    Y, tr = generators.generate(row["mechanism"], prm, n, P_TAXA,
                                row["depth"], seed=seeds[rep])
    group = tr["group"]
    N = tr["depths"].astype(float)
    res = two_channel_test(Y, N, group, nu=N, K=K,
                           seed=[20260818, 7, cell, rep])
    A = tr["abs_da_truth"].astype(bool)
    # label-blind information covariate (upstream X_j)
    D = (Y > 0)
    qhat = np.clip(res["pi"][None, :] * 0 + np.nan, 0, 1)  # placeholder
    # reconstruct qhat info proxy: use detection variance proxy from D and
    # positives count (both label-blind summaries)
    dbar = D.mean(axis=0)
    info_det = (dbar * (1 - dbar)) * len(N)
    info_int = D.sum(axis=0).astype(float)
    info = info_det + info_int
    return res["p_comb"], A, info


def weighted_bh(p, w, alpha=ALPHA):
    m = len(p)
    w = np.asarray(w, dtype=float)
    w = np.where(w > 0, w * m / w.sum(), 0.0)      # normalize sum w = m
    padj = np.where(w > 0, p / w, np.inf)          # p_j / w_j step-up
    order = np.argsort(padj)
    thresh = alpha * np.arange(1, m + 1) / m
    passed = padj[order] <= thresh
    rej = np.zeros(m, dtype=bool)
    if passed.any():
        rej[order[: np.max(np.where(passed)[0]) + 1]] = True
    return rej


rows = []
for rep in range(R):
    t0 = time.time()
    p1, A1, i1 = run_cell(1005, rep)
    p2, A2, i2 = run_cell(1008, rep)
    p = np.concatenate([p1, p2])
    A = np.concatenate([A1, A2])
    info = np.concatenate([i1, i2])

    procs = {}
    procs["plain"] = weighted_bh(p, np.ones_like(p))
    gate = (info >= np.median(info)).astype(float)
    procs["gated"] = weighted_bh(p, gate)
    procs["weighted"] = weighted_bh(p, info)
    for name, rej in procs.items():
        nr = int(rej.sum())
        fp = int((rej & ~A).sum())
        tp = int((rej & A).sum())
        rows.append(dict(rep=rep, proc=name, nrej=nr,
                         fdp=fp / nr if nr else 0.0,
                         tpr=tp / A.sum(),
                         kept=int((({"gated": gate}.get(name,
                                    np.ones_like(p)) > 0)).sum()),
                         secs=round(time.time() - t0, 1)))
    print(rows[-3:], flush=True)

df = pd.DataFrame(rows)
df.to_csv("/home/claude/ch_smoke/weighting_detail.csv", index=False)
s = df.groupby("proc").agg(fdp=("fdp", "mean"), fdp_sem=("fdp", "sem"),
                           tpr=("tpr", "mean"), tpr_sem=("tpr", "sem"),
                           nrej=("nrej", "mean")).round(4)
s.to_csv("/home/claude/ch_smoke/weighting_summary.csv")
print(s.to_string())
