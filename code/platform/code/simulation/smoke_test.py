"""End-to-end smoke test: generate -> truth -> reference detector -> metrics.

Runs a 2-cell mini-grid (informative structural zeros on/off) x R = 50
replicates through the full chain with the ``three_layer`` generator and a
deliberately simple *reference detector* (self-contained placeholder for the
real estimators, which live in the separate estimation prototype):

* DA detection : per-taxon Welch t-test on log10 relative abundances
                 (pseudo-count 0.5), Benjamini-Hochberg at 0.05;
* zero source  : plug-in score  P(structural | Y_ij = 0) ~= 1 - (1-m_j)^{N_i}
                 with m_j the control-group mean relative abundance
                 (probability that a *present* taxon stays unobserved at
                 depth N_i, inverted);
* effect size  : difference of group means of log2 relative abundance.

Prints a results table and writes ``results/smoke_results.csv``.
Run:  ``python smoke_test.py``  (from the simulation/ directory).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

import generators
import metrics

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

R_REPLICATES = 50
ALPHA = 0.05
P_TAXA = 100
N_PER_GROUP = 50
DEPTH = 20000

CELLS = [
    {
        "cell": "informative_on",
        "params": {
            "structural_zero_rate": 0.1,
            "informative_zeros": True,
            "effect_size": 2.0,
            # low overdispersion so the reference detector has non-
            # degenerate power at n = 50/group (smoke test only)
            "dispersion": 100.0,
            # high prevalence: the reference t-test detects compositional
            # effects only where taxa are present (genus-level regime)
            "base_prevalence": 0.9,
        },
        "seed": 20260701 * 10007 + 0,
    },
    {
        "cell": "informative_off",
        "params": {
            "structural_zero_rate": 0.1,
            "informative_zeros": False,
            "effect_size": 2.0,
            "dispersion": 100.0,
            "base_prevalence": 0.9,
        },
        "seed": 20260701 * 10007 + 1,
    },
]


# ---------------------------------------------------------------------------
# reference detector (placeholder-quality, documented as such)
# ---------------------------------------------------------------------------

def bh_reject(pvals, alpha=ALPHA):
    """Benjamini-Hochberg rejection mask at level ``alpha``."""
    pvals = np.asarray(pvals, dtype=float)
    p = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresh = alpha * np.arange(1, p + 1) / p
    below = ranked <= thresh
    reject = np.zeros(p, dtype=bool)
    if below.any():
        k = np.max(np.where(below))
        reject[order[: k + 1]] = True
    return reject


def reference_detector(Y, group):
    """Return (reject mask, log2fc hat, zero-source score matrix)."""
    group = np.asarray(group)
    totals = Y.sum(axis=1, keepdims=True)
    rel = (Y + 0.5) / (totals + 0.5 * Y.shape[1])
    log_rel = np.log10(rel)
    a = log_rel[group == 0]
    b = log_rel[group == 1]
    tstat, pvals = stats.ttest_ind(b, a, axis=0, equal_var=False)
    pvals = np.where(np.isnan(pvals), 1.0, pvals)
    reject = bh_reject(pvals)

    log2fc = (np.log2(rel[group == 1]).mean(axis=0)
              - np.log2(rel[group == 0]).mean(axis=0))

    # zero-source plug-in score using control-group mean abundance
    m_hat = rel[group == 0].mean(axis=0)
    n_i = totals[:, 0].astype(float)
    with np.errstate(invalid="ignore"):
        p_sampling_zero = np.exp(n_i[:, None] * np.log1p(-np.clip(m_hat, 0, 1)[None, :]))
    score_structural = 1.0 - p_sampling_zero
    return reject, log2fc, np.clip(score_structural, 0.0, 1.0)


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

def run_cell(cell, R=R_REPLICATES, n=N_PER_GROUP, p=P_TAXA, depth=DEPTH):
    child_seeds = np.random.SeedSequence(cell["seed"]).spawn(R)
    rows = []
    for r in range(R):
        Y, truth = generators.generate(
            "three_layer", cell["params"], n=n, p=p, depths=depth,
            seed=child_seeds[r],
        )
        reject, log2fc_hat, score = reference_detector(Y, truth["group"])

        fdp_r, n_rej = metrics.fdp(reject, truth["da_taxa"])
        tpr_r, _ = metrics.tpr(reject, truth["da_taxa"])
        log2fc_true = np.where(
            truth["da_taxa"], np.log2(truth["effect_size"]), 0.0
        )
        bias = metrics.effect_size_bias(
            log2fc_hat, log2fc_true, mask=truth["da_taxa"]
        )
        zs = metrics.zero_source_metrics(
            score, truth["structural_zeros"], Y
        )
        rows.append(
            dict(cell=cell["cell"], rep=r, fdp=fdp_r, tpr=tpr_r,
                 n_rej=n_rej, bias_da=bias["bias"],
                 auc=zs["auc"], brier=zs["brier"])
        )
    return rows


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_rows = []
    for cell in CELLS:
        all_rows.extend(run_cell(cell))
    df = pd.DataFrame(all_rows)

    summary = []
    for cell_name, g in df.groupby("cell"):
        fdr_hat, fdr_se, _ = metrics.empirical_rate(g["fdp"].to_numpy())
        tpr_hat, tpr_se, _ = metrics.empirical_rate(g["tpr"].to_numpy())
        auc_hat, auc_se, _ = metrics.empirical_rate(g["auc"].to_numpy())
        summary.append(
            dict(cell=cell_name, R=len(g),
                 emp_fdr=fdr_hat, fdr_mc_se=fdr_se,
                 tpr=tpr_hat, tpr_mc_se=tpr_se,
                 bias_da=g["bias_da"].mean(),
                 auc_zero_src=auc_hat, auc_mc_se=auc_se,
                 brier=g["brier"].mean())
        )
    out = pd.DataFrame(summary)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("smoke test: three_layer, R =", R_REPLICATES,
          ", n =", N_PER_GROUP, "/group, p =", P_TAXA,
          ", depth =", DEPTH)
    print(out.to_string(index=False))
    path = os.path.join(RESULTS_DIR, "smoke_results.csv")
    out.to_csv(path, index=False)
    df.to_csv(os.path.join(RESULTS_DIR, "smoke_replicates.csv"), index=False)
    print(f"\nwrote {path}")
    return out


if __name__ == "__main__":
    main()
