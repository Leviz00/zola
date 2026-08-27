"""Prespecified check of posterior-weighted DA (paper section 6.4, feasibility).

Runs the two prespecified scenarios of ``weighting.py`` on the
``three_layer`` baseline generator:

* NON-INFORMATIVE structural zeros (``informative_zeros = False``, absence
  independent of group): the weighted methods must NOT break FDR.
  Criterion: |FDR_hat - 0.05| <= 0.01 at R = 100 (n_replicates_screen).
  The Monte Carlo SE of FDR_hat at R = 100 is itself ~0.01, so every
  decision is reported with its SE; the check is a screening gate, not the
  final confirmation (which uses R = 2000 per design.py).
* INFORMATIVE structural zeros (absence aligned with the case group):
  quantitative report of FDR improvement (FDR_naive - FDR_weighted) and
  power loss (TPR_naive - TPR_weighted) of the weighted variants.

Detectors compared (all at the locked alpha = 0.05):
    naive_welch_t            unweighted anchor (smoke-test detector family)
    tss_wilcoxon             unweighted naive baseline (baselines.md sec. 7)
    weighted_welch_t         + oracle weights      (upper bound of benefit)
    weighted_welch_t         + placeholder weights (estimator stand-in)
    exclusion_wilcoxon       + oracle weights
    exclusion_wilcoxon       + placeholder weights

Statistic definitions (preregistration): per-replicate FDP (0 if no
rejections); empirical FDR = mean of per-replicate FDPs (*not* pooled
rejections); power = mean of per-replicate TPRs; MC SE = sd/sqrt(R).

Outputs (results/)
------------------
``weighting_check_replicates.csv`` : one row per scenario x method x rep
``weighting_check_summary.csv``    : per scenario x method FDR/power + the
                                     prespecified improvement/loss contrasts

Run:  ``python run_weighting_check.py``  (from the simulation/ directory).
"""

from __future__ import annotations

import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import generators
import metrics
from baselines_py import ALPHA, naive_welch_t, tss_wilcoxon
from weighting import (
    exclusion_wilcoxon,
    oracle_weights,
    placeholder_weights,
    weighted_welch_t,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

R_REPLICATES = 100  # design.py n_replicates_screen (screening value)
P_TAXA = 100
N_PER_GROUP = 100
DEPTH = 20000
FDR_TOL = 0.01  # prespecified: |FDR_hat - 0.05| <= FDR_TOL (non-informative)

# Scenario grid.  Regimes (all three_layer, n = 100/group, depth 20000):
# * sz10/sz30 (effect 2.0, da_fraction 0.1, dispersion 100, prevalence 0.9):
#   the smoke-test regime.  WARNING: with 10% DA mass at fold-change 2 the
#   TSS/log-rel values are compositionally biased and *every* unweighted
#   method already violates the FDR gate (compositional confound, not a
#   zero-layer effect); these cells quantify the informative-zero
#   improvement only.
# * mild_*: effect 1.5, da_fraction 0.05 -- compositionally mild, so the
#   unweighted anchors pass the gate and the prespecified "weight injection
#   must not break FDR" check is unconfounded.
# * null_*: da_fraction 0 (global null, no compositional confound at all);
#   empirical FDR here equals the family-wise rejection rate (any rejection
#   is false).  The cleanest calibration probe of weight injection.
SCENARIOS = {
    "noninformative_sz10": dict(structural_zero_rate=0.1,
                                informative_zeros=False),
    "noninformative_sz30": dict(structural_zero_rate=0.3,
                                informative_zeros=False),
    "informative_sz10": dict(structural_zero_rate=0.1,
                             informative_zeros=True),
    "informative_sz30": dict(structural_zero_rate=0.3,
                             informative_zeros=True),
    "null_noninformative": dict(structural_zero_rate=0.3,
                                informative_zeros=False, da_fraction=0.0),
    "null_informative": dict(structural_zero_rate=0.3,
                             informative_zeros=True, da_fraction=0.0),
    "mild_noninformative_sz30": dict(structural_zero_rate=0.3,
                                     informative_zeros=False,
                                     effect_size=1.5, da_fraction=0.05),
    "mild_informative_sz30": dict(structural_zero_rate=0.3,
                                  informative_zeros=True,
                                  effect_size=1.5, da_fraction=0.05),
}
BASE_PARAMS = dict(effect_size=2.0, dispersion=100.0, base_prevalence=0.9)
BASE_SEED = 20260701 * 10007 + 5000  # disjoint from grid/smoke seeds

METHODS = [
    "naive_welch_t",
    "tss_wilcoxon",
    "weighted_welch_t_oracle",
    "weighted_welch_t_placeholder",
    "exclusion_wilcoxon_oracle",
    "exclusion_wilcoxon_placeholder",
]


def detect_all(Y, truth):
    """Run every detector on one dataset; returns {method: reject mask}."""
    group = truth["group"]
    out = {
        "naive_welch_t": naive_welch_t(Y, group)["reject"],
        "tss_wilcoxon": tss_wilcoxon(Y, group)["reject"],
    }
    W_or = oracle_weights(Y, truth)
    W_ph = placeholder_weights(Y, group)
    out["weighted_welch_t_oracle"] = weighted_welch_t(Y, group, W_or)["reject"]
    out["weighted_welch_t_placeholder"] = weighted_welch_t(
        Y, group, W_ph)["reject"]
    out["exclusion_wilcoxon_oracle"] = exclusion_wilcoxon(
        Y, group, W_or)["reject"]
    out["exclusion_wilcoxon_placeholder"] = exclusion_wilcoxon(
        Y, group, W_ph)["reject"]
    return out


def run_one(args):
    scenario, params, rep_seed, rep = args
    Y, truth = generators.generate(
        "three_layer", params, n=N_PER_GROUP, p=P_TAXA, depths=DEPTH,
        seed=rep_seed,
    )
    rejects = detect_all(Y, truth)
    rows = []
    for method, reject in rejects.items():
        fdp_r, n_rej = metrics.fdp(reject, truth["da_taxa"])
        tpr_r, _ = metrics.tpr(reject, truth["da_taxa"])
        rows.append(dict(scenario=scenario, method=method, rep=rep,
                         fdp=fdp_r, tpr=tpr_r, n_rej=n_rej))
    return rows


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tasks = []
    for s_idx, (name, sp) in enumerate(sorted(SCENARIOS.items())):
        params = dict(BASE_PARAMS)
        params.update(sp)
        seeds = np.random.SeedSequence(BASE_SEED + s_idx).spawn(R_REPLICATES)
        for r in range(R_REPLICATES):
            tasks.append((name, params, seeds[r], r))
    print(f"weighting check: {len(SCENARIOS)} scenarios x {len(METHODS)} "
          f"methods x R = {R_REPLICATES} = "
          f"{len(tasks) * len(METHODS)} detector runs "
          f"(three_layer, n={N_PER_GROUP}/group, p={P_TAXA}, depth={DEPTH}, "
          f"alpha={ALPHA})")

    t0 = time.time()
    with Pool(processes=2) as pool:
        nested = pool.map(run_one, tasks, chunksize=5)
    rep_df = pd.DataFrame([row for rows in nested for row in rows])
    print(f"done in {time.time() - t0:.0f}s")

    summary = []
    for (scenario, method), g in rep_df.groupby(["scenario", "method"]):
        fdr_hat, fdr_se, _ = metrics.empirical_rate(g["fdp"].to_numpy())
        tpr_hat, tpr_se, _ = metrics.empirical_rate(g["tpr"].to_numpy())
        summary.append(dict(
            scenario=scenario, method=method, R=len(g),
            emp_fdr=fdr_hat, fdr_mc_se=fdr_se,
            power=tpr_hat, power_mc_se=tpr_se,
            mean_rejections=g["n_rej"].mean(),
        ))
    out = pd.DataFrame(summary)

    # prespecified contrasts: within each scenario, vs. naive_welch_t
    base = out[out["method"] == "naive_welch_t"].set_index("scenario")
    out["fdr_improvement_vs_naive"] = out.apply(
        lambda r: base.loc[r["scenario"], "emp_fdr"] - r["emp_fdr"], axis=1)
    out["power_loss_vs_naive"] = out.apply(
        lambda r: base.loc[r["scenario"], "power"] - r["power"], axis=1)
    # prespecified gate: non-informative scenarios (including the global
    # null and the compositionally mild regime) must keep FDR at 0.05
    non_inf = out["scenario"].str.contains("noninformative")
    out["fdr_gate_pass"] = np.where(
        non_inf, (out["emp_fdr"] - ALPHA).abs() <= FDR_TOL + 1e-12, np.nan)

    rep_path = os.path.join(RESULTS_DIR, "weighting_check_replicates.csv")
    sum_path = os.path.join(RESULTS_DIR, "weighting_check_summary.csv")
    rep_df.to_csv(rep_path, index=False)
    out.to_csv(sum_path, index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(out.sort_values(["scenario", "method"]).to_string(index=False))
    print(f"\nwrote {sum_path}\nwrote {rep_path}")
    return out


if __name__ == "__main__":
    main()
