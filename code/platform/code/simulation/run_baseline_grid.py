"""Run the pure-Python TSS+Wilcoxon+BH baseline over the 48-cell fractional grid.

First end-to-end baseline of the simulation study (overall_review.md C2(a)):
the ``baselines.md`` section-7 baseline implemented in pure Python
(``baselines_py.tss_wilcoxon``) is evaluated on every cell of
``configs/config_fractional.csv`` with R = ``n_replicates_screen`` = 100
replicates per cell (design.py refinement, pilot-justified screening value).

Statistic definitions (preregistration, paper section 6.4): per-replicate
FDP (0 when no rejections) -> empirical FDR = mean FDP with MC SE
sd(FDP)/sqrt(R); per-replicate TPR -> cell power = mean TPR.  All summary
numbers are recomputable from the per-replicate CSV.

Outputs (results/)
------------------
``baseline_tss_wilcoxon_replicates.csv`` : one row per cell x replicate
``baseline_tss_wilcoxon_grid.csv``       : one summary row per cell

Run:  ``python run_baseline_grid.py``  (from the simulation/ directory).
"""

from __future__ import annotations

import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import design
import generators
import metrics
from baselines_py import ALPHA, tss_wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
CONFIG_CSV = os.path.join(HERE, "configs", "config_fractional.csv")

P_TAXA = 100  # taxa per dataset (same as smoke_test.py; not a design factor)


def run_one_replicate(args):
    """Generate one dataset and run the baseline; returns per-replicate row."""
    cell_id, mechanism, params, n, depth, rep_seed, rep = args
    Y, truth = generators.generate(
        mechanism, params, n=n, p=P_TAXA, depths=depth, seed=rep_seed
    )
    res = tss_wilcoxon(Y, truth["group"], alpha=ALPHA)
    fdp_r, n_rej = metrics.fdp(res["reject"], truth["da_taxa"])
    tpr_r, _ = metrics.tpr(res["reject"], truth["da_taxa"])
    return dict(cell_id=cell_id, rep=rep, fdp=fdp_r, tpr=tpr_r, n_rej=n_rej)


def cell_tasks(row, R):
    child_seeds = np.random.SeedSequence(int(row["seed"])).spawn(R)
    params = design.params_for_cell(row)
    return [
        (int(row["cell_id"]), row["mechanism"], params, int(row["n"]),
         int(row["depth"]), child_seeds[r], r)
        for r in range(R)
    ]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg = pd.read_csv(CONFIG_CSV)
    R = int(cfg["n_replicates_screen"].iloc[0])
    assert (cfg["n_replicates_screen"] == R).all()

    tasks = []
    for _, row in cfg.iterrows():
        tasks.extend(cell_tasks(row, R))
    print(f"fractional grid: {len(cfg)} cells x R = {R} replicates "
          f"= {len(tasks)} datasets, p = {P_TAXA}, alpha = {ALPHA}")

    t0 = time.time()
    with Pool(processes=2) as pool:
        rows = pool.map(run_one_replicate, tasks, chunksize=20)
    rep_df = pd.DataFrame(rows)
    print(f"done in {time.time() - t0:.0f}s")

    summary = []
    for cell_id, g in rep_df.groupby("cell_id"):
        fdr_hat, fdr_se, _ = metrics.empirical_rate(g["fdp"].to_numpy())
        tpr_hat, tpr_se, _ = metrics.empirical_rate(g["tpr"].to_numpy())
        summary.append(dict(
            cell_id=cell_id, R=len(g),
            emp_fdr=fdr_hat, fdr_mc_se=fdr_se,
            power=tpr_hat, power_mc_se=tpr_se,
            mean_rejections=g["n_rej"].mean(),
        ))
    out = cfg.merge(pd.DataFrame(summary), on="cell_id")

    rep_path = os.path.join(RESULTS_DIR, "baseline_tss_wilcoxon_replicates.csv")
    grid_path = os.path.join(RESULTS_DIR, "baseline_tss_wilcoxon_grid.csv")
    rep_df.to_csv(rep_path, index=False)
    out.to_csv(grid_path, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    cols = ["mechanism", "informative_zeros", "structural_zero_rate",
            "emp_fdr", "fdr_mc_se", "power"]
    print("\nby mechanism x informative_zeros (mean over cells):")
    print(out.groupby(["mechanism", "informative_zeros"])[
        ["emp_fdr", "power", "mean_rejections"]].mean().to_string())
    print(f"\nwrote {grid_path}\nwrote {rep_path}")
    return out


if __name__ == "__main__":
    main()
