"""R = 2000 confirmation run for the prespecified FDR gate (null scenario).

design.py reserves ``n_replicates`` = 2000 for the final confirmation of key
cells; the prespecified screening check in ``run_weighting_check.py`` uses
R = 100, where the Monte Carlo SE of the empirical FDR under a global null
(sqrt(0.05*0.95/100) ~= 0.022) exceeds the gate tolerance 0.01.  This script
runs the single cleanest gate cell -- ``null_noninformative`` (three_layer,
structural_zero_rate 0.3, non-informative, da_fraction 0: global null, so
the empirical FDR equals the family-wise rejection rate) -- at R = 2000,
where SE ~= 0.005 and the |FDR_hat - 0.05| <= 0.01 criterion is meaningful.

Outputs ``results/null_confirmation_R2000.csv``.
Run:  ``python run_null_confirmation.py``  (from the simulation/ directory).
"""

from __future__ import annotations

import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import metrics
from run_weighting_check import (
    BASE_PARAMS,
    DEPTH,
    N_PER_GROUP,
    P_TAXA,
    detect_all,
)
import generators

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

R = 2000
PARAMS = dict(BASE_PARAMS)
PARAMS.update(structural_zero_rate=0.3, informative_zeros=False,
              da_fraction=0.0)
SEED = 20260701 * 10007 + 7000  # disjoint from the screening seeds


def run_one(args):
    rep_seed, rep = args
    Y, truth = generators.generate(
        "three_layer", PARAMS, n=N_PER_GROUP, p=P_TAXA, depths=DEPTH,
        seed=rep_seed,
    )
    rejects = detect_all(Y, truth)
    return {m: int(r.any()) for m, r in rejects.items()}  # FDP = 1{any rej}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    seeds = np.random.SeedSequence(SEED).spawn(R)
    t0 = time.time()
    with Pool(processes=2) as pool:
        rows = pool.map(run_one, [(seeds[r], r) for r in range(R)],
                        chunksize=10)
    print(f"done in {time.time() - t0:.0f}s")
    df = pd.DataFrame(rows)
    out = []
    for m in df.columns:
        fdr_hat, fdr_se, _ = metrics.empirical_rate(df[m].to_numpy())
        out.append(dict(scenario="null_noninformative", method=m, R=R,
                        emp_fdr=fdr_hat, fdr_mc_se=fdr_se,
                        gate_pass=abs(fdr_hat - 0.05) <= 0.01 + 1e-12))
    out = pd.DataFrame(out)
    path = os.path.join(RESULTS_DIR, "null_confirmation_R2000.csv")
    out.to_csv(path, index=False)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(out.to_string(index=False))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
