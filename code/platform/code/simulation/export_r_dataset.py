"""Export simulated datasets for the R baselines (baselines.md exchange format).

Writes, per scenario, the Python -> R exchange files of baselines.md:
``cell<name>_rep<r>_counts.csv`` (samples x taxa raw integer counts) and
``cell<name>_rep<r>_meta.csv`` (sample_id, group), plus a
``cell<name>_rep<r>_truth.csv`` (taxon, da) so the returned discoveries can
be scored against the simulation truth.

Scenarios mirror run_weighting_check.py (three_layer, informative structural
zeros on/off, structural_zero_rate 0.3) so the R smoke results are directly
comparable with the pure-Python track.

Run:  ``python export_r_dataset.py``  (from the simulation/ directory).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import generators

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results", "r_exchange")

SCENARIOS = {
    "inf_on": dict(structural_zero_rate=0.3, informative_zeros=True),
    "inf_off": dict(structural_zero_rate=0.3, informative_zeros=False),
}
BASE_PARAMS = dict(effect_size=2.0, dispersion=100.0, base_prevalence=0.9)
N_PER_GROUP = 100
P_TAXA = 100
DEPTH = 20000
BASE_SEED = 20260701 * 10007 + 9000  # disjoint from grid/check seeds


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for s_idx, (name, sp) in enumerate(sorted(SCENARIOS.items())):
        params = dict(BASE_PARAMS)
        params.update(sp)
        seed = np.random.SeedSequence(BASE_SEED + s_idx).spawn(1)[0]
        Y, truth = generators.generate(
            "three_layer", params, n=N_PER_GROUP, p=P_TAXA, depths=DEPTH,
            seed=seed,
        )
        sample_ids = [f"s{i:03d}" for i in range(Y.shape[0])]
        taxa = [f"taxon{j:03d}" for j in range(Y.shape[1])]
        stem = f"cell{name}_rep0"
        pd.DataFrame(Y, index=sample_ids, columns=taxa).to_csv(
            os.path.join(OUT_DIR, f"{stem}_counts.csv"))
        pd.DataFrame({"sample_id": sample_ids,
                      "group": truth["group"]}).to_csv(
            os.path.join(OUT_DIR, f"{stem}_meta.csv"), index=False)
        pd.DataFrame({"taxon": taxa,
                      "da": truth["da_taxa"].astype(int)}).to_csv(
            os.path.join(OUT_DIR, f"{stem}_truth.csv"), index=False)
        print(f"{name}: zeros={np.mean(Y == 0):.3f} "
              f"n_da={int(truth['da_taxa'].sum())} -> {stem}_*.csv")


if __name__ == "__main__":
    main()
