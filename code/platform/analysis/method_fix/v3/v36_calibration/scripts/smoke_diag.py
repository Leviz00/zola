"""smoke_diag.py — v3.6 冒烟 + 池化可交换性诊断（4 格 × rep0）。"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

SIM_V3 = "/mnt/agents/output/code/simulation_v3"
V36 = "/mnt/agents/output/analysis/method_fix/v3/v36_calibration"
CONFIG = SIM_V3 + "/configs/config_supplementary.csv"
sys.path.insert(0, V36)
sys.path.insert(1, SIM_V3)
import design  # noqa: E402
import generators  # noqa: E402
from perm_glm import calibrated_test, pool_diagnostic  # noqa: E402

CELLS = [1004, 1005, 1007, 1009]


def replay(cell, rep=0):
    cfg = pd.read_csv(CONFIG)
    row = cfg[cfg.cell_id == cell].iloc[0]
    seeds = np.random.SeedSequence(int(row["seed"])).spawn(20)
    prm = design.params_for_cell(row)
    prm["effect_mode"] = "absolute"
    Y, truth = generators.generate(row["mechanism"], prm, n=int(row["n"]),
                                   p=100, depths=int(row["depth"]),
                                   seed=seeds[rep])
    return np.asarray(Y, float), truth


def main():
    for cell in CELLS:
        Y, truth = replay(cell)
        group = truth["group"]
        N = truth["depths"].astype(float)
        da = truth["abs_da_truth"]
        t0 = time.time()
        r = calibrated_test(Y, group, N, W=None, K=20)
        el = time.time() - t0
        rej = r["reject"]
        fp = int((rej & ~da).sum()); tp = int((rej & da).sum())
        fdp = fp / (fp + tp) if fp + tp else 0.0
        tpr = tp / max(int(da.sum()), 1)
        mu_hat = Y.sum(0) / np.maximum(N.sum(), 1)
        d = pool_diagnostic(r["null"], mu_hat, r["alpha_hat"])
        print(f"cell {cell}: n_rej={rej.sum()} FDP={fdp:.3f} TPR={tpr:.3f} "
              f"FWER(LKO)={r['fwer']:.2f} t={el:.0f}s")
        print(f"  pool: null_mean={d['null_mean']:.2f} (chi2=1) "
              f"q99={d['null_q99']:.2f} (chi2 6.63) "
              f"perTaxonMeanSD={d['per_taxon_null_mean_sd']:.2f} "
              f"(MC {d['mc_sd_expected']:.2f}) "
              f"r(logmu)={d.get('corr_nullmean_logmu', np.nan):.3f} "
              f"r(logalpha)={d.get('corr_nullmean_logalpha', np.nan):.3f} "
              f"pooling_ok={d['pooling_ok']}")


if __name__ == "__main__":
    main()
