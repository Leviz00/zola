"""run_perm_v3.py — v3.1 置换诊断：量化组条件权重泄漏幅度。

cells {6,22} × rep0 × 10 次 group 标签置换。每次置换：
  重拟合 W1（group 条件、phi_known=φ̂_cnt 连贯）权重 →
  exclusion_wilcoxon(Y, group_perm, W) → 拒绝率（100 类群中被拒比例）。
真 γ1=0（cell22 非 informative）/ 置换破坏真关联（cell6 informative? 见
config）时，无泄漏通道的权重应给出 ≈α=0.05 的拒绝率；γ̂1 噪声→不对称
剔除的通道越大，拒绝率越高。

计数块不含 group，全部置换共用同一次精炼（与 run_one_v3 同一 φ̂_cnt）。
用法: python3 run_perm_v3.py --cell 22 --rep 0 --nperm 10 --out perm.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

SIM_DIR = "/mnt/agents/output/code/simulation"
EST_V3 = "/mnt/agents/output/code/estimation_v3"
CONFIG_CSV = os.path.join(SIM_DIR, "configs", "config_fractional.csv")
P_TAXA = 100
R_SPAWN = 20

sys.path.insert(0, EST_V3)
sys.path.insert(1, SIM_DIR)
import design  # noqa: E402
import generators  # noqa: E402
import composite_likelihood as est_cl  # noqa: E402
import composite_likelihood_cov as est_clcov  # noqa: E402
import composite_likelihood_count as est_clc  # noqa: E402
import posterior as est_post  # noqa: E402
from weighting import exclusion_wilcoxon, validate_weights  # noqa: E402
from run_one_v3 import count_refine, LAM_MAIN, LAM_RETRY, DET_MAXITER  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int, required=True)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--nperm", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    cfg = pd.read_csv(CONFIG_CSV)
    row = cfg[cfg.cell_id == args.cell].iloc[0]
    child_seeds = np.random.SeedSequence(int(row["seed"])).spawn(R_SPAWN)
    params = design.params_for_cell(row)
    Y, truth = generators.generate(
        row["mechanism"], params, n=int(row["n"]), p=P_TAXA,
        depths=int(row["depth"]), seed=child_seeds[args.rep])
    group = np.asarray(truth["group"])
    Y = np.asarray(Y, dtype=float)
    D = (Y > 0).astype(float)
    N = Y.sum(axis=1).astype(float)
    n, p = Y.shape

    t0 = time.time()
    f_det = est_cl.fit_composite(D, N, phi_known=None, multi_start=False,
                                 maxiter=DET_MAXITER)
    f_cnt = count_refine(est_clc, Y, N, f_det, LAM_MAIN, p)
    if f_cnt["on_boundary"]:
        f2 = count_refine(est_clc, Y, N, f_det, LAM_RETRY, p)
        if not f2["on_boundary"]:
            f_cnt = f2
    phi_hat, theta_hat = f_cnt["phi"], f_cnt["theta"]
    print(f"count block done: phi={phi_hat:.3g} ({time.time()-t0:.0f}s)",
          flush=True)

    rows = []
    for perm in range(args.nperm):
        rng = np.random.default_rng(91000 + perm)
        gperm = rng.permutation(group)
        Wd = np.column_stack([np.ones(n), gperm.astype(float)])
        t0 = time.time()
        f_w1 = est_clcov.fit_composite_cov(D, Wd, N, phi_known=phi_hat)
        P = est_post.zero_source_posterior_cov(f_w1["Gamma"], Wd,
                                               theta_hat, phi_hat, N)
        W = validate_weights(P, Y)
        rej = exclusion_wilcoxon(Y, gperm, W)["reject"]
        rows.append(dict(
            cell_id=args.cell, rep=args.rep, perm=perm,
            n_rej=int(rej.sum()), frac_rej=float(rej.mean()),
            abs_gamma1_mean=float(np.abs(f_w1["Gamma"][:, 1]).mean()),
            t_fit=time.time() - t0))
        print(f"perm {perm}: n_rej={rej.sum()} "
              f"frac={rej.mean():.3f} ({rows[-1]['t_fit']:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"saved {args.out}; mean frac_rej={df.frac_rej.mean():.3f}")


if __name__ == "__main__":
    main()
