"""run_fix_v4.py — SPEC-CH Amendment A3 rerun of the four flagged cells.

Fixes (theory-internal, not ad hoc):
  * intensity offset -> masked median-ratio (anchor assumption declared);
    design-depth offset becomes the sensitivity arm. Addresses the
    compositional spillover of presence-DA cells (2001/2002) = the P0
    normalization question answered inside the pipeline.
  * permutation stratified by realized-library deciles when the
    library~group diagnostic fires (always applied here for 2005) =
    Theorem A Remark(i) conditioning. Addresses the composite-null
    leakage measured on 2005 (typeI 0.112 -> expect ~0.05).
  * 2006 base_prevalence 0.5 -> 0.6 (oracle detectability >= 0.5).
Cells 2000/2003/2004/2007 keep their A2 results (passed).
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, time
import numpy as np
import pandas as pd
import multiprocessing as mp

sys.path.insert(0, "/home/claude/ch_smoke/code/simulation_v3")
sys.path.insert(0, "/home/claude/ch_smoke")

FIX_CELLS = {
    2001: dict(mech="three_layer_real", depth=20000, n=300,
               params=dict(da_fraction=0.0, presence_da_fraction=0.10,
                           presence_effect_or=0.25)),
    2002: dict(mech="three_layer_real", depth=20000, n=300,
               params=dict(effect_size=2.0, da_fraction=0.05,
                           presence_da_fraction=0.05,
                           presence_effect_or=0.25)),
    2005: dict(mech="beta_binomial", depth=20000, n=300,
               params=dict(effect_size=4.0, structural_zero_rate=0.0,
                           informative_zeros=False, dispersion=3.0,
                           base_prevalence=0.5, effect_mode="absolute",
                           depth_cv=1.0)),
    2006: dict(mech="zigdm_like", depth=20000, n=300,
               params=dict(effect_size=4.0, structural_zero_rate=0.0,
                           informative_zeros=False, dispersion=15.0,
                           base_prevalence=0.6, effect_mode="absolute",
                           depth_cv=1.0)),
}


def one_run(args):
    cell, rep = args
    import sys
    sys.path.insert(0, "/home/claude/ch_smoke/code/simulation_v3")
    sys.path.insert(0, "/home/claude/ch_smoke")
    import numpy as np
    from scipy.stats import ranksums, spearmanr
    from generators_ext import generate_ext
    from twochannel import (two_channel_test, bh_reject, median_ratio_offset)

    spec = FIX_CELLS[cell]
    t0 = time.time()
    seed = np.random.SeedSequence([20260819, cell]).spawn(20)[rep]
    Y, tr = generate_ext(spec["mech"], spec["params"], spec["n"], 100,
                         spec["depth"], seed=seed)
    g = tr["group"]; N = tr["depths"].astype(float)
    A = tr["abs_da_truth"].astype(bool)
    Pt = tr.get("pres_da_truth", np.zeros(100, dtype=bool)).astype(bool)
    Ut = A | Pt

    lib = Y.sum(axis=1).astype(float)
    # library~group diagnostic (Theorem A condition check)
    lr = ranksums(lib[g == 1], lib[g == 0]).pvalue
    strata = None
    if lr < 0.10:                       # diagnostic fires -> stratify
        strata = np.digitize(lib, np.quantile(lib, np.linspace(0, 1, 11)[1:-1]))
    nu = median_ratio_offset(Y)

    res = two_channel_test(Y, N, g, nu=nu, K=999,
                           seed=[20260819, 6, cell, rep], strata=strata)
    rej = bh_reject(res["p_comb"], 0.05)

    def ft(rej, truth):
        nr = int(rej.sum()); fp = int((rej & ~truth).sum())
        tp = int((rej & truth).sum())
        return nr, (fp / nr if nr else 0.0), (tp / truth.sum() if truth.sum() else np.nan)

    nr, fdp, tpr = ft(rej, Ut)
    _, _, tprA = ft(rej, A)
    _, _, tprP = ft(rej, Pt)
    rej_det = bh_reject(res["p_det"], 0.05)
    rej_int = bh_reject(res["p_int"], 0.05)
    _, fdpD, tprD = ft(rej_det, Pt if Pt.any() else A)
    _, fdpI, tprI = ft(rej_int, A)
    orc = np.nan
    if cell in (2005, 2006) and A.any():
        hits = 0
        pres = tr["presence"]
        for j in np.where(A)[0]:
            m = pres[:, j]
            if (m & (g == 1)).sum() >= 3 and (m & (g == 0)).sum() >= 3:
                ra = Y[m, j] / N[m]
                pv = ranksums(ra[g[m] == 1], ra[g[m] == 0]).pvalue
                hits += pv < 0.05 / max(A.sum(), 1)
        orc = hits / A.sum()
    return dict(cell=cell, rep=rep, nrej=nr, fdp=fdp, tpr=tpr, tprA=tprA,
                tprP=tprP, fdp_det=fdpD, tpr_det=tprD, fdp_int=fdpI,
                tpr_int=tprI,
                typeI=float((res["p_comb"][~Ut] < 0.05).mean()),
                oracle_tpr=orc, strata_used=strata is not None,
                lib_p=lr, secs=round(time.time() - t0, 1))


if __name__ == "__main__":
    jobs = [(c, r) for c in FIX_CELLS for r in range(20)]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    rows = []
    with ctx.Pool(2) as pool:
        for i, r in enumerate(pool.imap_unordered(one_run, jobs)):
            rows.append(r)
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(jobs)}] {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows).sort_values(["cell", "rep"])
    df.to_csv("/home/claude/ch_smoke/fix_v4_detail.csv", index=False)
    g = df.groupby("cell")
    s = pd.DataFrame({"fdp": g["fdp"].mean(), "fdp_mcse": g["fdp"].sem(),
                      "tpr": g["tpr"].mean(), "tprA": g["tprA"].mean(),
                      "tprP": g["tprP"].mean(), "typeI": g["typeI"].mean(),
                      "oracle": g["oracle_tpr"].mean(),
                      "strata_rate": g["strata_used"].mean(),
                      "fdp_det": g["fdp_det"].mean(),
                      "tpr_det": g["tpr_det"].mean()}).round(4)
    s.to_csv("/home/claude/ch_smoke/fix_v4_summary.csv")
    print(s.to_string()); print("DONE", flush=True)
