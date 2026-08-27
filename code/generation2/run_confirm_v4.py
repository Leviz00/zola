"""run_confirm_v4.py — SPEC-CH Amendment A2 confirmatory run of the v4
battery: 8 cells (2000-2007) x R=20, K=999, dual-truth scoring. Cells and
frozen parameters imported from run_pilot_v4.CELLS (authoritative table)."""
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


def one_run(args):
    cell, rep = args
    import sys
    sys.path.insert(0, "/home/claude/ch_smoke/code/simulation_v3")
    sys.path.insert(0, "/home/claude/ch_smoke")
    import numpy as np
    from scipy.stats import ranksums
    from generators_ext import generate_ext
    from twochannel import two_channel_test, bh_reject
    from run_pilot_v4 import CELLS, P, BASE_SEED, scorecard, fdp_tpr

    spec = CELLS[cell]
    t0 = time.time()
    seed = np.random.SeedSequence([BASE_SEED, cell]).spawn(20)[rep]
    Y, tr = generate_ext(spec["mech"], spec["params"], spec["n"], P,
                         spec["depth"], seed=seed)
    g = tr["group"]; N = tr["depths"].astype(float)
    A = tr["abs_da_truth"].astype(bool)
    Pt = tr.get("pres_da_truth", np.zeros(P, dtype=bool)).astype(bool)
    Ut = A | Pt
    sc = scorecard(Y, N)
    res = two_channel_test(Y, N, g, nu=N, K=999, seed=[20260819, 5, cell, rep])
    rej = bh_reject(res["p_comb"], 0.05)
    nr, fdp, tpr = fdp_tpr(rej, Ut)
    _, _, tprA = fdp_tpr(rej, A)
    _, _, tprP = fdp_tpr(rej, Pt)
    rej_det = bh_reject(res["p_det"], 0.05)
    rej_int = bh_reject(res["p_int"], 0.05)
    _, fdpD, tprD = fdp_tpr(rej_det, Pt if Pt.any() else A)
    _, fdpI, tprI = fdp_tpr(rej_int, A)
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
    return dict(cell=cell, rep=rep, **sc, nrej=nr, fdp=fdp, tpr=tpr,
                tprA=tprA, tprP=tprP, fdp_det=fdpD, tpr_det=tprD,
                fdp_int=fdpI, tpr_int=tprI,
                typeI=float((res["p_comb"][~Ut] < 0.05).mean()),
                oracle_tpr=orc, phi_hat=res["phi_hat"],
                secs=round(time.time() - t0, 1))


if __name__ == "__main__":
    from run_pilot_v4 import CELLS
    jobs = [(c, r) for c in CELLS for r in range(20)]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    rows = []
    with ctx.Pool(2) as pool:
        for i, r in enumerate(pool.imap_unordered(one_run, jobs)):
            rows.append(r)
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/{len(jobs)}] {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows).sort_values(["cell", "rep"])
    df.to_csv("/home/claude/ch_smoke/confirm_v4_detail.csv", index=False)
    g = df.groupby("cell")
    s = pd.DataFrame({
        "fdp": g["fdp"].mean(), "fdp_mcse": g["fdp"].sem(),
        "tpr": g["tpr"].mean(), "tpr_mcse": g["tpr"].sem(),
        "tprA": g["tprA"].mean(), "tprP": g["tprP"].mean(),
        "fdp_det": g["fdp_det"].mean(), "tpr_det": g["tpr_det"].mean(),
        "fdp_int": g["fdp_int"].mean(), "tpr_int": g["tpr_int"].mean(),
        "typeI": g["typeI"].mean(), "oracle": g["oracle_tpr"].mean(),
        "zero": g["zero"].mean(), "uplift": g["uplift"].mean(),
        "nrej": g["nrej"].mean()}).round(4)
    s.to_csv("/home/claude/ch_smoke/confirm_v4_summary.csv")
    print(s.to_string())
    print("DONE", flush=True)
