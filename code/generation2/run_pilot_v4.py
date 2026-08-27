"""run_pilot_v4.py — pilot (R=3, K=499) of the Amendment-A2 candidate cells.

Cells 2000-2007: realistic backbone x {intensity / presence / mixed /
gradient}, adversarial-but-detectable BB & ZIGDM, and the depth-cv twin of
cell 1000. For each: realism scorecard vs the empirical yardstick, dual-truth
two-channel metrics, oracle-exclusion detectability proxy (Wilcoxon on
relative abundance over truly-present cells) for the adversarial cells.
Dev level; acceptance bands in AMENDMENT A2 (SPEC_CH).
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, time
import numpy as np
import pandas as pd
from scipy.stats import ranksums

sys.path.insert(0, "/home/claude/ch_smoke/code/simulation_v3")
sys.path.insert(0, "/home/claude/ch_smoke")
from generators_ext import generate_ext                 # noqa: E402
from twochannel import two_channel_test, bh_reject      # noqa: E402

P = 100
K = 499
R = 3
ALPHA = 0.05

CELLS = {
    2000: dict(mech="three_layer_real", depth=20000, n=300,   # REAL-INT
               params=dict(effect_size=2.0, da_fraction=0.10,
                           presence_da_fraction=0.0)),
    2001: dict(mech="three_layer_real", depth=20000, n=300,   # REAL-PRES
               params=dict(da_fraction=0.0, presence_da_fraction=0.10,
                           presence_effect_or=0.25)),
    2002: dict(mech="three_layer_real", depth=20000, n=300,   # REAL-MIX
               params=dict(effect_size=2.0, da_fraction=0.05,
                           presence_da_fraction=0.05,
                           presence_effect_or=0.25)),
    2003: dict(mech="three_layer_real", depth=20000, n=100,   # REAL-HARD
               params=dict(effect_size=2.0, da_fraction=0.10)),
    2004: dict(mech="three_layer_real", depth=20000, n=300,   # REAL-GRAD
               params=dict(effect_size=1.5, da_fraction=0.10)),
    2005: dict(mech="beta_binomial", depth=20000, n=300,
               params=dict(effect_size=4.0, structural_zero_rate=0.0,
                           informative_zeros=False, dispersion=3.0,
                           base_prevalence=0.5, effect_mode="absolute",
                           depth_cv=1.0)),
    2006: dict(mech="zigdm_like", depth=20000, n=300,
               params=dict(effect_size=4.0, structural_zero_rate=0.0,
                           informative_zeros=False, dispersion=15.0,
                           base_prevalence=0.5, effect_mode="absolute",
                           depth_cv=1.0)),
    2007: dict(mech="three_layer", depth=5000, n=100,     # cv-twin of 1000
               params=dict(effect_size=2.0, structural_zero_rate=0.1,
                           informative_zeros=False, dispersion=3000.0,
                           effect_mode="absolute", depth_cv=1.0)),
}
BASE_SEED = 20260819


def scorecard(Y, depths):
    Yf = Y.astype(float)
    zr = (Yf == 0).mean(); dcv = depths.std() / depths.mean()
    prev = (Yf > 0).mean(0)
    rel = Yf / np.maximum(depths[:, None], 1)
    span = np.log10(np.maximum(rel.mean(0), 1e-12))
    mu = Yf.mean(0); va = Yf.var(0); m0 = mu > 0
    slope = np.polyfit(np.log(mu[m0]), np.log(np.maximum(va[m0], 1e-12)), 1)[0]
    return dict(zero=round(zr, 3), dcv=round(dcv, 2),
                prev_med=round(np.median(prev), 2),
                prev_max=round(prev.max(), 2),
                span=round(span.max() - np.percentile(span, 25), 1),
                mv=round(slope, 2))


def fdp_tpr(rej, truth):
    nr = int(rej.sum()); fp = int((rej & ~truth).sum()); tp = int((rej & truth).sum())
    return nr, (fp / nr if nr else 0.0), (tp / truth.sum() if truth.sum() else np.nan)


if __name__ == "__main__":
    rows = []
    for cell, spec in CELLS.items():
        for rep in range(R):
            t0 = time.time()
            seed = np.random.SeedSequence([BASE_SEED, cell]).spawn(R)[rep]
            Y, tr = generate_ext(spec["mech"], spec["params"], spec["n"], P,
                                 spec["depth"], seed=seed)
            g = tr["group"]; N = tr["depths"].astype(float)
            A = tr["abs_da_truth"].astype(bool)
            Pt = tr.get("pres_da_truth", np.zeros(P, dtype=bool)).astype(bool)
            Ut = A | Pt
            sc = scorecard(Y, N)
            lib = Y.sum(1)
            sc["uplift"] = round(lib[g == 1].mean() / lib[g == 0].mean(), 2)

            res = two_channel_test(Y, N, g, nu=N, K=K, seed=[20260819, cell, rep])
            rej = bh_reject(res["p_comb"], ALPHA)
            nr, fdp, tpr = fdp_tpr(rej, Ut)
            _, _, tprA = fdp_tpr(rej, A)
            _, _, tprP = fdp_tpr(rej, Pt)
            t1 = float((res["p_comb"][~Ut] < 0.05).mean())

            # oracle-exclusion detectability proxy (adversarial cells)
            orc = np.nan
            if cell in (2005, 2006) and A.any():
                hits = 0
                pres = tr["presence"]
                for j in np.where(A)[0]:
                    m = pres[:, j]
                    if (m & (g == 1)).sum() >= 3 and (m & (g == 0)).sum() >= 3:
                        ra = Y[m, j] / N[m]
                        pv = ranksums(ra[g[m] == 1], ra[g[m] == 0]).pvalue
                        hits += pv < ALPHA / max(A.sum(), 1)   # Bonferroni proxy
                orc = hits / A.sum()

            rows.append(dict(cell=cell, rep=rep, mech=spec["mech"], **sc,
                             nrej=nr, fdp=round(fdp, 3), tpr=tpr, tprA=tprA,
                             tprP=tprP, typeI=round(t1, 3), oracle_tpr=orc,
                             phi_hat=res["phi_hat"],
                             secs=round(time.time() - t0, 1)))
            print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("/home/claude/ch_smoke/pilot_v4_detail.csv", index=False)
    agg = dict(zero="mean", dcv="mean", prev_med="mean", prev_max="mean",
               span="mean", mv="mean", uplift="mean", fdp="mean", tpr="mean",
               tprA="mean", tprP="mean", typeI="mean", oracle_tpr="mean",
               nrej="mean", secs="mean")
    s = df.groupby("cell").agg(agg).round(3)
    s.to_csv("/home/claude/ch_smoke/pilot_v4_summary.csv")
    print("\n=== PILOT SUMMARY (R=3 means) ===")
    print(s.to_string())
