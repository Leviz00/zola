#!/usr/bin/env python3
"""Amendment EX-A1 (post-hoc, documented): e-BH sensitivity column under a
fixed tail-indicator transform, alongside the frozen identity-f arm.

Frozen E6 (f = identity on the chi2-sum C) yielded zero e-BH rejections in
every cohort: e = (K+1) C0 / sum_k Ck is capped by permutation null mass
(observed max 282 on MBQC, 17 on AGP, vs e-BH thresholds m/(alpha k)).
Standard remedy within the same invariance family: concentrate f in the
tail. EX-A1 adds f_c(T) = 1{T >= c} with c fixed a priori at the
chi-square(2) upper tail matched to the permutation resolution:
c = -2 ln(1e-4) = 18.4207 (K+1 = 1e4). Validity needs only fixed c +
exchangeability (E[e] <= 1); no per-taxon tuning. Both arms reported.
Run with PYTHONHASHSEED=0 (native seeds contain no hash; flag kept for
uniformity)."""
import os, sys
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, "/home/claude/ch_smoke")
import run_rich as rr
from twochannel import (fit_detection_curves, tnb_null_residuals,
                        median_ratio_offset, bh_reject)

rr.K = 9999
C_THRESH = -2.0 * np.log(1e-4)          # 18.420681
OUT = "/home/claude/zola/exhibit"
ARCH = pd.read_csv(f"{OUT}/ebh_lists.csv")


def chi2sum_perm(Y, N, group, m, nu, seed, strata=None, clusters=None):
    """Mirror of the statistic layer, returning the C=(K+1,p) matrix and
    the official p_comb for bridging."""
    K = rr.K
    Y = np.asarray(Y); n, p = Y.shape
    x = np.where(np.asarray(group) == 1, 0.5, -0.5)
    D = (Y > 0).astype(float)
    act = (D.sum(0) >= 3) & ((1 - D).sum(0) >= 3)
    R1 = D - m
    R1[:, ~act] = 0.0
    int_ok = np.zeros(p, dtype=bool)
    R2 = np.zeros((n, p))
    for j in range(p):
        pos = Y[:, j] > 0
        if (pos & (x > 0)).sum() < 3 or (pos & (x < 0)).sum() < 3:
            continue
        s = tnb_null_residuals(Y[pos, j].astype(float), np.log(nu[pos]))
        if s is None:
            continue
        R2[pos, j] = s
        int_ok[j] = True
    X = rr.perm_labels(x, K, seed, strata, clusters)
    Xc = X - X.mean(axis=1, keepdims=True)
    U1, U2 = Xc @ R1, Xc @ R2

    def stud(U):
        mu = U.mean(0, keepdims=True); sd = U.std(0, keepdims=True)
        return (U - mu) / np.where(sd < 1e-12, 1.0, sd)

    Z1, Z2 = stud(U1), stud(U2)
    Z1[:, ~act] = 0.0
    Z2[:, ~int_ok] = 0.0

    def wsp(M):
        return rankdata(-M, method="max", axis=0) / M.shape[0]

    def perm_p(M):
        return (1.0 + (M[1:] >= M[0][None, :]).sum(0)) / (K + 1.0)

    P1, P2 = wsp(Z1 ** 2), wsp(Z2 ** 2)
    eps = 1.0 / (2 * (K + 1))
    T1 = np.tan(np.pi * (0.5 - np.clip(P1, eps, 1 - eps)))
    T2 = np.tan(np.pi * (0.5 - np.clip(P2, eps, 1 - eps)))
    a = act.astype(float) + int_ok.astype(float)
    ACAT = (T1 * act[None] + T2 * int_ok[None]) / np.where(a > 0, a, 1.0)
    p_comb = perm_p(ACAT); p_comb[~(act | int_ok)] = 1.0
    return (Z1 ** 2 + Z2 ** 2), (act | int_ok), p_comb


def ebh(e, alpha=0.05):
    m = len(e)
    o = np.argsort(-e)
    k = np.arange(1, m + 1)
    ok = np.where(e[o] >= m / (alpha * k))[0]
    rej = np.zeros(m, bool)
    if len(ok):
        rej[o[:ok.max() + 1]] = True
    return rej


rows = []
for name, (Y, N, g, taxa, st, cl, tag) in rr.load_cohorts(
        ("ibdmdb", "mbqc", "agp")).items():
    prev = (Y > 0).mean(0)
    keep = np.sort(np.argsort(-prev)[:100])
    Yk, tk = Y[:, keep], taxa[keep]
    nu = median_ratio_offset(Yk)
    qhat = fit_detection_curves((Yk > 0).astype(float), N)["qhat"]
    C, tested, p_comb = chi2sum_perm(Yk, N, g, qhat, nu,
                                     seed=[20260826, tag],
                                     strata=st, clusters=cl)
    rej_bh = bh_reject(p_comb, 0.05)
    K = rr.K
    # identity arm (frozen E6) — bridge against first run
    tot = C.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        e_id = np.where(tot > 0, (K + 1.0) * C[0] / tot, 0.0)
    e_id[~tested] = 0.0
    arch = ARCH[ARCH.cohort == name]
    tmap = {str(t): j for j, t in enumerate(tk)}
    for _, r0 in arch.iterrows():
        assert abs(e_id[tmap[r0.taxon]] - r0.evalue) < 0.02, \
            f"identity-e bridge fail {name} {r0.taxon}"
    # EX-A1 indicator arm
    exceed = (C >= C_THRESH).sum(0)          # includes observed row
    e_ind = np.where((C[0] >= C_THRESH) & tested,
                     (K + 1.0) / np.maximum(exceed, 1), 0.0)
    r_id, r_ind = ebh(e_id), ebh(e_ind)
    null_exc = float((C[1:] >= C_THRESH).mean(0).mean() * K)
    print(f"[{name}] BH {int(rej_bh.sum())} | e-BH id {int(r_id.sum())} "
          f"| e-BH ind {int(r_ind.sum())} | mean null exceed "
          f"{null_exc:.2f}", flush=True)
    for j in range(len(tk)):
        if rej_bh[j] or r_id[j] or r_ind[j]:
            rows.append(dict(cohort=name, taxon=str(tk[j]),
                             p_comb=round(float(p_comb[j]), 5),
                             e_identity=round(float(e_id[j]), 2),
                             e_indicator=round(float(e_ind[j]), 1),
                             bh=bool(rej_bh[j]), ebh_id=bool(r_id[j]),
                             ebh_ind=bool(r_ind[j])))

t = pd.DataFrame(rows)
t.to_csv(f"{OUT}/ebh_lists_a1.csv", index=False)
surv = t[t.bh].groupby("cohort").ebh_ind.sum()
tot = t[t.bh].groupby("cohort").size()
print("\nBH discoveries surviving indicator e-BH:")
print(pd.DataFrame(dict(bh=tot, survive_ebh=surv)))
print("DONE")
