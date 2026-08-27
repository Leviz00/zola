#!/usr/bin/env python3
"""E4/E5/E6 per SPEC_EXHIBIT: attribution accuracy (spike rerun with
per-taxon persistence), e-BH sensitivity column (native rerun with
chi2-sum e-values), AGP-14 effect sizes. K=9999, frozen seeds; bridge
assertions against archived real10k CSVs. Run with PYTHONHASHSEED=0."""
import os, sys
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
assert os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0"
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, "/home/claude/ch_smoke")
import run_rich as rr
from twochannel import (fit_detection_curves, tnb_null_residuals,
                        median_ratio_offset, bh_reject)

rr.K = 9999
A = "/home/claude/zola/archives"
OUT = "/home/claude/zola/exhibit"
os.makedirs(OUT, exist_ok=True)
ARCH_TAXA = pd.read_csv(f"{A}/real10k_results/real10k/real10k_taxa.csv")
ARCH_SUM = pd.read_csv(f"{A}/real10k_results/real10k/real10k_summary.csv")


def two_channel_m_ext(Y, N, group, m, nu, seed, strata=None, clusters=None):
    """Verbatim mirror of rr.two_channel_m + chi2-sum e-values (SPEC E6).
    Any deviation from the mirror is a bug; p_comb is asserted identical
    to rr.two_channel_m under the same seed by the caller."""
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
    p_det = perm_p(Z1 ** 2); p_det[~act] = 1.0
    attrib = np.where(Z1[0] ** 2 >= Z2[0] ** 2, "det", "int")
    # --- E6 addition: chi2-sum e-values (f = identity on C) ---
    C = Z1 ** 2 + Z2 ** 2
    tot = C.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(tot > 0, (K + 1.0) * C[0] / tot, 0.0)
    e[~(act | int_ok)] = 0.0
    return dict(p_comb=p_comb, p_det=p_det,
                p_int=np.where(int_ok, perm_p(Z2 ** 2), 1.0),
                attribution=attrib, det_ok=act, evalue=e)


def ebh(e, alpha=0.05):
    m = len(e)
    o = np.argsort(-e)
    es = e[o]
    k = np.arange(1, m + 1)
    ok = np.where(es >= m / (alpha * k))[0]
    rej = np.zeros(m, bool)
    if len(ok):
        rej[o[:ok.max() + 1]] = True
    return rej


spike_rows, ebh_rows, eff_rows, bridge_rows = [], [], [], []

for name, (Y, N, g, taxa, st, cl, tag) in rr.load_cohorts(
        ("ibdmdb", "mbqc", "agp")).items():
    prev_full = (Y > 0).mean(0)
    keep = np.sort(np.argsort(-prev_full)[:100])
    Yk, tk = Y[:, keep], taxa[keep]
    prev = (Yk > 0).mean(0)
    nu = median_ratio_offset(Yk)
    D = (Yk > 0)
    qhat = fit_detection_curves(D.astype(float), N)["qhat"]

    # ---- native unadj: reference run (rr) + ext mirror ----
    res_ref = rr.two_channel_m(Yk, N, g, qhat, nu, seed=[20260826, tag],
                               strata=st, clusters=cl)
    res = two_channel_m_ext(Yk, N, g, qhat, nu, seed=[20260826, tag],
                            strata=st, clusters=cl)
    assert np.array_equal(res_ref["p_comb"], res["p_comb"]), "mirror drift"
    rej = bh_reject(res["p_comb"], 0.05)
    rejU = rej

    # bridge vs archived per-taxon rows (unadj columns)
    arch = ARCH_TAXA[ARCH_TAXA.cohort == name]
    tmap = {str(t): j for j, t in enumerate(tk)}
    n_match = 0
    for _, r0 in arch.iterrows():
        j = tmap[r0.taxon]
        ok_p = abs(res["p_comb"][j] - r0.p_comb) < 5e-4
        ok_r = bool(rej[j]) == bool(r0.rejected)
        if ok_p and ok_r:
            n_match += 1
        else:
            print(f"BRIDGE MISS {name} {r0.taxon}: p {res['p_comb'][j]:.4f}"
                  f" vs {r0.p_comb}, rej {bool(rej[j])} vs {r0.rejected}")
    arch_nrej = int(ARCH_SUM[(ARCH_SUM.cohort == name) &
                             (ARCH_SUM.analysis == "native_unadj")
                             ].n_rej.values[0])
    bridge_rows.append(dict(cohort=name, kind="native",
                            archived_rows=len(arch), matched=n_match,
                            n_rej=int(rej.sum()), archived_n_rej=arch_nrej))
    print(f"[{name}] native bridge: {n_match}/{len(arch)} rows, "
          f"n_rej {int(rej.sum())} (arch {arch_nrej})", flush=True)
    assert n_match == len(arch), f"{name}: native bridge failed"
    assert int(rej.sum()) == arch_nrej

    # ---- E6: e-BH sensitivity on the native list ----
    erej = ebh(res["evalue"], 0.05)
    for j in range(len(tk)):
        if rej[j] or erej[j]:
            ebh_rows.append(dict(
                cohort=name, taxon=str(tk[j]),
                p_comb=round(float(res["p_comb"][j]), 5),
                evalue=round(float(res["evalue"][j]), 2),
                bh=bool(rej[j]), ebh=bool(erej[j]),
                channel=res["attribution"][j]))
    print(f"[{name}] BH {int(rej.sum())} vs e-BH {int(erej.sum())}",
          flush=True)

    # ---- E4: AGP effect sizes on the official 14 ----
    if name == "agp":
        off14 = arch[arch.rejected == True].taxon.tolist()
        assert len(off14) == 14
        case = g == 1
        r_norm = Yk / nu[:, None]
        for t in off14:
            j = tmap[t]
            dj = D[:, j]
            a = int((dj & case).sum()); b = int((~dj & case).sum())
            c = int((dj & ~case).sum()); d = int((~dj & ~case).sum())
            hald = 0 in (a, b, c, d)
            if hald:
                orr = ((a + .5) * (d + .5)) / ((b + .5) * (c + .5))
            else:
                orr = (a * d) / (b * c)
            nc, n0 = int((dj & case).sum()), int((dj & ~case).sum())
            amp = (r_norm[dj & case, j].mean() /
                   r_norm[dj & ~case, j].mean()) if min(nc, n0) >= 3 \
                else np.nan
            r0 = arch[arch.taxon == t].iloc[0]
            eff_rows.append(dict(
                taxon=t, prev=round(float(prev[j]), 3),
                det_case=round(a / case.sum(), 3),
                det_ctrl=round(c / (~case).sum(), 3),
                det_OR=round(float(orr), 3), haldane=hald,
                amp_ratio=(round(float(amp), 3)
                           if np.isfinite(amp) else None),
                n_det_case=nc, n_det_ctrl=n0,
                channel=r0.channel, p_comb=r0.p_comb,
                survives_rich=bool(r0.rejected_rich)))

    # ---- E5: spike arms, per-taxon persistence ----
    order = np.argsort(-prev)
    tiers = [order[:33], order[33:66], order[66:]]
    r = np.random.default_rng([20260820, 99, tag])
    sel = np.concatenate([r.choice(t, 5, replace=False) for t in tiers])
    for armname in ("intensity", "presence"):
        Ys = Yk.copy()
        case = g == 1
        for i, j in enumerate(sel):
            if armname == "intensity":
                f = 2 if i % 2 == 0 else 4
                m_ = case & (Ys[:, j] > 0)
                Ys[m_, j] = np.round(Ys[m_, j] * f).astype(Ys.dtype)
            else:
                m_ = case & (Ys[:, j] > 0)
                drop = r.random(m_.sum()) < 0.5
                Ys[np.where(m_)[0][drop], j] = 0
        Ds = (Ys > 0)
        qs = fit_detection_curves(Ds.astype(float), N)["qhat"]
        nus = median_ratio_offset(Ys)
        res_s = rr.two_channel_m(Ys, N, g, qs, nus,
                                 seed=[20260827, tag,
                                       hash(armname) % 997],
                                 strata=st, clusters=cl)
        rej_s = bh_reject(res_s["p_comb"], 0.05)
        inj = "int" if armname == "intensity" else "det"
        for i, j in enumerate(sel):
            spike_rows.append(dict(
                cohort=name, arm=armname, taxon=str(tk[j]), tier=i // 5,
                injected=inj, recovered=bool(rej_s[j]),
                attributed=res_s["attribution"][j],
                p_comb=round(float(res_s["p_comb"][j]), 5),
                p_det=round(float(res_s["p_det"][j]), 5),
                p_int=round(float(res_s["p_int"][j]), 5),
                prev=round(float(prev[j]), 3)))
        got = int(rej_s[sel].sum())
        arch_rec = int(ARCH_SUM[(ARCH_SUM.cohort == name) &
                                (ARCH_SUM.analysis ==
                                 f"spike_{armname}_unadj")
                                ].rej_det.values[0])
        bridge_rows.append(dict(cohort=name, kind=f"spike_{armname}",
                                archived_rows=15, matched=got,
                                n_rej=got, archived_n_rej=arch_rec))
        print(f"[{name}] spike {armname}: recovered {got} "
              f"(arch {arch_rec})", flush=True)
        assert abs(got - arch_rec) <= 1, f"{name} {armname} bridge fail"

pd.DataFrame(spike_rows).to_csv(f"{OUT}/attribution_accuracy.csv",
                                index=False)
pd.DataFrame(ebh_rows).to_csv(f"{OUT}/ebh_lists.csv", index=False)
pd.DataFrame(eff_rows).to_csv(f"{OUT}/agp_effect_sizes.csv", index=False)
pd.DataFrame(bridge_rows).to_csv(f"{OUT}/exhibit_bridge_checks.csv",
                                 index=False)

# ---- attribution accuracy summary ----
sp = pd.DataFrame(spike_rows)
rec = sp[sp.recovered]
summ = rec.groupby(["cohort", "arm"]).apply(
    lambda x: pd.Series(dict(recovered=len(x),
                             misattributed=int((x.attributed !=
                                                x.injected).sum()))),
    include_groups=False).reset_index()
print("\n=== attribution among recovered spikes ===")
print(summ.to_string(index=False))
tot = rec.groupby("arm").apply(
    lambda x: pd.Series(dict(recovered=len(x),
                             misattributed=int((x.attributed !=
                                                x.injected).sum()))),
    include_groups=False).reset_index()
print(tot.to_string(index=False))
summ.to_csv(f"{OUT}/attribution_summary.csv", index=False)
print("DONE")
