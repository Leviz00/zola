"""run_rich.py -- SPEC-RICH: richness-adjusted detection channel.

E1: simulation validation (specific cell 2001 vs broad community-shift
    cell 2008-COMM), unadjusted vs adjusted arms.
E2: three cohorts, native + presence spike-in, both arms; survival of the
    archived discovery lists; community-level richness test.
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, time
import numpy as np
import pandas as pd
from scipy.special import expit, logit as slogit
from scipy.stats import rankdata, ranksums

sys.path.insert(0, "/home/claude/ch_smoke/code/simulation_v3")
sys.path.insert(0, "/home/claude/ch_smoke")
from twochannel import (fit_detection_curves, tnb_null_residuals,
                        median_ratio_offset, bh_reject)

UP = "/mnt/user-data/uploads"
OUT = "/home/claude/ch_smoke/rich"
os.makedirs(OUT, exist_ok=True)
K = 999


# ---------------- permutations (strata OR clusters) ------------------------

def perm_labels(x, K, seed, strata=None, clusters=None):
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    n = len(x)
    X = np.empty((K + 1, n))
    X[0] = x
    if clusters is not None:
        clusters = np.asarray(clusters)
        uc, first = np.unique(clusters, return_index=True)
        cx = x[first]
        pos = {c: np.where(clusters == c)[0] for c in uc}
        for k in range(1, K + 1):
            pcx = rng.permutation(cx)
            xk = np.empty(n)
            for c, v in zip(uc, pcx):
                xk[pos[c]] = v
            X[k] = xk
    elif strata is None:
        for k in range(1, K + 1):
            X[k] = rng.permutation(x)
    else:
        strata = np.asarray(strata)
        for k in range(1, K + 1):
            xk = x.copy()
            for s in np.unique(strata):
                m = strata == s
                xk[m] = rng.permutation(x[m])
            X[k] = xk
    return X


def two_channel_m(Y, N, group, m, nu, seed, strata=None, clusters=None):
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
    X = perm_labels(x, K, seed, strata, clusters)
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
    return dict(p_comb=p_comb, p_det=p_det,
                p_int=np.where(int_ok, perm_p(Z2 ** 2), 1.0),
                attribution=attrib, det_ok=act)


# ---------------- richness-adjusted propensity -----------------------------

def m_rich_rasch_trim(Y, N, qhat):
    """Row-effect (Rasch-style) adjustment: m_ij = expit(logit qhat + g_i).

    gamma_i = per-sample MLE of a shared detection-efficiency shift
    (label-blind); sufficient statistic = row total (richness), entering
    nonlinearly (Amendment RICH-A3)."""
    D = (Y > 0).astype(float)
    n, p = D.shape
    off = slogit(np.clip(qhat, 1e-6, 1 - 1e-6))
    prev = D.mean(0)
    ref = np.argsort(-prev)[:50]                    # reference set (A4)
    gam = np.zeros(n)
    nfail = 0

    def fit_gamma(y, o):
        g = 0.0
        for _ in range(40):
            mu = expit(np.clip(o + g, -30, 30))
            s1 = float((y - mu).sum())
            s2 = float(np.maximum(mu * (1 - mu), 1e-9).sum())
            step = s1 / s2
            g += step
            if abs(step) < 1e-9:
                return g, True
            if abs(g) > 30:
                return g, False
        return g, True

    for i in range(n):
        y = D[i, ref]; o = off[i, ref]
        g, ok = fit_gamma(y, o)
        if ok:
            # trimmed refit: drop the 20% most outlying reference taxa
            mu = expit(np.clip(o + g, -30, 30))
            dev = np.abs(y - mu) / np.sqrt(np.maximum(mu * (1 - mu), 1e-9))
            keep = dev <= np.quantile(dev, 0.8)
            g2, ok2 = fit_gamma(y[keep], o[keep])
            if ok2:
                g = g2
        if ok:
            gam[i] = g
        else:
            nfail += 1
    M = expit(np.clip(off + gam[:, None], -30, 30))
    return M, nfail


def m_rich(Y, N, qhat):
    """FINAL (RICH-A5): linear leave-one-out richness covariate."""
    D = (Y > 0).astype(float)
    n, p = D.shape
    Rtot = D.sum(1)
    off_all = slogit(np.clip(qhat, 1e-6, 1 - 1e-6))
    M = qhat.copy()
    nfail = 0
    for j in range(p):
        y = D[:, j]
        if y.sum() < 3 or (1 - y).sum() < 3:
            continue
        R = Rtot - y
        zR = (R - R.mean()) / max(R.std(), 1e-12)
        off = off_all[:, j]
        Xd = np.column_stack([np.ones(n), zR])
        b = np.zeros(2)
        ok = True
        for _ in range(30):
            eta = np.clip(off + Xd @ b, -30, 30)
            mu = expit(eta)
            W = np.maximum(mu * (1 - mu), 1e-9)
            g = Xd.T @ (y - mu)
            H = (Xd * W[:, None]).T @ Xd
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                ok = False; break
            b = b + step
            if np.abs(step).max() < 1e-8:
                break
            if np.abs(b).max() > 30:
                ok = False; break
        if ok:
            M[:, j] = expit(np.clip(off + Xd @ b, -30, 30))
        else:
            nfail += 1
    return M, nfail


# ---------------- E1 -------------------------------------------------------

CELLS = {
    2001: dict(mech="three_layer_real", depth=20000, n=300,
               params=dict(da_fraction=0.0, presence_da_fraction=0.10,
                           presence_effect_or=0.25)),
    2008: dict(mech="three_layer_real", depth=20000, n=300,   # COMM
               params=dict(da_fraction=0.0, presence_da_fraction=1.0,
                           presence_effect_or=0.45)),
    2009: dict(mech="three_layer_real", depth=20000, n=300,   # GLOBAL thin
               params=dict(da_fraction=0.0, presence_da_fraction=0.0),
               thin=0.15),
}


def e1(R=10, only=None):
    from generators_ext import generate_ext
    rows = []
    for cell, spec in CELLS.items():
        if only and cell != only:
            continue
        for rep in range(R):
            seed = np.random.SeedSequence([20260819, cell]).spawn(20)[rep]
            Y, tr = generate_ext(spec["mech"], spec["params"], spec["n"],
                                 100, spec["depth"], seed=seed)
            g = tr["group"].astype(int)
            Pt = tr.get("pres_da_truth",
                        np.zeros(100, bool)).astype(bool)
            if spec.get("thin"):
                rng_t = np.random.default_rng([20260825, cell, rep])
                case_rows = np.where(g == 1)[0]
                sub = Y[case_rows]
                mask = (sub > 0) & (rng_t.random(sub.shape) < spec["thin"])
                sub[mask] = 0
                Y[case_rows] = sub
            lib = Y.sum(1).astype(float)
            nu = median_ratio_offset(Y, n_ref=50)
            D = (Y > 0)
            qhat = fit_detection_curves(D.astype(float), lib)["qhat"]
            mr, nfail = m_rich(Y, lib, qhat)
            for arm, m in (("unadj", qhat), ("rich", mr)):
                res = two_channel_m(Y, lib, g, m, nu,
                                    seed=[20260824, cell, rep])
                rej = bh_reject(res["p_comb"], 0.05)
                nr = int(rej.sum())
                tp = int((rej & Pt).sum())
                null_p = res["p_comb"][~Pt]
                rows.append(dict(cell=cell, rep=rep, arm=arm, nrej=nr,
                                 tpr=tp / Pt.sum() if Pt.sum() else np.nan,
                                 fdp=(nr - tp) / nr if nr else 0.0,
                                 typeI=float((null_p < 0.05).mean()),
                                 typeI_det=float((res["p_det"][~Pt] < 0.05
                                                  ).mean()),
                                 typeI_int=float((res["p_int"][~Pt] < 0.05
                                                  ).mean()),
                                 n_truth=int(Pt.sum()), nfail=nfail))
            print(rows[-2:], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/rich_e1_detail.csv", index=False)
    s = df.groupby(["cell", "arm"]).mean(numeric_only=True).round(4)
    s.to_csv(f"{OUT}/rich_e1_summary.csv")
    print(s.to_string(), flush=True)


# ---------------- E2 -------------------------------------------------------

def load_cohorts(which):
    out = {}
    if "ibdmdb" in which:
        z = np.load(f"{UP}/zola_project/realdata/data/ibdmdb_genus.npz",
                    allow_pickle=True)
        Y, N, taxa = z["Y"], z["depths"].astype(float), z["taxa"]
        md = pd.read_csv(f"{UP}/zola_project 2/datasets/ibdmdb/"
                         "ibdmdb_16S_sample_metadata.csv", low_memory=False)
        md = md.set_index(md["sample_id"].astype(str))
        samp = [str(s) for s in z["samples"]]
        diag = md.loc[samp, "diagnosis"].values
        subj = md.loc[samp, "subject_id"].astype(str).values
        g = np.where(pd.Series(diag).isin(["CD", "UC"]).values, 1, 0)
        out["ibdmdb"] = (Y, N, g, taxa, None, subj, 1)
    if "mbqc" in which:
        z = np.load(f"{UP}/zola_project/realdata/data/mbqc_genus.npz",
                    allow_pickle=True)
        Y, N, taxa = z["Y"], z["depths"].astype(float), z["taxa"]
        ids = [str(s).split(".") for s in z["samples"]]
        bl = np.array([f[1] if len(f) > 3 else "?" for f in ids])
        hl = np.array([f[3] if len(f) > 3 else "?" for f in ids])
        idx = np.where((bl == "4") | (bl == "6"))[0]
        gg = (bl[idx] == "6").astype(int)
        r = np.random.default_rng(20260304)
        sub = np.sort(np.concatenate([
            r.choice(idx[gg == 0], 350, replace=False),
            r.choice(idx[gg == 1], 350, replace=False)]))
        out["mbqc"] = (Y[sub], N[sub], (bl[sub] == "6").astype(int), taxa,
                       hl[sub], None, 2)
    if "agp" in which:
        z = np.load(f"{UP}/zola_project/realdata/data/agp_genus.npz",
                    allow_pickle=True)
        Y, N, taxa = z["Y"], z["depths"].astype(float), z["taxa"]
        md = pd.read_csv(f"{UP}/zola_project 2/datasets/agp/"
                         "agp_sample_metadata_9511fecal.csv",
                         low_memory=False)
        md = md.set_index(md["sample_name"].astype(str))
        samp = [str(s) for s in z["samples"]]
        ib = md.reindex(samp)["ibd"].astype(str).values
        has = np.array(["diagnosed" in v.lower() or "self" in v.lower()
                        for v in ib])
        no = np.array([v == "I do not have this condition" for v in ib])
        idx = np.where(has | no)[0]
        gg = has[idx].astype(int)
        r = np.random.default_rng(20260304)
        sub = np.sort(np.concatenate([
            r.choice(idx[gg == 0], 350, replace=False),
            r.choice(idx[gg == 1], min(350, int(gg.sum())), replace=False)]))
        out["agp"] = (Y[sub], N[sub], has[sub].astype(int), taxa, None,
                      None, 3)
    return out


def e2(which=("ibdmdb", "agp", "mbqc")):
    rows, surv_rows = [], []
    for name, (Y, N, g, taxa, st, cl, tag) in load_cohorts(which).items():
        prev = (Y > 0).mean(0)
        keep = np.sort(np.argsort(-prev)[:100])
        Yk, tk = Y[:, keep], taxa[keep]
        nu = median_ratio_offset(Yk)
        D = (Yk > 0)
        rich = D.sum(1)
        rich_p = ranksums(rich[g == 1], rich[g == 0]).pvalue
        t0 = time.time()
        qhat = fit_detection_curves(D.astype(float), N)["qhat"]
        mr, nfail = m_rich(Yk, N, qhat)
        old = pd.read_csv(f"/home/claude/ch_smoke/real/real_{name}_taxa.csv")
        old_rej = set(old[old.rejected].taxon.astype(str))
        for arm, m in (("unadj", qhat), ("rich", mr)):
            res = two_channel_m(Yk, N, g, m, nu, seed=[20260820, tag],
                                strata=st, clusters=cl)
            rej = bh_reject(res["p_comb"], 0.05)
            newlist = set(np.asarray(tk)[rej].astype(str))
            n_surv = len(old_rej & newlist)
            rows.append(dict(cohort=name, arm=arm, n_rej=int(rej.sum()),
                             rej_det=int((rej & (res["attribution"] ==
                                                 "det")).sum()),
                             old_n=len(old_rej), old_survive=n_surv,
                             rich_group_p=round(rich_p, 5), nfail=nfail,
                             secs=round(time.time() - t0, 1)))
            print(rows[-1], flush=True)
            if arm == "rich":
                for t in sorted(old_rej):
                    i = np.where(np.asarray(tk).astype(str) == t)[0]
                    if len(i):
                        surv_rows.append(dict(
                            cohort=name, taxon=t,
                            p_comb_rich=float(res["p_comb"][i[0]]),
                            p_det_rich=float(res["p_det"][i[0]]),
                            still_rejected=bool(rej[i[0]])))
        # presence spike-in, both arms
        order = np.argsort(-(Yk > 0).mean(0))
        tiers = [order[:33], order[33:66], order[66:]]
        rs = np.random.default_rng([20260820, 99, tag])
        sel = np.concatenate([rs.choice(t, 5, replace=False) for t in tiers])
        Ys = Yk.copy()
        case = g == 1
        for i, j in enumerate(sel):
            mmask = case & (Ys[:, j] > 0)
            drop = rs.random(mmask.sum()) < 0.5
            Ys[np.where(mmask)[0][drop], j] = 0
        Ds = (Ys > 0)
        qs = fit_detection_curves(Ds.astype(float), N)["qhat"]
        mrs, _ = m_rich(Ys, N, qs)
        nus = median_ratio_offset(Ys)
        for arm, m in (("unadj", qs), ("rich", mrs)):
            res = two_channel_m(Ys, N, g, m, nus, seed=[20260821, tag, 500],
                                strata=st, clusters=cl)
            rej = bh_reject(res["p_comb"], 0.05)
            rows.append(dict(cohort=name, arm=f"spike_{arm}",
                             n_rej=int(rej.sum()),
                             rej_det=int(rej[sel].sum()),
                             old_n=15, old_survive=int(rej[sel].sum()),
                             rich_group_p=np.nan, nfail=0, secs=0))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/rich_e2_summary.csv", index=False)
    pd.DataFrame(surv_rows).to_csv(f"{OUT}/rich_e2_survival.csv",
                                   index=False)
    print("E2 done", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("e1", "both"):
        e1(only=int(sys.argv[2]) if len(sys.argv) > 2 else None)
    if which in ("e2", "both"):
        e2()
