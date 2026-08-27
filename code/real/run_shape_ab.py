"""run_shape_ab.py -- SPEC-SHAPE: depth-adjustment shape A/B on MBQC.

E1: permutation pipeline with custom detection-propensity m (occupancy /
    log-linear / constant), native lists + presence spike-in recovery.
E2: pseudo-null calibration under engineered depth confounding, asymptotic
    Wald arms vs permutation arms.
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, time
import numpy as np
import pandas as pd
from scipy.special import expit, logit as slogit
from scipy.stats import rankdata

sys.path.insert(0, "/home/claude/ch_smoke")
from twochannel import (fit_detection_curves, tnb_null_residuals,
                        median_ratio_offset, bh_reject)

UP = "/mnt/user-data/uploads"
OUT = "/home/claude/ch_smoke/shape"
os.makedirs(OUT, exist_ok=True)
K = 999


# ---------------- custom-m two-channel (mirrors two_channel_test) ----------

def perm_labels(x, K, seed, strata=None):
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    n = len(x)
    X = np.empty((K + 1, n))
    X[0] = x
    if strata is None:
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


def two_channel_custom(Y, N, group, m, nu, K=K, seed=0, strata=None,
                       det_only=False):
    Y = np.asarray(Y); n, p = Y.shape
    x = np.where(np.asarray(group) == 1, 0.5, -0.5)
    D = (Y > 0).astype(float)
    act = (D.sum(0) >= 3) & ((1 - D).sum(0) >= 3)
    R1 = D - m
    R1[:, ~act] = 0.0

    int_ok = np.zeros(p, dtype=bool)
    R2 = np.zeros((n, p))
    if not det_only:
        for j in range(p):
            pos = Y[:, j] > 0
            if (pos & (x > 0)).sum() < 3 or (pos & (x < 0)).sum() < 3:
                continue
            s = tnb_null_residuals(Y[pos, j].astype(float), np.log(nu[pos]))
            if s is None:
                continue
            R2[pos, j] = s
            int_ok[j] = True

    X = perm_labels(x, K, seed, strata)
    Xc = X - X.mean(axis=1, keepdims=True)
    U1 = Xc @ R1
    U2 = Xc @ R2

    def stud(U):
        mu = U.mean(0, keepdims=True); sd = U.std(0, keepdims=True)
        return (U - mu) / np.where(sd < 1e-12, 1.0, sd)

    Z1, Z2 = stud(U1), stud(U2)
    Z1[:, ~act] = 0.0
    Z2[:, ~int_ok] = 0.0

    def wsp(M):                     # tie-inclusive within-set p
        return rankdata(-M, method="max", axis=0) / M.shape[0]

    def perm_p(M):
        return (1.0 + (M[1:] >= M[0][None, :]).sum(0)) / (K + 1.0)

    p_det = perm_p(Z1 ** 2)
    p_det[~act] = 1.0
    if det_only:
        return dict(p_det=p_det, det_ok=act)
    P1, P2 = wsp(Z1 ** 2), wsp(Z2 ** 2)
    eps = 1.0 / (2 * (K + 1))
    T1 = np.tan(np.pi * (0.5 - np.clip(P1, eps, 1 - eps)))
    T2 = np.tan(np.pi * (0.5 - np.clip(P2, eps, 1 - eps)))
    a = act.astype(float) + int_ok.astype(float)
    ACAT = (T1 * act[None] + T2 * int_ok[None]) / np.where(a > 0, a, 1.0)
    p_comb = perm_p(ACAT)
    p_comb[~(act | int_ok)] = 1.0
    attrib = np.where(Z1[0] ** 2 >= Z2[0] ** 2, "det", "int")
    return dict(p_comb=p_comb, p_det=p_det,
                p_int=np.where(int_ok, perm_p(Z2 ** 2), 1.0),
                attribution=attrib, det_ok=act)


# ---------------- propensity models ---------------------------------------

def m_occupancy(Y, N):
    D = (Y > 0).astype(float)
    det = fit_detection_curves(D, N)
    return det["qhat"]


def m_loglin(Y, N):
    """Per-taxon logistic D ~ 1 + logN via Newton iterations."""
    D = (Y > 0).astype(float)
    n, p = D.shape
    z = np.log(np.maximum(N, 1.0))
    z = (z - z.mean()) / max(z.std(), 1e-12)
    M = np.empty_like(D)
    for j in range(p):
        y = D[:, j]
        if y.sum() < 3 or (1 - y).sum() < 3:
            M[:, j] = y.mean()
            continue
        b = np.array([slogit(np.clip(y.mean(), 1e-3, 1 - 1e-3)), 0.0])
        Xd = np.column_stack([np.ones(n), z])
        ok = True
        for _ in range(25):
            eta = Xd @ b
            mu = expit(np.clip(eta, -30, 30))
            W = np.maximum(mu * (1 - mu), 1e-9)
            g = Xd.T @ (y - mu)
            H = (Xd * W[:, None]).T @ Xd
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                ok = False
                break
            b = b + step
            if np.abs(step).max() < 1e-8:
                break
            if np.abs(b).max() > 50:
                ok = False
                break
        M[:, j] = expit(np.clip(Xd @ b, -30, 30)) if ok else y.mean()
    return M


def m_const(Y, N):
    D = (Y > 0).astype(float)
    return np.tile(D.mean(0), (Y.shape[0], 1))


# ---------------- MBQC loading (identical to run_real_ch) ------------------

def load_mbqc():
    z = np.load(f"{UP}/zola_project/realdata/data/mbqc_genus.npz",
                allow_pickle=True)
    Y, N, taxa = z["Y"], z["depths"].astype(float), z["taxa"]
    ids = [str(s).split(".") for s in z["samples"]]
    bl = np.array([f[1] if len(f) > 3 else "?" for f in ids])
    hl = np.array([f[3] if len(f) > 3 else "?" for f in ids])
    m46 = (bl == "4") | (bl == "6")
    idx = np.where(m46)[0]
    g = (bl[idx] == "6").astype(int)
    r = np.random.default_rng(20260304)
    sub = np.concatenate([
        r.choice(idx[g == 0], min(350, (g == 0).sum()), replace=False),
        r.choice(idx[g == 1], min(350, (g == 1).sum()), replace=False)])
    sub = np.sort(sub)
    Ym, Nm = Y[sub], N[sub]
    gm = (bl[sub] == "6").astype(int)
    st = hl[sub]
    prev = (Ym > 0).mean(0)
    keep = np.sort(np.argsort(-prev)[:100])
    return Ym[:, keep], Nm, gm, st, taxa[keep], bl, sub


# ---------------- E1 -------------------------------------------------------

def e1():
    Yk, N, g, st, tk, _, _ = load_mbqc()
    nu = median_ratio_offset(Yk)
    arms = {"O": m_occupancy, "L": m_loglin, "C": m_const}
    det_lists, rows = {}, []
    # spike selection: same seed logic as run_real_ch (seed_tag=2)
    prev = (Yk > 0).mean(0)
    order = np.argsort(-prev)
    tiers = [order[:33], order[33:66], order[66:]]
    r = np.random.default_rng([20260820, 99, 2])
    sel = np.concatenate([r.choice(t, 5, replace=False) for t in tiers])

    for tag, mf in arms.items():
        t0 = time.time()
        m = mf(Yk, N)
        res = two_channel_custom(Yk, N, g, m, nu, seed=[20260820, 2],
                                 strata=st)
        rej = bh_reject(res["p_comb"], 0.05)
        det_rej = bh_reject(res["p_det"], 0.05)
        det_lists[tag] = set(np.where(det_rej)[0])
        # presence spike (same construction as run_real_ch, presence arm)
        Ys = Yk.copy()
        case = g == 1
        rs = np.random.default_rng([20260820, 99, 2])
        sel2 = np.concatenate([rs.choice(t, 5, replace=False) for t in tiers])
        assert (sel2 == sel).all()
        for i, j in enumerate(sel):
            mmask = case & (Ys[:, j] > 0)
            drop = rs.random(mmask.sum()) < 0.5
            idx = np.where(mmask)[0][drop]
            Ys[idx, j] = 0
        ms = mf(Ys, N)
        nus = median_ratio_offset(Ys)
        res_s = two_channel_custom(Ys, N, g, ms, nus,
                                   seed=[20260821, 2, 500], strata=st)
        rej_s = bh_reject(res_s["p_comb"], 0.05)
        rows.append(dict(arm=tag, n_rej=int(rej.sum()),
                         n_det=int(det_rej.sum()),
                         spike_rec=round(float(rej_s[sel].mean()), 3),
                         n_rec=int(rej_s[sel].sum()),
                         secs=round(time.time() - t0, 1)))
        print(rows[-1], flush=True)
    for tag in ("L", "C"):
        inter = len(det_lists["O"] & det_lists[tag])
        union = len(det_lists["O"] | det_lists[tag])
        rows.append(dict(arm=f"jaccard_O_{tag}",
                         n_rej=inter, n_det=union,
                         spike_rec=round(inter / union, 3) if union else 1.0,
                         n_rec=0, secs=0))
    pd.DataFrame(rows).to_csv(f"{OUT}/shape_e1.csv", index=False)
    print("E1 done", flush=True)


# ---------------- E2 -------------------------------------------------------

def wald_p(D_j, g, Z):
    """Logistic Wald p for g with covariates Z (n x q); None on failure."""
    import numpy.linalg as la
    n = len(D_j)
    Xd = np.column_stack([np.ones(n), g] + ([Z] if Z is not None else []))
    b = np.zeros(Xd.shape[1])
    b[0] = slogit(np.clip(D_j.mean(), 1e-3, 1 - 1e-3))
    for _ in range(30):
        eta = np.clip(Xd @ b, -30, 30)
        mu = expit(eta)
        W = np.maximum(mu * (1 - mu), 1e-9)
        gr = Xd.T @ (D_j - mu)
        H = (Xd * W[:, None]).T @ Xd
        try:
            step = la.solve(H, gr)
        except la.LinAlgError:
            return np.nan
        b = b + step
        if np.abs(step).max() < 1e-8:
            break
        if np.abs(b).max() > 60:
            return np.nan
    try:
        se = np.sqrt(la.inv(H)[1, 1])
    except la.LinAlgError:
        return np.nan
    from scipy.stats import norm
    return 2 * norm.sf(abs(b[1]) / max(se, 1e-12))


def e2(R=20):
    Yk, N, g0, st, tk, _, _ = load_mbqc()
    # BL4-only pseudo-null universe
    bl4 = g0 == 0
    Y4, N4 = Yk[bl4], N[bl4]
    prev = (Y4 > 0).mean(0)
    keep = np.sort(np.argsort(-prev)[:100])
    Y4 = Y4[:, keep]
    D4 = (Y4 > 0).astype(float)
    n = len(N4)
    z = np.log(np.maximum(N4, 1.0))
    z = (z - z.mean()) / z.std()
    qhat = m_occupancy(Y4, N4)
    lq = slogit(np.clip(qhat, 1e-6, 1 - 1e-6))
    act = (D4.sum(0) >= 3) & ((1 - D4).sum(0) >= 3)
    dec = np.digitize(N4, np.quantile(N4, np.linspace(0, 1, 11)[1:-1]))
    mo = qhat
    mc = np.tile(D4.mean(0), (n, 1))

    rows = []
    for kappa in (0.0, 1.0, 2.0):
        for rep in range(R):
            rng = np.random.default_rng([20260822, int(kappa * 10), rep])
            g = (rng.random(n) < expit(kappa * z)).astype(int)
            if g.sum() < 20 or (1 - g).sum() < 20:
                continue
            rho = np.corrcoef(g, z)[0, 1]
            out = dict(kappa=kappa, rep=rep, rho=round(float(rho), 3))
            # asymptotic Wald arms
            for tag, Zc in (("wald_none", None), ("wald_logN", z[:, None])):
                ps = np.array([wald_p(D4[:, j], g, Zc)
                               for j in range(100) if act[j]])
                ps = ps[np.isfinite(ps)]
                out[f"{tag}_t1"] = float((ps < 0.05).mean())
                out[f"{tag}_bh"] = int(bh_reject(ps, 0.05).sum())
            ps = []
            for j in range(100):
                if not act[j]:
                    continue
                ps.append(wald_p(D4[:, j], g, lq[:, [j]]))
            ps = np.array(ps); ps = ps[np.isfinite(ps)]
            out["wald_zola_t1"] = float((ps < 0.05).mean())
            out["wald_zola_bh"] = int(bh_reject(ps, 0.05).sum())
            # permutation arms (detection channel only)
            for tag, m, sarg in (("perm_plain_C", mc, None),
                                 ("perm_plain_O", mo, None),
                                 ("perm_strata_O", mo, dec)):
                res = two_channel_custom(Y4, N4, g, m, nu=N4, K=K,
                                         seed=[20260823, int(kappa * 10),
                                               rep, hash(tag) % 997],
                                         strata=sarg, det_only=True)
                pd_ = res["p_det"][res["det_ok"]]
                out[f"{tag}_t1"] = float((pd_ < 0.05).mean())
                out[f"{tag}_bh"] = int(bh_reject(pd_, 0.05).sum())
            rows.append(out)
            if rep % 5 == 0:
                print(out, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/shape_e2_detail.csv", index=False)
    s = df.groupby("kappa").mean(numeric_only=True).round(4)
    s.to_csv(f"{OUT}/shape_e2_summary.csv")
    print(s.to_string(), flush=True)
    print("E2 done", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("e1", "both"):
        e1()
    if which in ("e2", "both"):
        e2()
