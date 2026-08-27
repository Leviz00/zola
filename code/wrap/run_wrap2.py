"""run_wrap2.py -- SPEC-WRAP-02: implant ZOLA parts INSIDE external methods.

Mode 1 (completion): LinDA / Wilcoxon statistic + ZOLA detection channel,
  jointly permutation-calibrated (mirrors two_channel_test combination).
Mode 2 (adjustment): ZINQ logistic component with no depth / log N /
  logit(qhat) covariate (R side: zinq_ln / zinq_zola).
Mode 3 (filter): LOCOM on top-k taxa by W-det (k matched per dataset to its
  default-filter kept count), filter.thresh=0 (R side: locom_zola).

Datasets identical to WRAP-01 / frozen battery. Before-arms are reused from
wrap_ds/p_*.csv files where they exist.
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import subprocess, sys, time
import numpy as np
import pandas as pd
from scipy.stats import rankdata, gaussian_kde

sys.path.insert(0, "/home/claude/ch_smoke/code/simulation_v3")
sys.path.insert(0, "/home/claude/ch_smoke")

from run_wrap import CELLS, bh, _linda_t                # noqa: E402


def perm_idx(n, K, seed):
    """(K, n) permutation-index matrix, int32 (int8 would overflow n>127)."""
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    P = np.empty((K, n), dtype=np.int32)
    for k in range(K):
        P[k] = rng.permutation(n)
    return P

K = 999
ALPHA = 0.05
DSROOT = "/home/claude/ch_smoke/wrap_ds"
RS = "/usr/bin/Rscript"
WRAPR = "/home/claude/ch_smoke/wrap_methods.R"


# ---------- generic joint two-channel completion ---------------------------

def completed_p(U_ext, U_det, det_ok, ext_ok):
    """(K+1,p) stat matrices -> jointly calibrated combined p (ACAT)."""
    Kp1 = U_ext.shape[0]

    def studentize(U):
        mu = U.mean(axis=0, keepdims=True)
        sd = U.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-12, 1.0, sd)
        return (U - mu) / sd

    Z1 = studentize(U_det); Z1[:, ~det_ok] = 0.0
    Z2 = studentize(U_ext); Z2[:, ~ext_ok] = 0.0

    def within_set_p(M):
        # tie-INCLUSIVE right-tail rank: p_r = #{M >= M_r}/(K+1). The stable
        # argsort variant favors row 0 (the observed row) on ties, which is
        # anticonservative for discrete statistics (rank sums, detection
        # counts); tie-inclusive ranks are symmetric and conservative.
        from scipy.stats import rankdata as _rd
        return _rd(-M, method="max", axis=0) / M.shape[0]

    P1 = within_set_p(Z1 ** 2)
    P2 = within_set_p(Z2 ** 2)
    eps = 1.0 / (2.0 * Kp1)
    T1 = np.tan(np.pi * (0.5 - np.clip(P1, eps, 1 - eps)))
    T2 = np.tan(np.pi * (0.5 - np.clip(P2, eps, 1 - eps)))
    act = det_ok.astype(float) + ext_ok.astype(float)
    act_safe = np.where(act > 0, act, 1.0)
    ACAT = (T1 * det_ok[None, :] + T2 * ext_ok[None, :]) / act_safe[None, :]
    obs = ACAT[0]
    p = (1.0 + (ACAT[1:] >= obs[None, :]).sum(axis=0)) / Kp1
    p[~(det_ok | ext_ok)] = 1.0
    attrib = np.where(Z1[0] ** 2 >= Z2[0] ** 2, "det", "ext")
    return p, attrib


def score(rej, Ut, A, Pt, p=None):
    nr = int(rej.sum()); fp = int((rej & ~Ut).sum()); tp = int((rej & Ut).sum())
    d = dict(nrej=nr, fdp=fp / nr if nr else 0.0,
             tpr=tp / Ut.sum() if Ut.sum() else np.nan,
             tprP=float((rej & Pt).sum() / Pt.sum()) if Pt.sum() else np.nan,
             tprA=float((rej & A).sum() / A.sum()) if A.sum() else np.nan)
    if p is not None:
        pn = np.asarray(p, float)[~Ut]
        pn = pn[np.isfinite(pn)]
        d["typeI"] = float((pn < 0.05).mean()) if len(pn) else np.nan
    return d


def r_method(method, dsdir, nperm=10000):
    out = os.path.join(dsdir, f"p_{method}.csv")
    if os.path.exists(out):
        os.remove(out)
    r = subprocess.run([RS, WRAPR, method, dsdir, out, str(nperm)],
                       capture_output=True, text=True, timeout=3600)
    if not os.path.exists(out):
        print(f"{method} FAILED:", r.stderr[-300:], flush=True)
        return None
    return pd.read_csv(out)["p"].to_numpy()


def one_dataset(cell, rep, modes):
    from generators_ext import generate_ext
    from twochannel import fit_detection_curves

    spec = CELLS[cell]
    seed = np.random.SeedSequence([20260819, cell]).spawn(20)[rep]
    Y, tr = generate_ext(spec["mech"], spec["params"], spec["n"], 100,
                         spec["depth"], seed=seed)
    g = tr["group"].astype(int)
    A = tr["abs_da_truth"].astype(bool)
    Pt = tr.get("pres_da_truth", np.zeros(100, dtype=bool)).astype(bool)
    Ut = A | Pt
    lib = Y.sum(axis=1).astype(float)
    n = len(g)
    D = (Y > 0).astype(float)
    dsdir = os.path.join(DSROOT, f"{cell}_{rep}")
    os.makedirs(dsdir, exist_ok=True)
    if not os.path.exists(os.path.join(dsdir, "Y.csv")):
        np.savetxt(os.path.join(dsdir, "Y.csv"), Y, fmt="%d", delimiter=",")
        pd.DataFrame({"group": g, "N": lib}).to_csv(
            os.path.join(dsdir, "meta.csv"), index=False)

    rows = []

    def add(mode, arm, p, attrib=None, secs=np.nan):
        rej = bh(p)
        d = score(rej, Ut, A, Pt, p=p)
        r = dict(cell=cell, rep=rep, mode=mode, arm=arm,
                 secs=round(secs, 1),
                 nfail=int((~np.isfinite(np.asarray(p, float))).sum()), **d)
        if attrib is not None:
            r["det_share"] = float((rej & (attrib == "det")).sum() /
                                   max(rej.sum(), 1))
        rows.append(r)

    # shared ZOLA parts (label-blind)
    t0 = time.time()
    det = fit_detection_curves(D, lib)
    qhat, det_ok = det["qhat"], det["active"]
    fit_secs = time.time() - t0
    np.savetxt(os.path.join(dsdir, "qhat.csv"), qhat, delimiter=",")
    x = np.where(g == 1, 0.5, -0.5)
    P = perm_idx(n, K, [20260819, 78, cell, rep])
    Xl = np.vstack([x[None, :], x[P]])            # (K+1, n) permuted labels
    Xc = Xl - Xl.mean(axis=1, keepdims=True)
    R1 = D - qhat
    R1[:, ~det_ok] = 0.0
    U_det = Xc @ R1

    # ---------------- mode 1: completion --------------------------------
    if "complete" in modes:
        # Wilcoxon channel
        t0 = time.time()
        rel = Y / np.maximum(lib[:, None], 1.0)
        Rk = np.empty_like(rel)
        for j in range(rel.shape[1]):
            Rk[:, j] = rankdata(rel[:, j])
        U_w = (Xl > 0).astype(float) @ Rk             # group-1 rank sums
        ext_ok = Rk.std(axis=0) > 1e-12
        p_after, attrib = completed_p(U_w, U_det, det_ok, ext_ok)
        add("complete_wilcoxon", "after", p_after, attrib,
            secs=time.time() - t0 + fit_secs)
        # LinDA-replica channel
        t0 = time.time()
        Wl = np.log((Y + 0.5) / np.maximum(lib[:, None], 1.0))
        xc_var = x.var()
        U_l = np.empty((K + 1, Y.shape[1]))
        for k in range(K + 1):
            tk, _ = _linda_t(Wl, Xl[k], xc_var)
            U_l[k] = tk
        p_after_l, attrib_l = completed_p(U_l, U_det, det_ok,
                                          np.ones(Y.shape[1], bool))
        add("complete_linda", "after", p_after_l, attrib_l,
            secs=time.time() - t0)

    # ---- mode 1b (post-hoc, Amendment W2-A1): positive-part Wilcoxon ----
    if "complete_pos" in modes:
        from twochannel import median_ratio_offset
        from scipy.stats import rankdata as _rd

        def poswil(norm, tag):
            t0 = time.time()
            p_taxa = Y.shape[1]
            U_pw = np.zeros((K + 1, p_taxa))
            ext_ok = np.zeros(p_taxa, dtype=bool)
            for j in range(p_taxa):
                pos = Y[:, j] > 0
                npos = int(pos.sum())
                if npos < 6:
                    continue
                r = rankdata(Y[pos, j] / norm[pos])
                ind = (Xl[:, pos] > 0).astype(float)  # (K+1, npos)
                cnt = ind.sum(axis=1)
                if cnt[0] < 3 or (npos - cnt[0]) < 3:
                    continue
                U_pw[:, j] = ind @ r - cnt * r.mean()
                ext_ok[j] = True
            Z = (U_pw - U_pw.mean(axis=0)) / np.where(
                U_pw.std(axis=0) < 1e-12, 1.0, U_pw.std(axis=0))
            pw_p = _rd(-(Z ** 2), method="max", axis=0)[0] / (K + 1.0)
            pw_p = np.where(ext_ok, pw_p, 1.0)
            add("poswilcoxon", f"before{tag}", pw_p, secs=time.time() - t0)
            p_after, attrib = completed_p(U_pw, U_det, det_ok, ext_ok)
            add("poswilcoxon", f"after{tag}", p_after, attrib)

        poswil(lib, "")                                # library-normalized
        poswil(median_ratio_offset(Y, n_ref=50), "_anch")  # poscounts anchor

    # ---------------- mode 2: ZINQ adjustment ---------------------------
    if "zinq" in modes:
        for meth, arm in (("zinq_ln", "logN"), ("zinq_zola", "zola")):
            t0 = time.time()
            pm = r_method(meth, dsdir)
            if pm is not None:
                add("zinq_adj", arm, pm, secs=time.time() - t0)

    # ---------------- mode 3: LOCOM filter ------------------------------
    if "locom" in modes:
        # k matched to the default-filter kept count from WRAP-01
        old = os.path.join(dsdir, f"p_locom_{cell}_{rep}.csv")
        k = 39
        if os.path.exists(old):
            pv0 = pd.read_csv(old)["p"].to_numpy()
            k = int(np.isfinite(pv0).sum()) or 39
        dbar = D.mean(axis=0)
        wdet = n * dbar * (1 - dbar)
        keep = np.argsort(-wdet)[:k] + 1              # 1-based for R
        np.savetxt(os.path.join(dsdir, "keep_zola.txt"), np.sort(keep),
                   fmt="%d")
        t0 = time.time()
        pm = r_method("locom_zola", dsdir)
        if pm is not None:
            add("locom_filter", "zola_k%d" % k, pm, secs=time.time() - t0)

    return rows


def main(cells, reps, modes, out_prefix, workers=2):
    import multiprocessing as mp
    jobs = [(c, r) for c in cells for r in reps]
    t0 = time.time()
    allrows = []
    if workers == 1:
        for c, r in jobs:
            allrows.extend(one_dataset(c, r, modes))
            print(f"[{c},{r}] {time.time()-t0:.0f}s", flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            args = [(c, r, modes) for c, r in jobs]
            for i, rws in enumerate(pool.starmap(one_dataset, args)):
                allrows.extend(rws)
                print(f"[{i+1}/{len(jobs)}] {time.time()-t0:.0f}s",
                      flush=True)
    df = pd.DataFrame(allrows)
    df.to_csv(f"/home/claude/ch_smoke/{out_prefix}_detail.csv", index=False)
    s = df.groupby(["cell", "mode", "arm"]).agg(
        fdp=("fdp", "mean"), fdp_sem=("fdp", "sem"),
        tpr=("tpr", "mean"), tpr_sem=("tpr", "sem"),
        tprA=("tprA", "mean"), tprP=("tprP", "mean"),
        typeI=("typeI", "mean"), nrej=("nrej", "mean"),
        nfail=("nfail", "mean"), secs=("secs", "mean")).round(4)
    s.to_csv(f"/home/claude/ch_smoke/{out_prefix}_summary.csv")
    print(s.to_string())
    return df


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        main([2002], [0], ["complete", "zinq", "locom"], "wrap2_smoke",
             workers=1)
    else:
        modes = sys.argv[2].split(",")
        cells = [int(c) for c in sys.argv[3].split(",")] if len(sys.argv) > 3 \
            else [2001, 2002, 2003]
        reps = list(range(int(sys.argv[4]))) if len(sys.argv) > 4 \
            else list(range(20))
        wk = int(sys.argv[5]) if len(sys.argv) > 5 else 2
        main(cells, reps, modes, f"wrap2_{mode}", workers=wk)
