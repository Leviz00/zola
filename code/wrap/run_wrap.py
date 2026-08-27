"""run_wrap.py -- SPEC-WRAP-01: three-arm wrap comparison of popular methods.

Cells 2001/2002/2003, final (A3) conventions, SAME datasets as the frozen
battery (SeedSequence([20260819, cell]).spawn(20)[rep]).  Arms per SPEC:
  A0 raw          : method's own p-values + BH(0.05)
  A1 perm         : K=999 label-permutation p of the method statistic + BH
                    (wilcoxon, linda-replica; locom/ldm are native-perm = A0)
  A2 +weights     : (A1 else A0) p-values + weighted BH, W-det and W-info
Two-channel reference row reuses the identical pipeline as fix3 (identity
cross-checked against fix3_v4_detail.csv).
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import subprocess, sys, time, json
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm, t as tdist, ranksums, gaussian_kde

sys.path.insert(0, "/home/claude/ch_smoke/code/simulation_v3")
sys.path.insert(0, "/home/claude/ch_smoke")

CELLS = {
    2001: dict(mech="three_layer_real", depth=20000, n=300,
               params=dict(da_fraction=0.0, presence_da_fraction=0.10,
                           presence_effect_or=0.25)),
    2002: dict(mech="three_layer_real", depth=20000, n=300,
               params=dict(effect_size=2.0, da_fraction=0.05,
                           presence_da_fraction=0.05,
                           presence_effect_or=0.25)),
    2003: dict(mech="three_layer_real", depth=20000, n=100,
               params=dict(effect_size=2.0, da_fraction=0.10)),
}
K = 999
ALPHA = 0.05
DSROOT = "/home/claude/ch_smoke/wrap_ds"
RS = "/usr/bin/Rscript"
WRAPR = "/home/claude/ch_smoke/wrap_methods.R"


def bh(p, alpha=ALPHA):
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    m = int(ok.sum())
    rej = np.zeros(len(p), dtype=bool)
    if m == 0:
        return rej
    idx = np.where(ok)[0]
    order = np.argsort(p[idx])
    thr = alpha * np.arange(1, m + 1) / m
    passed = p[idx][order] <= thr
    if passed.any():
        rej[idx[order[: np.max(np.where(passed)[0]) + 1]]] = True
    return rej


def weighted_bh(p, w, alpha=ALPHA):
    p = np.asarray(p, dtype=float); w = np.asarray(w, dtype=float)
    ok = np.isfinite(p) & (w > 0)
    rej = np.zeros(len(p), dtype=bool)
    m = int(ok.sum())
    if m == 0:
        return rej
    idx = np.where(ok)[0]
    wn = w[idx] * m / w[idx].sum()
    padj = p[idx] / wn
    order = np.argsort(padj)
    thr = alpha * np.arange(1, m + 1) / m
    passed = padj[order] <= thr
    if passed.any():
        rej[idx[order[: np.max(np.where(passed)[0]) + 1]]] = True
    return rej


# ---------------- statistics -----------------------------------------------

def perm_matrix(n, g, K, seed):
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    P = np.empty((K, n), dtype=np.int8)
    for k in range(K):
        P[k] = rng.permutation(g)
    return P


def wilcoxon_stats(Y, N, g, P):
    """Rank-sum z per taxon; asymptotic p and permutation p."""
    rel = Y / np.maximum(N[:, None], 1.0)
    n = len(g); n1 = int(g.sum()); n0 = n - n1
    R = np.empty_like(rel)
    for j in range(rel.shape[1]):
        R[:, j] = rankdata(rel[:, j])
    mu = n1 * (n + 1) / 2.0
    # tie-corrected variance per taxon
    sig = np.empty(rel.shape[1])
    for j in range(rel.shape[1]):
        _, cnt = np.unique(rel[:, j], return_counts=True)
        tie = (cnt ** 3 - cnt).sum()
        sig[j] = np.sqrt(n0 * n1 / 12.0 * ((n + 1) - tie / (n * (n - 1))))
    obs = g @ R
    z = (obs - mu) / np.maximum(sig, 1e-12)
    p_asym = 2 * norm.sf(np.abs(z))
    perm = P @ R                                   # K x p
    pz = np.abs(perm - mu)
    p_perm = (1.0 + (pz >= np.abs(obs - mu)[None, :] - 1e-12).sum(0)) / (P.shape[0] + 1.0)
    return p_asym, p_perm


def _linda_t(W, x, xc_var):
    """Per-taxon OLS slope t with mode bias-correction (LinDA replica)."""
    xc = x - x.mean()
    beta = (xc @ W) / (xc_var * len(x))
    fit = np.outer(xc, beta)
    resid = W - W.mean(0, keepdims=True) - fit
    df = len(x) - 2
    s2 = (resid ** 2).sum(0) / df
    se = np.sqrt(s2 / (xc_var * len(x)))
    # mode of beta via KDE argmax (LinDA bias correction)
    kde = gaussian_kde(beta)
    grid = np.linspace(beta.min(), beta.max(), 512)
    bias = grid[np.argmax(kde(grid))]
    return (beta - bias) / np.maximum(se, 1e-12), df


def linda_replica(Y, N, g, P):
    W = np.log((Y + 0.5) / np.maximum(N[:, None], 1.0))
    xc_var = g.var()
    tobs, df = _linda_t(W, g.astype(float), xc_var)
    p_asym = 2 * tdist.sf(np.abs(tobs), df)
    cnt = np.zeros(Y.shape[1])
    for k in range(P.shape[0]):
        tk, _ = _linda_t(W, P[k].astype(float), xc_var)
        cnt += (np.abs(tk) >= np.abs(tobs) - 1e-12)
    p_perm = (1.0 + cnt) / (P.shape[0] + 1.0)
    return tobs, p_asym, p_perm


def pydeseq2_p(Y, g):
    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except Exception:
        return None
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    counts = pd.DataFrame(Y, columns=[f"t{j}" for j in range(Y.shape[1])])
    meta = pd.DataFrame({"group": pd.Categorical(g.astype(str))})
    try:
        dds = DeseqDataSet(counts=counts, metadata=meta, design="~group",
                           quiet=True)
        dds.deseq2()
        st = DeseqStats(dds, contrast=["group", "1", "0"], quiet=True)
        st.summary()
        return st.results_df["pvalue"].to_numpy()
    except Exception as e:
        print("pydeseq2 fail:", e, flush=True)
        return None


def r_method(method, dsdir, tag, nperm=10000):
    out = os.path.join(dsdir, f"p_{method}_{tag}.csv")
    if os.path.exists(out):
        os.remove(out)
    r = subprocess.run([RS, WRAPR, method, dsdir, out, str(nperm)],
                       capture_output=True, text=True, timeout=3600)
    if not os.path.exists(out):
        print(f"[{tag}] {method} FAILED:", r.stderr[-500:], flush=True)
        return None
    return pd.read_csv(out)["p"].to_numpy()


# ---------------- per-dataset driver ---------------------------------------

def score(rej, Ut, A, Pt, p=None):
    nr = int(rej.sum()); fp = int((rej & ~Ut).sum()); tp = int((rej & Ut).sum())
    d = dict(nrej=nr, fdp=fp / nr if nr else 0.0,
             tpr=tp / Ut.sum() if Ut.sum() else np.nan,
             tprP=float((rej & Pt).sum() / Pt.sum()) if Pt.sum() else np.nan)
    if p is not None:
        pn = p[~Ut]
        pn = pn[np.isfinite(pn)]
        d["typeI"] = float((pn < 0.05).mean()) if len(pn) else np.nan
    return d


def one_dataset(cell, rep, methods, locom_nperm=10000):
    from generators_ext import generate_ext
    from twochannel import two_channel_test, median_ratio_offset

    spec = CELLS[cell]
    seed = np.random.SeedSequence([20260819, cell]).spawn(20)[rep]
    Y, tr = generate_ext(spec["mech"], spec["params"], spec["n"], 100,
                         spec["depth"], seed=seed)
    g = tr["group"].astype(int)
    A = tr["abs_da_truth"].astype(bool)
    Pt = tr.get("pres_da_truth", np.zeros(100, dtype=bool)).astype(bool)
    Ut = A | Pt
    lib = Y.sum(axis=1).astype(float)
    D = (Y > 0)
    n = len(g)
    dbar = D.mean(0)
    Wdet = n * dbar * (1 - dbar)
    Winfo = Wdet + D.sum(0)

    dsdir = os.path.join(DSROOT, f"{cell}_{rep}")
    os.makedirs(dsdir, exist_ok=True)
    np.savetxt(os.path.join(dsdir, "Y.csv"), Y, fmt="%d", delimiter=",")
    pd.DataFrame({"group": g, "N": lib}).to_csv(
        os.path.join(dsdir, "meta.csv"), index=False)

    P = perm_matrix(n, g, K, [20260819, 77, cell, rep])
    rows = []

    def add(method, arm, p, weight="none", w=None, secs=np.nan,
            typeI_from=None):
        rej = weighted_bh(p, w) if w is not None else bh(p)
        d = score(rej, Ut, A, Pt, p=typeI_from)
        rows.append(dict(cell=cell, rep=rep, method=method, arm=arm,
                         weight=weight, secs=round(secs, 1),
                         nfail=int((~np.isfinite(np.asarray(p, float))).sum()),
                         **d))

    # --- two-channel reference (identical to fix3 pipeline) ---
    if "twochannel" in methods:
        t0 = time.time()
        nu = median_ratio_offset(Y, n_ref=50)
        res = two_channel_test(Y, lib, g, nu=nu, K=K,
                               seed=[20260819, 9, cell, rep])
        pc = res["p_comb"]
        el = time.time() - t0
        add("twochannel", "A1", pc, secs=el, typeI_from=pc)
        add("twochannel", "A2", pc, "Wdet", Wdet)
        add("twochannel", "A2", pc, "Winfo", Winfo)

    # --- wilcoxon ---
    if "wilcoxon" in methods:
        t0 = time.time()
        p_asym, p_perm = wilcoxon_stats(Y, lib, g, P)
        el = time.time() - t0
        add("wilcoxon", "A0", p_asym, secs=el, typeI_from=p_asym)
        add("wilcoxon", "A1", p_perm, typeI_from=p_perm)
        add("wilcoxon", "A2", p_perm, "Wdet", Wdet)
        add("wilcoxon", "A2", p_perm, "Winfo", Winfo)

    # --- linda replica (perm) + R original (raw) ---
    if "linda" in methods:
        t0 = time.time()
        tobs, p_rep_asym, p_perm = linda_replica(Y, lib, g, P)
        el = time.time() - t0
        pr = r_method("linda", dsdir, f"{cell}_{rep}")
        if pr is not None:
            add("linda", "A0", pr, secs=el, typeI_from=pr)
            ok = np.isfinite(pr) & np.isfinite(p_rep_asym)
            corr = float(np.corrcoef(-np.log10(np.maximum(pr[ok], 1e-300)),
                                     -np.log10(np.maximum(p_rep_asym[ok],
                                                          1e-300)))[0, 1])
        else:
            add("linda", "A0", p_rep_asym, secs=el, typeI_from=p_rep_asym)
            corr = np.nan
        rows[-1]["replica_corr"] = corr
        add("linda", "A1", p_perm, typeI_from=p_perm)
        add("linda", "A2", p_perm, "Wdet", Wdet)
        add("linda", "A2", p_perm, "Winfo", Winfo)

    # --- pydeseq2 raw ---
    if "deseq2" in methods:
        t0 = time.time()
        pdq = pydeseq2_p(Y, g)
        if pdq is not None:
            el = time.time() - t0
            add("deseq2", "A0", pdq, secs=el, typeI_from=pdq)
            add("deseq2", "A2", pdq, "Wdet", Wdet)
            add("deseq2", "A2", pdq, "Winfo", Winfo)

    # --- R methods: zinq (raw+weights), locom/ldm (native-perm+weights) ---
    for meth in ("zinq", "locom", "ldm"):
        if meth not in methods:
            continue
        t0 = time.time()
        pm = r_method(meth, dsdir, f"{cell}_{rep}", nperm=locom_nperm)
        el = time.time() - t0
        if pm is None:
            continue
        arm0 = "A1n" if meth in ("locom", "ldm") else "A0"
        add(meth, arm0, pm, secs=el, typeI_from=pm)
        add(meth, "A2", pm, "Wdet", Wdet)
        add(meth, "A2", pm, "Winfo", Winfo)

    return rows


def main(cells, reps, methods, out_prefix, locom_nperm=10000, workers=3):
    import multiprocessing as mp
    jobs = [(c, r) for c in cells for r in reps]
    t0 = time.time()
    allrows = []
    if workers == 1:
        for c, r in jobs:
            allrows.extend(one_dataset(c, r, methods, locom_nperm))
            print(f"[{c},{r}] {time.time()-t0:.0f}s", flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            args = [(c, r, methods, locom_nperm) for c, r in jobs]
            for i, rws in enumerate(pool.starmap(one_dataset, args)):
                allrows.extend(rws)
                print(f"[{i+1}/{len(jobs)}] {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(allrows)
    df.to_csv(f"/home/claude/ch_smoke/{out_prefix}_detail.csv", index=False)
    gcols = ["cell", "method", "arm", "weight"]
    s = df.groupby(gcols).agg(
        fdp=("fdp", "mean"), fdp_sem=("fdp", "sem"),
        tpr=("tpr", "mean"), tpr_sem=("tpr", "sem"),
        tprP=("tprP", "mean"), typeI=("typeI", "mean"),
        nrej=("nrej", "mean"), nfail=("nfail", "mean"),
        secs=("secs", "mean")).round(4)
    s.to_csv(f"/home/claude/ch_smoke/{out_prefix}_summary.csv")
    print(s.to_string())
    return df


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if mode == "smoke":
        methods = sys.argv[2].split(",") if len(sys.argv) > 2 else [
            "wilcoxon", "linda", "deseq2", "zinq", "locom", "ldm"]
        main([2002], [0], methods, "wrap_smoke", workers=1)
    else:
        methods = sys.argv[2].split(",")
        cells = [int(c) for c in sys.argv[3].split(",")] if len(sys.argv) > 3 \
            else [2001, 2002, 2003]
        reps = list(range(int(sys.argv[4]))) if len(sys.argv) > 4 \
            else list(range(20))
        nperm = int(sys.argv[5]) if len(sys.argv) > 5 else 10000
        wk = int(sys.argv[6]) if len(sys.argv) > 6 else 3
        main(cells, reps, methods, f"wrap_{mode}", locom_nperm=nperm,
             workers=wk)
