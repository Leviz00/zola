"""run_real_ext.py -- SPEC-EXT: external methods on the real cohorts.

Same matrices/subsamples/spike seeds as the official K=9999 analysis.
Methods: ZINQ/LOCOM/LDM/LinDA (R, defaults), TSS+Wilcoxon (perm with
design units, K=9999), DESeq2 (native only). Datasets per cohort:
native, spike_pres, spike_int.
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import subprocess, sys, time
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, "/home/claude/ch_smoke")
import run_rich as rr
from twochannel import bh_reject

OUT = "/home/claude/ch_smoke/ext"
DS = "/home/claude/ch_smoke/ext_ds"
os.makedirs(OUT, exist_ok=True)
os.makedirs(DS, exist_ok=True)
RS = "/usr/bin/Rscript"
WRAPR = "/home/claude/ch_smoke/wrap_methods.R"
KW = 9999


def wilcoxon_perm(Y, N, g, st, cl, seed):
    rel = Y / np.maximum(N[:, None], 1.0)
    n, p = rel.shape
    R = np.empty_like(rel)
    for j in range(p):
        R[:, j] = rankdata(rel[:, j])
    x = np.where(g == 1, 1.0, 0.0)
    X = rr.perm_labels(x, KW, seed, st, cl)
    U = X @ R - X.sum(1, keepdims=True) * (R.mean(0)[None, :])
    Z = (U - U.mean(0)) / np.where(U.std(0) < 1e-12, 1.0, U.std(0))
    pv = rankdata(-(Z ** 2), method="max", axis=0)[0] / (KW + 1.0)
    return pv


def r_method(meth, dsdir, nperm=10000):
    out = os.path.join(dsdir, f"p_{meth}.csv")
    if os.path.exists(out):
        os.remove(out)
    t0 = time.time()
    r = subprocess.run([RS, WRAPR, meth, dsdir, out, str(nperm)],
                       capture_output=True, text=True, timeout=3600)
    if not os.path.exists(out):
        print(f"{meth} FAILED:", r.stderr[-200:], flush=True)
        return None, time.time() - t0
    return pd.read_csv(out)["p"].to_numpy(), time.time() - t0


def main():
    official = pd.read_csv("/home/claude/ch_smoke/real10k/real10k_taxa.csv")
    rows = []
    for name, (Y, N, g, taxa, st, cl, tag) in rr.load_cohorts(
            ("ibdmdb", "mbqc", "agp")).items():
        prev = (Y > 0).mean(0)
        keep = np.sort(np.argsort(-prev)[:100])
        Yk = Y[:, keep]
        tk = np.asarray(taxa[keep]).astype(str)
        off = official[(official.cohort == name) & official.rejected]
        off_set = set(off.taxon.astype(str))
        # spike selection (official seeds)
        prev_k = (Yk > 0).mean(0)
        order = np.argsort(-prev_k)
        tiers = [order[:33], order[33:66], order[66:]]
        r = np.random.default_rng([20260820, 99, tag])
        sel = np.concatenate([r.choice(t, 5, replace=False) for t in tiers])
        datasets = {"native": Yk}
        for armname in ("int", "pres"):
            Ys = Yk.copy()
            case = g == 1
            for i, j in enumerate(sel):
                if armname == "int":
                    f = 2 if i % 2 == 0 else 4
                    m_ = case & (Ys[:, j] > 0)
                    Ys[m_, j] = np.round(Ys[m_, j] * f).astype(Ys.dtype)
                else:
                    m_ = case & (Ys[:, j] > 0)
                    drop = r.random(m_.sum()) < 0.5
                    Ys[np.where(m_)[0][drop], j] = 0
            datasets[f"spike_{armname}"] = Ys

        slow = set()
        for ds, Yd in datasets.items():
            dsdir = os.path.join(DS, f"{name}_{ds}")
            os.makedirs(dsdir, exist_ok=True)
            np.savetxt(os.path.join(dsdir, "Y.csv"), Yd, fmt="%d",
                       delimiter=",")
            pd.DataFrame({"group": g, "N": N}).to_csv(
                os.path.join(dsdir, "meta.csv"), index=False)
            for meth in ("linda", "zinq", "locom", "ldm"):
                if meth in slow and ds == "spike_int":
                    continue
                pv, secs = r_method(meth, dsdir)
                if pv is None:
                    continue
                if secs > 120:
                    slow.add(meth)
                rej = bh_reject(pv, 0.05)
                newset = set(tk[rej])
                rows.append(dict(
                    cohort=name, dataset=ds, method=meth,
                    n_rej=int(rej.sum()),
                    overlap_official=len(off_set & newset),
                    jaccard=round(len(off_set & newset) /
                                  max(len(off_set | newset), 1), 3),
                    spike_rec=int(rej[sel].sum()) if ds != "native"
                    else np.nan,
                    hits_core=int(len({"Akkermansia", "Campylobacter"} &
                                      newset)) if name == "agp" else np.nan,
                    nfail=int(np.isnan(pv).sum()),
                    secs=round(secs, 1)))
                print(rows[-1], flush=True)
            # wilcoxon (python, design units, K=9999)
            t0 = time.time()
            pv = wilcoxon_perm(Yd, N, g, st, cl,
                               [20260828, tag, hash(ds) % 997])
            rej = bh_reject(pv, 0.05)
            newset = set(tk[rej])
            rows.append(dict(cohort=name, dataset=ds, method="wilcoxon",
                             n_rej=int(rej.sum()),
                             overlap_official=len(off_set & newset),
                             jaccard=round(len(off_set & newset) /
                                           max(len(off_set | newset), 1), 3),
                             spike_rec=int(rej[sel].sum())
                             if ds != "native" else np.nan,
                             hits_core=int(len({"Akkermansia",
                                                "Campylobacter"} & newset))
                             if name == "agp" else np.nan,
                             nfail=0, secs=round(time.time() - t0, 1)))
            print(rows[-1], flush=True)
        # deseq2 native only
        try:
            from run_wrap import pydeseq2_p
            t0 = time.time()
            pdq = pydeseq2_p(Yk, g)
            if pdq is not None:
                rej = bh_reject(pdq, 0.05)
                newset = set(tk[rej])
                rows.append(dict(cohort=name, dataset="native",
                                 method="deseq2", n_rej=int(rej.sum()),
                                 overlap_official=len(off_set & newset),
                                 jaccard=round(len(off_set & newset) /
                                               max(len(off_set | newset),
                                                   1), 3),
                                 spike_rec=np.nan,
                                 hits_core=int(len({"Akkermansia",
                                                    "Campylobacter"} &
                                                   newset))
                                 if name == "agp" else np.nan,
                                 nfail=int(np.isnan(pdq).sum()),
                                 secs=round(time.time() - t0, 1)))
                print(rows[-1], flush=True)
        except Exception as e:
            print("deseq2 skipped:", e, flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/ext_summary.csv", index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
