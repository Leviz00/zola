"""run_rev_anc_real.py -- ANCOM-BC2 on the three real cohorts (EXT protocol:
same top-100 matrices, same spike constructions/seeds). IBDMDB runs with
rand_formula='(1|subj)' (cluster-respecting, its shipped mixed mode);
MBQC/AGP sample-level. Scoring mirrors ext_summary columns.
"""
import os, sys, time
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import subprocess
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude/ch_smoke")
import run_rich as rr

OUT = "/home/claude/ch_smoke/rev"
DS = "/home/claude/ch_smoke/rev_ds/anc_real"
os.makedirs(OUT, exist_ok=True)
os.makedirs(DS, exist_ok=True)
RS = "/usr/bin/Rscript"
SCRIPT = "/home/claude/ch_smoke/rev_ancombc.R"

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

    for ds, Yd in datasets.items():
        dsdir = os.path.join(DS, f"{name}_{ds}")
        os.makedirs(dsdir, exist_ok=True)
        np.savetxt(os.path.join(dsdir, "Y.csv"), Yd, fmt="%d", delimiter=",")
        meta = {"group": g, "N": N}
        if name == "ibdmdb":
            meta["subject"] = cl
        pd.DataFrame(meta).to_csv(os.path.join(dsdir, "meta.csv"),
                                  index=False)
        out = os.path.join(dsdir, "p_ancombc.csv")
        if os.path.exists(out):
            os.remove(out)
        t0 = time.time()
        cmd = [RS, SCRIPT, dsdir, out]
        if name == "ibdmdb":
            cmd.append("subject")
        pr = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=3600)
        secs = time.time() - t0
        if not os.path.exists(out):
            print(f"{name}/{ds} FAILED: {pr.stderr[-250:]}", flush=True)
            rows.append(dict(cohort=name, dataset=ds, method="ancombc2",
                             mode="mixed" if name == "ibdmdb" else "fixed",
                             n_rej=-1, secs=round(secs, 1)))
            continue
        d = pd.read_csv(out)
        diff = d["diff"].to_numpy().astype(bool)
        newset = set(tk[diff])
        rows.append(dict(
            cohort=name, dataset=ds, method="ancombc2",
            mode="mixed" if name == "ibdmdb" else "fixed",
            n_rej=int(diff.sum()),
            n_tested=int(np.isfinite(d["p"].to_numpy()).sum()),
            overlap_official=len(off_set & newset),
            spike_rec=int(diff[sel].sum()) if ds != "native" else np.nan,
            hits_core=int(len({"Akkermansia", "Campylobacter"} & newset))
            if name == "agp" else np.nan,
            szero_n=int(d["szero"].sum()),
            secs=round(secs, 1)))
        print(rows[-1], flush=True)

pd.DataFrame(rows).to_csv(f"{OUT}/anc_real_summary.csv", index=False)
print("saved anc_real_summary.csv")
