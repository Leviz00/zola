"""run_rev_cluster.py -- SPEC-REV-COMPUTE R1 driver.

LinDA mixed-effects (~grp+(1|subj)) and LDM cluster mode on the SAME
IBDMDB top-100 matrices and spike constructions as EXT / the official
K=9999 analysis. Writes rev/rev_cluster_summary.csv.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import subprocess, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude/ch_smoke")
import run_rich as rr
from twochannel import bh_reject

OUT = "/home/claude/ch_smoke/rev"
DS = "/home/claude/ch_smoke/rev_ds"
os.makedirs(OUT, exist_ok=True)
os.makedirs(DS, exist_ok=True)
RS = "/usr/bin/Rscript"
SCRIPT = "/home/claude/ch_smoke/rev_cluster.R"

Y, N, g, taxa, st, cl, tag = rr.load_cohorts(("ibdmdb",))["ibdmdb"]
prev = (Y > 0).mean(0)
keep = np.sort(np.argsort(-prev)[:100])
Yk = Y[:, keep]
tk = np.asarray(taxa[keep]).astype(str)

# spike selection: official seeds (identical to EXT / run_real_k10k)
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

rows = []
for ds, Yd in datasets.items():
    dsdir = os.path.join(DS, f"ibdmdb_{ds}")
    os.makedirs(dsdir, exist_ok=True)
    np.savetxt(os.path.join(dsdir, "Y.csv"), Yd, fmt="%d", delimiter=",")
    pd.DataFrame({"group": g, "N": N, "subject": cl}).to_csv(
        os.path.join(dsdir, "meta.csv"), index=False)
    for meth in ("linda_mix", "ldm_cluster"):
        out = os.path.join(dsdir, f"p_{meth}.csv")
        if os.path.exists(out):
            os.remove(out)
        t0 = time.time()
        rr_ = subprocess.run([RS, SCRIPT, meth, dsdir, out],
                             capture_output=True, text=True, timeout=3600)
        secs = time.time() - t0
        if not os.path.exists(out):
            print(f"{ds}/{meth} FAILED:", rr_.stderr[-300:], flush=True)
            rows.append(dict(dataset=ds, method=meth, n_rej=-1,
                             spike_rec=np.nan, nfail=-1, secs=round(secs, 1),
                             note="FAILED"))
            continue
        pv = pd.read_csv(out)["p"].to_numpy()
        rej = bh_reject(pv, 0.05)
        rows.append(dict(dataset=ds, method=meth, n_rej=int(rej.sum()),
                         spike_rec=int(rej[sel].sum()) if ds != "native"
                         else np.nan,
                         nfail=int(np.isnan(pv).sum()),
                         secs=round(secs, 1),
                         note=";".join(tk[rej][:6])))
        print(rows[-1], flush=True)

pd.DataFrame(rows).to_csv(f"{OUT}/rev_cluster_summary.csv", index=False)
print("saved rev_cluster_summary.csv")
