"""Post-hoc resolution check (disclosed): rich arm at K=9999 on ibdmdb+agp."""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "/home/claude/ch_smoke")
import run_rich as rr
from twochannel import fit_detection_curves, median_ratio_offset, bh_reject

rr.K = 9999
for name, tup in rr.load_cohorts(("ibdmdb", "agp")).items():
    Y, N, g, taxa, st, cl, tag = tup
    prev = (Y > 0).mean(0)
    keep = np.sort(np.argsort(-prev)[:100])
    Yk, tk = Y[:, keep], taxa[keep]
    nu = median_ratio_offset(Yk)
    D = (Yk > 0)
    qhat = fit_detection_curves(D.astype(float), N)["qhat"]
    mr, _ = rr.m_rich(Yk, N, qhat)
    for arm, m in (("unadj", qhat), ("rich", mr)):
        res = rr.two_channel_m(Yk, N, g, m, nu, seed=[20260826, tag],
                               strata=st, clusters=cl)
        rej = bh_reject(res["p_comb"], 0.05)
        old = pd.read_csv(f"real/real_{name}_taxa.csv")
        old_rej = old[old.rejected].taxon.astype(str).tolist()
        print(f"\n{name} {arm} K=9999: n_rej={rej.sum()}")
        for t in old_rej:
            i = np.where(np.asarray(tk).astype(str) == t)[0]
            if len(i):
                print(f"  {t:<35} p_comb={res['p_comb'][i[0]]:.5f} "
                      f"p_det={res['p_det'][i[0]]:.5f} "
                      f"rej={bool(rej[i[0]])}")
