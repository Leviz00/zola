"""run_fullspec.py -- SPEC-REV-COMPUTE R3: full-spectrum cohort analysis.

No top-100 filter: every genus in the constructed tables enters; the
channels' own activity gates are the abstention mechanism. Arms:
  unadj  -- plain BH over all taxa (primary)
  wdet   -- W-det weighted BH (sensitivity; naive information proxy)
  rich   -- richness-diagnostic column (as official)
K = 9999, same cohort subsamples/units as the official analysis.
"""
import os, sys, time
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude/ch_smoke")
import run_rich as rr
from twochannel import fit_detection_curves, median_ratio_offset, bh_reject
from run_wrap import weighted_bh

K = int(sys.argv[1]) if len(sys.argv) > 1 else 9999
rr.K = K
OUT = "/home/claude/ch_smoke/rev"
os.makedirs(OUT, exist_ok=True)

official = pd.read_csv("/home/claude/ch_smoke/real10k/real10k_taxa.csv")

rows, taxa_rows = [], []
for name, (Y, N, g, taxa, st, cl, tag) in rr.load_cohorts(
        ("ibdmdb", "mbqc", "agp")).items():
    t0 = time.time()
    tot = Y.sum(0)
    nz = tot > 0
    Yf, tf = Y[:, nz], np.asarray(taxa)[nz].astype(str)
    n, p = Yf.shape
    prev = (Yf > 0).mean(0)
    top100 = set(np.asarray(taxa)[np.sort(np.argsort(-(Y > 0).mean(0))[:100])
                                  ].astype(str))
    off = official[(official.cohort == name) & official.rejected]
    off_set = set(off.taxon.astype(str))

    D = (Yf > 0)
    nu = median_ratio_offset(Yf)
    det = fit_detection_curves(D.astype(float), N)
    qhat = det["qhat"]
    pi_hat = np.asarray(det["pi"], float)
    ridge_frac = float((pi_hat > 0.99).mean())
    mr, nfail_rich = rr.m_rich(Yf, N, qhat)

    res_by = {}
    for arm, m in (("unadj", qhat), ("rich", mr)):
        res = rr.two_channel_m(Yf, N, g, m, nu, seed=[20260829, tag],
                               strata=st, clusters=cl)
        res_by[arm] = res
    resU = res_by["unadj"]
    det_ok = D.sum(0) >= 3
    det_ok &= (~D).sum(0) >= 3
    int_ok = np.array([((Yf[:, j] > 0) & (g == 1)).sum() >= 3 and
                       ((Yf[:, j] > 0) & (g == 0)).sum() >= 3
                       for j in range(p)])
    tested = det_ok | int_ok
    dbar = D.mean(0)
    Wdet = n * dbar * (1 - dbar)

    arms = {
        "unadj": bh_reject(resU["p_comb"], 0.05),
        "wdet": weighted_bh(resU["p_comb"], Wdet),
        "rich": bh_reject(res_by["rich"]["p_comb"], 0.05),
    }
    for arm, rej in arms.items():
        newset = set(tf[rej])
        rows.append(dict(
            cohort=name, arm=arm, K=K,
            p_total=int(Y.shape[1]), p_nonzero=p,
            n_det_ok=int(det_ok.sum()), n_int_ok=int(int_ok.sum()),
            n_abstain=int((~tested).sum()),
            ridge_frac=round(ridge_frac, 3),
            n_rej=int(rej.sum()),
            in_top100=sum(1 for t in newset if t in top100),
            beyond_top100=sum(1 for t in newset if t not in top100),
            overlap_official=len(off_set & newset),
            official_n=len(off_set),
            core_akk=int("Akkermansia" in newset) if name == "agp" else -1,
            core_camp=int("Campylobacter" in newset) if name == "agp" else -1,
            nfail_rich=nfail_rich,
            secs=round(time.time() - t0, 1)))
        print(rows[-1], flush=True)
    rejU = arms["unadj"]
    for j in np.where(rejU | arms["rich"] | arms["wdet"])[0]:
        taxa_rows.append(dict(
            cohort=name, taxon=tf[j], prevalence=round(float(prev[j]), 4),
            in_top100=tf[j] in top100,
            p_comb=float(resU["p_comb"][j]),
            channel=str(resU["attribution"][j]),
            rej_unadj=bool(rejU[j]), rej_wdet=bool(arms["wdet"][j]),
            rej_rich=bool(arms["rich"][j]),
            was_official=tf[j] in off_set))

pd.DataFrame(rows).to_csv(f"{OUT}/fullspec_summary.csv", index=False)
pd.DataFrame(taxa_rows).to_csv(f"{OUT}/fullspec_taxa.csv", index=False)
print("saved fullspec_summary.csv / fullspec_taxa.csv")
