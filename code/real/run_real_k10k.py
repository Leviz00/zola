"""run_real_k10k.py -- Amendment A4: full real-data rerun at K=9999.

Per cohort: native (unadj primary + richness-diagnostic column),
intensity spike arm, presence spike arm (unadj + rich column),
richness~group community test. Same seeds/subsamples/units as REAL-CH.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, time
import numpy as np
import pandas as pd
from scipy.stats import ranksums

sys.path.insert(0, "/home/claude/ch_smoke")
import run_rich as rr
from twochannel import fit_detection_curves, median_ratio_offset, bh_reject

rr.K = 9999
OUT = "/home/claude/ch_smoke/real10k"
os.makedirs(OUT, exist_ok=True)


def analyze(name, Yk, N, g, tk, st, cl, tag):
    rows, taxa_rows = [], []
    nu = median_ratio_offset(Yk)
    D = (Yk > 0)
    rich = D.sum(1)
    rich_p = ranksums(rich[g == 1], rich[g == 0]).pvalue
    dg = ranksums(N[g == 1], N[g == 0]).pvalue
    lib = Yk.sum(1)
    lg = ranksums(lib[g == 1], lib[g == 0]).pvalue
    qhat = fit_detection_curves(D.astype(float), N)["qhat"]
    mr, _ = rr.m_rich(Yk, N, qhat)
    res_by_arm = {}
    for arm, m in (("unadj", qhat), ("rich", mr)):
        t0 = time.time()
        res = rr.two_channel_m(Yk, N, g, m, nu, seed=[20260826, tag],
                               strata=st, clusters=cl)
        rej = bh_reject(res["p_comb"], 0.05)
        res_by_arm[arm] = (res, rej)
        rows.append(dict(cohort=name, analysis=f"native_{arm}",
                         n_rej=int(rej.sum()),
                         rej_det=int((rej & (res["attribution"] ==
                                             "det")).sum()),
                         rej_int=int((rej & (res["attribution"] ==
                                             "int")).sum()),
                         rich_group_p=round(rich_p, 6),
                         depth_group_p=round(dg, 4),
                         lib_group_p=round(lg, 4),
                         secs=round(time.time() - t0, 1)))
        print(rows[-1], flush=True)
    resU, rejU = res_by_arm["unadj"]
    resR, rejR = res_by_arm["rich"]
    for j in range(len(tk)):
        if (resU["p_comb"][j] < 0.05) or rejU[j] or rejR[j]:
            taxa_rows.append(dict(
                cohort=name, taxon=str(tk[j]),
                p_comb=resU["p_comb"][j], p_det=resU["p_det"][j],
                p_int=resU["p_int"][j], z_det=0.0,
                channel=resU["attribution"][j], rejected=bool(rejU[j]),
                p_comb_rich=resR["p_comb"][j],
                p_det_rich=resR["p_det"][j],
                rejected_rich=bool(rejR[j])))

    # spike-ins (same construction/seeds as run_real_ch)
    prev = (Yk > 0).mean(0)
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
        arms = [("unadj", qs)]
        if armname == "presence":
            mrs, _ = rr.m_rich(Ys, N, qs)
            arms.append(("rich", mrs))
        for arm, m in arms:
            res = rr.two_channel_m(Ys, N, g, m, nus,
                                   seed=[20260827, tag,
                                         hash(armname) % 997],
                                   strata=st, clusters=cl)
            rej = bh_reject(res["p_comb"], 0.05)
            base = rejU
            extra = int((rej & ~base).sum()) - int(rej[sel].sum())
            rows.append(dict(cohort=name, analysis=f"spike_{armname}_{arm}",
                             n_rej=int(rej.sum()),
                             rej_det=int(rej[sel].sum()),
                             rej_int=round(float(rej[sel].mean()), 3),
                             rich_group_p=max(extra, 0),
                             depth_group_p=np.nan, lib_group_p=np.nan,
                             secs=0))
            print(rows[-1], flush=True)
    return rows, taxa_rows


def main():
    allr, allt = [], []
    for name, (Y, N, g, taxa, st, cl, tag) in rr.load_cohorts(
            ("ibdmdb", "mbqc", "agp")).items():
        prev = (Y > 0).mean(0)
        keep = np.sort(np.argsort(-prev)[:100])
        rows, taxa_rows = analyze(name, Y[:, keep], N, g, taxa[keep],
                                  st, cl, tag)
        allr += rows
        allt += taxa_rows
    pd.DataFrame(allr).to_csv(f"{OUT}/real10k_summary.csv", index=False)
    pd.DataFrame(allt).to_csv(f"{OUT}/real10k_taxa.csv", index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
