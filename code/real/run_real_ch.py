"""run_real_ch.py — two-channel reanalysis of the three cohorts + spike-ins.

Conventions (mirror v3.6/v3.7 prep for comparability; SPEC-CH A3 defaults):
  * top-100 prevalence genera per cohort;
  * observed depths as N (real-data convention); intensity offset =
    masked median-ratio (A3 default), observed-depth offset as sensitivity;
  * permutation designs: ibdmdb subject-cluster (diagnosis is subject-level),
    mbqc BL4-vs-BL6 stratified by handling-lab letter (field 3 of the sample
    id), agp plain labels; K=999;
  * diagnostics reported per cohort: depth~group, library~group (Theorem A);
  * spike-ins (per V37 + new presence design): 15 taxa (5 high/5 mid/5 low
    prevalence, seeded), INTENSITY arm = case positives x2 (8 taxa) or x4
    (7 taxa); PRESENCE arm = case present-cells deleted w.p. 0.5.
    Recovery = injected taxa rejected by BH(0.05); false alarms tracked on
    the non-injected complement relative to the baseline rejection list.
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, time, json
import numpy as np
import pandas as pd
from scipy.stats import ranksums

sys.path.insert(0, "/home/claude/ch_smoke/code/simulation_v3")
sys.path.insert(0, "/home/claude/ch_smoke")
from twochannel import two_channel_test, bh_reject, median_ratio_offset

UP = "/mnt/user-data/uploads"
OUT = "/home/claude/ch_smoke/real"
os.makedirs(OUT, exist_ok=True)
K = 999
RNG = np.random.default_rng(20260819)


def top100(Y):
    prev = (Y > 0).mean(0)
    keep = np.argsort(-prev)[:100]
    return np.sort(keep)


def run_cohort(name, Y, N, group, taxa, strata=None, clusters=None,
               seed_tag=0):
    keep = top100(Y)
    Yk = Y[:, keep]; tk = taxa[keep]
    nu = median_ratio_offset(Yk)
    res = two_channel_test(Yk, N, group, nu=nu, K=K,
                           seed=[20260820, seed_tag],
                           strata=strata, clusters=clusters)
    rej = bh_reject(res["p_comb"], 0.05)
    out = pd.DataFrame(dict(taxon=tk, p_comb=res["p_comb"],
                            p_det=res["p_det"], p_int=res["p_int"],
                            z_det=res["z_det"], z_int=res["z_int"],
                            channel=res["attribution"], rejected=rej))
    out.to_csv(f"{OUT}/real_{name}_taxa.csv", index=False)
    dg = ranksums(N[group == 1], N[group == 0]).pvalue
    lib = Yk.sum(1); lg = ranksums(lib[group == 1], lib[group == 0]).pvalue
    summ = dict(cohort=name, n=len(group), n1=int((group == 1).sum()),
                n_rej=int(rej.sum()),
                rej_det=int((rej & (out.channel == "det")).sum()),
                rej_int=int((rej & (out.channel == "int")).sum()),
                depth_group_p=round(dg, 4), lib_group_p=round(lg, 4),
                phi_hat=res["phi_hat"])
    print(name, summ, flush=True)
    if rej.sum():
        print(out[rej].sort_values("p_comb").head(20).to_string(index=False),
              flush=True)
    return keep, res, out, summ


def spike(name, Y, N, group, taxa, keep, base_rej, strata=None,
          clusters=None, seed_tag=0):
    """Intensity and presence spike-ins on the top-100 submatrix."""
    Yk = Y[:, keep].copy(); tk = taxa[keep]
    prev = (Yk > 0).mean(0)
    order = np.argsort(-prev)
    tiers = [order[:33], order[33:66], order[66:]]
    r = np.random.default_rng([20260820, 99, seed_tag])
    sel = np.concatenate([r.choice(t, 5, replace=False) for t in tiers])
    rows = []
    for arm in ("intensity", "presence"):
        Ys = Yk.copy()
        case = group == 1
        for i, j in enumerate(sel):
            if arm == "intensity":
                f = 2 if i % 2 == 0 else 4
                m = case & (Ys[:, j] > 0)
                Ys[m, j] = np.round(Ys[m, j] * f).astype(Ys.dtype)
            else:
                m = case & (Ys[:, j] > 0)
                drop = r.random(m.sum()) < 0.5
                idx = np.where(m)[0][drop]
                Ys[idx, j] = 0
        nu = median_ratio_offset(Ys)
        res = two_channel_test(Ys, N, group, nu=nu, K=K,
                               seed=[20260821, seed_tag, hash(arm) % 1000],
                               strata=strata, clusters=clusters)
        rej = bh_reject(res["p_comb"], 0.05)
        rec = rej[sel].mean()
        extra = int((rej & ~base_rej).sum()) - int(rej[sel].sum())
        rows.append(dict(cohort=name, arm=arm, recovered=round(float(rec), 3),
                         n_rec=int(rej[sel].sum()),
                         extra_rej_nonspike=max(extra, 0),
                         rej_total=int(rej.sum())))
        print(rows[-1], flush=True)
    return rows


def main(which):
    all_summ, all_spike = [], []

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
        group = np.where(pd.Series(diag).isin(["CD", "UC"]).values, 1, 0)
        keep, res, out, s = run_cohort("ibdmdb", Y, N, group, taxa,
                                       clusters=subj, seed_tag=1)
        base_rej = out.rejected.values
        all_summ.append(s)
        all_spike += spike("ibdmdb", Y, N, group, taxa, keep, base_rej,
                           clusters=subj, seed_tag=1)

    if "mbqc" in which:
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
        keep, res, out, s = run_cohort("mbqc", Ym, Nm, gm, taxa,
                                       strata=st, seed_tag=2)
        base_rej = out.rejected.values
        all_summ.append(s)
        all_spike += spike("mbqc", Ym, Nm, gm, taxa, keep, base_rej,
                           strata=st, seed_tag=2)

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
        g = has[idx].astype(int)
        r = np.random.default_rng(20260304)
        n1 = min(350, int(g.sum()))
        sub = np.concatenate([
            r.choice(idx[g == 0], min(350, int((g == 0).sum())), replace=False),
            r.choice(idx[g == 1], n1, replace=False)])
        sub = np.sort(sub)
        Ya, Na = Y[sub], N[sub]
        ga = has[sub].astype(int)
        keep, res, out, s = run_cohort("agp", Ya, Na, ga, taxa, seed_tag=3)
        base_rej = out.rejected.values
        all_summ.append(s)
        all_spike += spike("agp", Ya, Na, ga, taxa, keep, base_rej,
                           seed_tag=3)

    pd.DataFrame(all_summ).to_csv(f"{OUT}/real_summary.csv", index=False)
    pd.DataFrame(all_spike).to_csv(f"{OUT}/real_spikein.csv", index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["ibdmdb", "mbqc", "agp"]
    main(which)
