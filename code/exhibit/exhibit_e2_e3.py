#!/usr/bin/env python3
"""E2: Table-3 paired bootstrap CIs from archived wrap detail CSVs.
E3: Clopper-Pearson 95% CIs for spike recovery x/15.
Per SPEC_EXHIBIT (frozen 2026-08-23)."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, beta
import os

A = "/home/claude/zola/archives"
OUT = "/home/claude/zola/exhibit"
os.makedirs(OUT, exist_ok=True)
CELLS = [2001, 2002, 2003]
BOOT_B = 10000
BOOT_SEED = 20260823

# ---- load detail tables -------------------------------------------------
main = pd.read_csv(f"{A}/wrap_results/wrap_main_detail.csv")
ldm = pd.read_csv(f"{A}/wrap_results/wrap_ldm_detail.csv")
des = pd.read_csv(f"{A}/ZOLA_rev_compute_results/wrap_deseq2_detail.csv")
anc = pd.read_csv(f"{A}/ZOLA_rev_compute_results/rev/anc_wrap_detail.csv")
anc["method"] = "ancombc2"
anc["weight"] = np.where(anc.arm == "A2", "Wdet", "none")

d = pd.concat([main, ldm, des, anc], ignore_index=True)

BASE = {"zinq": "A0", "deseq2": "A0", "ancombc2": "A0",
        "locom": "A1n", "ldm": "A1n"}
ORDER = ["twochannel", "zinq", "ldm", "locom", "wilcoxon", "linda",
         "deseq2", "ancombc2"]
# official pooled numbers to reproduce (manuscript Table 3)
OFFICIAL = {"twochannel": (0.023, 11, 0), "zinq": (0.020, 10, 0),
            "ldm": (0.027, 7, 0), "locom": (0.002, 2, 1),
            "wilcoxon": (0.015, 7, 0), "linda": (0.005, 2, 0),
            "deseq2": (0.037, 12, 0), "ancombc2": (-0.002, 2, 2)}

rng = np.random.default_rng(BOOT_SEED)
rows = []
for meth in ORDER:
    base_arm = BASE.get(meth, "A1")
    b = d[(d.method == meth) & (d.arm == base_arm) & (d.weight == "none")
          & (d.cell.isin(CELLS))]
    w = d[(d.method == meth) & (d.arm == "A2") & (d.weight == "Wdet")
          & (d.cell.isin(CELLS))]
    m = b.merge(w, on=["cell", "rep"], suffixes=("_b", "_w"))
    diff = (m.tpr_w - m.tpr_b).values
    n = len(diff)
    mean = diff.mean()
    pos, neg = int((diff > 1e-12).sum()), int((diff < -1e-12).sum())
    # percentile bootstrap of the mean paired difference
    idx = rng.integers(0, n, size=(BOOT_B, n))
    bm = diff[idx].mean(axis=1)
    lo, hi = np.percentile(bm, [2.5, 97.5])
    try:
        pw = wilcoxon(diff[np.abs(diff) > 1e-12],
                      alternative="greater").pvalue if pos + neg else 1.0
    except ValueError:
        pw = 1.0
    off = OFFICIAL[meth]
    ok_mean = abs(round(mean, 3) - off[0]) < 1e-9
    ok_sign = (pos, neg) == (off[1], off[2])
    rows.append(dict(method=meth, base=base_arm, n_pairs=n,
                     dtpr_mean=round(mean, 4), ci_lo=round(lo, 4),
                     ci_hi=round(hi, 4), pos=pos, neg=neg,
                     p_wilcoxon=round(pw, 5),
                     fdp_w=round(m.fdp_w.mean(), 3),
                     bridge_mean=("OK" if ok_mean else
                                  f"MISMATCH({round(mean,3)} vs {off[0]})"),
                     bridge_sign=("OK" if ok_sign else
                                  f"MISMATCH({pos}+/{neg}- vs "
                                  f"{off[1]}+/{off[2]}-")))
t = pd.DataFrame(rows)
t.to_csv(f"{OUT}/tab3_ci.csv", index=False)
print(t.to_string(index=False))

# ---- E3: spike Clopper-Pearson ------------------------------------------
def cp(x, n, a=0.05):
    lo = beta.ppf(a / 2, x, n - x + 1) if x > 0 else 0.0
    hi = beta.ppf(1 - a / 2, x + 1, n - x) if x < n else 1.0
    return lo, hi

S = [("ibdmdb", "presence", 11), ("mbqc", "presence", 14),
     ("agp", "presence", 13), ("ibdmdb", "intensity", 0),
     ("mbqc", "intensity", 8), ("agp", "intensity", 8)]
# cross-check against archived real_spikein.csv (K=999-era counts identical
# to official K=9999 table 4) and real10k_summary
rs = pd.read_csv(f"{A}/ZOLA_results_bundle/real_spikein.csv")
r10 = pd.read_csv(f"{A}/real10k_results/real10k/real10k_summary.csv")
rows = []
for coh, arm, x in S:
    lo, hi = cp(x, 15)
    old = rs[(rs.cohort == coh) & (rs.arm == arm)].n_rec.values
    r10row = r10[(r10.cohort == coh) &
                 (r10.analysis == f"spike_{arm}_unadj")]
    k10 = int(r10row.rej_det.values[0]) if len(r10row) else None
    rows.append(dict(cohort=coh, arm=arm, x=x, n=15,
                     rate=round(x / 15, 3), ci_lo=round(lo, 3),
                     ci_hi=round(hi, 3),
                     check_k999=int(old[0]) if len(old) else None,
                     check_k10k=k10))
t3 = pd.DataFrame(rows)
t3.to_csv(f"{OUT}/spike_ci.csv", index=False)
print()
print(t3.to_string(index=False))
