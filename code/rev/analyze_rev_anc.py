"""analyze_rev_anc.py -- post-battery analysis for ANCOM-BC2:
1. wrap-style A0/A2 arms on cells 2001-2003 from saved raw p
   (BH / W-det weighted BH, mean-1 normalization), paired gains.
2. battery summary table (its shipped call) with MC SEMs.
"""
import os, sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, "/home/claude/ch_smoke")
from run_fix3_v4 import FIX_CELLS
from generators_ext import generate_ext
from run_wrap import weighted_bh, bh

DS = "/home/claude/ch_smoke/rev_ds/anc"
rows = []
for cell in (2001, 2002, 2003):
    spec = FIX_CELLS[cell]
    for rep in range(20):
        f = os.path.join(DS, f"{cell}_{rep}", "p_ancombc.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f)
        pv = d["p"].to_numpy()
        seed = np.random.SeedSequence([20260819, cell]).spawn(20)[rep]
        Y, tr = generate_ext(spec["mech"], spec["params"], spec["n"], 100,
                             spec["depth"], seed=seed)
        A = tr["abs_da_truth"].astype(bool)
        Pt = tr.get("pres_da_truth", np.zeros(100, dtype=bool)).astype(bool)
        Ut = A | Pt
        D = (Y > 0)
        n = Y.shape[0]
        dbar = D.mean(0)
        Wdet = n * dbar * (1 - dbar)
        for arm, rej in (("A0", bh(pv)), ("A2", weighted_bh(pv, Wdet))):
            nr = int(rej.sum())
            fp = int((rej & ~Ut).sum()); tp = int((rej & Ut).sum())
            rows.append(dict(cell=cell, rep=rep, arm=arm, nrej=nr,
                             fdp=fp / nr if nr else 0.0,
                             tpr=tp / Ut.sum()))

d = pd.DataFrame(rows)
d.to_csv("/home/claude/ch_smoke/rev/anc_wrap_detail.csv", index=False)
print("== per-cell arm means (raw-p BH vs +Wdet) ==")
print(d.groupby(["cell", "arm"])[["fdp", "tpr"]].mean().round(3))
print("\n== paired pooled delta (A2 - A0) ==")
b = d[d.arm == "A0"].set_index(["cell", "rep"])
a = d[d.arm == "A2"].set_index(["cell", "rep"])
common = b.index.intersection(a.index)
dt = (a.loc[common, "tpr"] - b.loc[common, "tpr"])
pos, neg = int((dt > 0).sum()), int((dt < 0).sum())
try:
    pw = wilcoxon(dt, alternative="greater").pvalue if (dt != 0).any() else 1.0
except Exception:
    pw = np.nan
print(f"mean dTPR {dt.mean():+.4f} ({pos}+/{neg}-)  wilcoxon p={pw:.4g}")
print(f"A2 mean FDP {a.loc[common,'fdp'].mean():.3f}")
