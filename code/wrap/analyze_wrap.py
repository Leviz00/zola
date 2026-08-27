"""analyze_wrap.py -- SPEC-WRAP-01 reading-line analysis over all batches."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

frames = []
for f in ("wrap_main_detail.csv", "wrap_ldm_detail.csv",
          "wrap_deseq2_detail.csv"):
    try:
        frames.append(pd.read_csv(f"/home/claude/ch_smoke/{f}"))
    except FileNotFoundError:
        print("missing:", f)
d = pd.concat(frames, ignore_index=True)
d = d.drop_duplicates(subset=["cell", "rep", "method", "arm", "weight"])

# ---- L2: paired upstream net gain: A2 vs the unweighted arm it sits on ----
print("\n=== L2 upstream net gain (paired per dataset, Wdet) ===")
rows = []
for meth in d.method.unique():
    dm = d[d.method == meth]
    base_arm = {"zinq": "A0", "deseq2": "A0", "locom": "A1n",
                "ldm": "A1n"}.get(meth, "A1")
    for cell in sorted(dm.cell.unique()):
        b = dm[(dm.cell == cell) & (dm.arm == base_arm) &
               (dm.weight == "none")].set_index("rep")
        for wt in ("Wdet", "Winfo"):
            a = dm[(dm.cell == cell) & (dm.arm == "A2") &
                   (dm.weight == wt)].set_index("rep")
            common = b.index.intersection(a.index)
            if not len(common):
                continue
            dt = (a.loc[common, "tpr"] - b.loc[common, "tpr"]).dropna()
            df_ = (a.loc[common, "fdp"] - b.loc[common, "fdp"]).dropna()
            try:
                pw = wilcoxon(dt, alternative="greater").pvalue \
                    if (dt != 0).any() else 1.0
            except Exception:
                pw = np.nan
            rows.append(dict(method=meth, cell=cell, weight=wt,
                             base=base_arm, n=len(dt),
                             dTPR=round(dt.mean(), 4),
                             pos=int((dt > 0).sum()),
                             neg=int((dt < 0).sum()),
                             p_wilcoxon=round(pw, 4),
                             dFDP=round(df_.mean(), 4)))
t = pd.DataFrame(rows)
print(t.to_string(index=False))
t.to_csv("/home/claude/ch_smoke/wrap_gain_table.csv", index=False)

# ---- pooled across cells per method (Wdet) ----
print("\n=== pooled per method (Wdet, all cells paired) ===")
for meth in d.method.unique():
    dm = d[d.method == meth]
    base_arm = {"zinq": "A0", "deseq2": "A0", "locom": "A1n",
                "ldm": "A1n"}.get(meth, "A1")
    b = dm[(dm.arm == base_arm) & (dm.weight == "none")]
    a = dm[(dm.arm == "A2") & (dm.weight == "Wdet")]
    m = b.merge(a, on=["cell", "rep"], suffixes=("_b", "_a"))
    dt = (m.tpr_a - m.tpr_b).dropna()
    if not len(dt):
        continue
    try:
        pw = wilcoxon(dt, alternative="greater").pvalue if (dt != 0).any() else 1.0
    except Exception:
        pw = np.nan
    print(f"{meth:<11} n={len(dt):3d}  dTPR={dt.mean():+.4f} "
          f"({(dt>0).sum()}+/{(dt<0).sum()}-)  p={pw:.2e}  "
          f"FDP(A2)={m.fdp_a.mean():.4f}")

# ---- L5 validity sweep: every A1/A2 arm ----
print("\n=== L5 validity: arms with FDP > 0.05 + 2*sem ===")
g = d[d.arm.isin(["A1", "A1n", "A2"])].groupby(
    ["cell", "method", "arm", "weight"])["fdp"].agg(["mean", "sem", "count"])
bad = g[g["mean"] > 0.05 + 2 * g["sem"].fillna(0)]
print(bad.round(4).to_string() if len(bad) else "none")

# ---- L1 raw calibration ----
print("\n=== L1 raw-arm miscalibration (A0 FDP > 0.10) ===")
g0 = d[d.arm == "A0"].groupby(["cell", "method"])[["fdp", "typeI"]].mean()
print(g0[g0.fdp > 0.10].round(4).to_string() if len(g0[g0.fdp > 0.10]) else
      "none")
print("\n(all raw arms)");  print(g0.round(4).to_string())
