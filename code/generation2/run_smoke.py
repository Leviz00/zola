"""run_smoke.py — dev-level smoke of the two-channel test on regenerated
confirmatory cells (SPEC deterministic-regeneration recipe).

Cells: 1000 (three_layer, phi-window), 1005 (zinb realfrac), 1008 (bridge,
hard), 1009 (bridge, informative absence). Reps 0..R-1, K permutations.

Truth conventions (dual-channel):
  intensity truth  A_j = abs_da_truth (10 taxa/rep)
  presence truth   P_j = designated_structural taxa IF informative cell
                          (group-aligned absence), ELSE empty
  union truth      U_j = A_j | P_j   (the combined test's estimand truth)

Data regeneration exactly per SPEC section 0:
  seeds = SeedSequence(config.seed).spawn(20)[rep]
  prm   = design.params_for_cell(row)   (effect_mode=absolute from config)
  Y,tr  = generators.generate(row.mechanism, prm, n, p=100, depths=row.depth,
                              seed=seeds[rep])
NOTE (dev disclosure): element-level equality with the original v34 npz files
cannot be verified here (originals not in the connected folders); the recipe,
seed rule and parameters match SPEC/config verbatim, and cell-level
signatures (1009: 30 taxa absent in all case samples, disjoint from abs_da)
are asserted below.
"""
from __future__ import annotations

import sys, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude/ch_smoke/code/simulation_v3")
sys.path.insert(0, "/home/claude/ch_smoke")
import design, generators                              # noqa: E402
from twochannel import two_channel_test, bh_reject     # noqa: E402

P_TAXA = 100
CELLS = [1000, 1005, 1008, 1009]
R = 5
K = 999
ALPHA = 0.05

cfg = pd.read_csv(
    "/home/claude/ch_smoke/code/simulation_v3/configs/config_supplementary.csv")
cfg = cfg.set_index("cell_id")

rows = []
taxon_rows = []
for cell in CELLS:
    row = cfg.loc[cell]
    prm = design.params_for_cell(row)
    n = int(row["n"])
    seeds = np.random.SeedSequence(int(row["seed"])).spawn(20)
    for rep in range(R):
        t0 = time.time()
        Y, tr = generators.generate(row["mechanism"], prm, n, P_TAXA,
                                    row["depth"], seed=seeds[rep])
        group = tr["group"]
        N = tr["depths"].astype(float)          # design-depth convention (est-arm)
        A = tr["abs_da_truth"].astype(bool)
        if bool(row["informative_zeros"]):
            desig = tr["designated_structural"]          # (2n, p)
            Pt = desig[group == 1].all(axis=0) & (desig.sum(axis=0) > 0)
        else:
            Pt = np.zeros(P_TAXA, dtype=bool)
        Ut = A | Pt

        # cell-signature asserts (1009)
        if cell == 1009:
            assert Pt.sum() == 30, f"1009 rep{rep}: expected 30 absent taxa, got {Pt.sum()}"
            assert not (A & Pt).any(), "abs_da and absent sets must be disjoint"
            assert (Y[np.ix_(group == 1, np.where(Pt)[0])] == 0).all()

        res = two_channel_test(Y, N, group, nu=N, K=K,
                               seed=20260818 + cell * 100 + rep)

        rej_comb = bh_reject(res["p_comb"], ALPHA)
        rej_det = bh_reject(res["p_det"], ALPHA)
        rej_int = bh_reject(res["p_int"], ALPHA)

        def fdp_tpr(rej, truth):
            nr = int(rej.sum())
            fp = int((rej & ~truth).sum())
            tp = int((rej & truth).sum())
            fdp = fp / nr if nr else 0.0
            tpr = tp / truth.sum() if truth.sum() else np.nan
            return nr, fdp, tpr

        nrC, fdpC, tprC = fdp_tpr(rej_comb, Ut)
        _, _, tprA = fdp_tpr(rej_comb, A)
        _, _, tprP = fdp_tpr(rej_comb, Pt) if Pt.any() else (0, 0.0, np.nan)
        nrD, fdpD, tprD = fdp_tpr(rej_det, Pt if Pt.any() else A)
        nrI, fdpI, tprI = fdp_tpr(rej_int, A)
        nulls = ~Ut
        t1 = (res["p_comb"][nulls] < 0.05).mean()

        rows.append(dict(cell=cell, rep=rep, n=2 * n,
                         phi_hat=res["phi_hat"],
                         n_rej_comb=nrC, fdp_union=fdpC, tpr_union=tprC,
                         tpr_A=tprA, tpr_presence=tprP,
                         n_rej_det=nrD, fdp_det=fdpD, tpr_det=tprD,
                         n_rej_int=nrI, fdp_int=fdpI, tpr_int=tprI,
                         typeI_null=t1,
                         det_tested=int(res["det_ok"].sum()),
                         int_tested=int(res["int_ok"].sum()),
                         secs=round(time.time() - t0, 1)))
        for j in np.where(rej_comb)[0]:
            taxon_rows.append(dict(cell=cell, rep=rep, taxon=j,
                                   p=res["p_comb"][j],
                                   channel=res["attribution"][j],
                                   is_A=bool(A[j]), is_P=bool(Pt[j])))
        print(rows[-1], flush=True)

df = pd.DataFrame(rows)
df.to_csv("/home/claude/ch_smoke/smoke_detail.csv", index=False)
pd.DataFrame(taxon_rows).to_csv("/home/claude/ch_smoke/smoke_rejections.csv",
                                index=False)

summ = df.groupby("cell").agg(
    fdp_union=("fdp_union", "mean"), tpr_union=("tpr_union", "mean"),
    tpr_A=("tpr_A", "mean"), tpr_presence=("tpr_presence", "mean"),
    fdp_det=("fdp_det", "mean"), tpr_det=("tpr_det", "mean"),
    fdp_int=("fdp_int", "mean"), tpr_int=("tpr_int", "mean"),
    typeI=("typeI_null", "mean"), rej=("n_rej_comb", "mean"),
    phi_hat=("phi_hat", "median"), secs=("secs", "mean")).round(3)
summ.to_csv("/home/claude/ch_smoke/smoke_summary.csv")
print("\n=== SUMMARY (cell means over reps) ===")
print(summ.to_string())
