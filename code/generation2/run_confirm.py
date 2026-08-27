"""run_confirm.py — SPEC-CH v1.0 confirmatory rerun.

8 cells x R=20, K=999, dual combination (ACAT primary / chi2 secondary),
dual-channel truth scoring. All conventions frozen in SPEC_CH.md BEFORE
this run; do not edit mid-run.
"""
from __future__ import annotations

import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # avoid BLAS thread pools in workers

import sys, time
import numpy as np
import pandas as pd
import multiprocessing as mp

sys.path.insert(0, "/home/claude/ch_smoke/code/simulation_v3")
sys.path.insert(0, "/home/claude/ch_smoke")
import design, generators                               # noqa: E402
from twochannel import two_channel_test, bh_reject      # noqa: E402

P_TAXA = 100
CELLS = [1000, 1002, 1004, 1005, 1006, 1007, 1008, 1009]
R = 20
K = 999
ALPHA = 0.05
SMOKE_SEEN = {(c, r) for c in (1000, 1005, 1008, 1009) for r in range(5)}

CFG = pd.read_csv(
    "/home/claude/ch_smoke/code/simulation_v3/configs/config_supplementary.csv"
).set_index("cell_id")


def fdp_tpr(rej, truth):
    nr = int(rej.sum())
    fp = int((rej & ~truth).sum())
    tp = int((rej & truth).sum())
    fdp = fp / nr if nr else 0.0
    tpr = tp / truth.sum() if truth.sum() else np.nan
    return nr, fdp, tpr


def one_run(args):
    cell, rep = args
    row = CFG.loc[cell]
    prm = design.params_for_cell(row)
    n = int(row["n"])
    seeds = np.random.SeedSequence(int(row["seed"])).spawn(20)
    t0 = time.time()
    Y, tr = generators.generate(row["mechanism"], prm, n, P_TAXA,
                                row["depth"], seed=seeds[rep])
    group = tr["group"]
    N = tr["depths"].astype(float)
    A = tr["abs_da_truth"].astype(bool)
    if bool(row["informative_zeros"]):
        desig = tr["designated_structural"]
        Pt = desig[group == 1].all(axis=0) & (desig.sum(axis=0) > 0)
        # SPEC-CH D1 cell-signature assertions
        assert not (A & Pt).any()
        assert (Y[np.ix_(group == 1, np.where(Pt)[0])] == 0).all()
        if cell == 1009:
            assert Pt.sum() == 30
    else:
        Pt = np.zeros(P_TAXA, dtype=bool)
    Ut = A | Pt

    res = two_channel_test(Y, N, group, nu=N, K=K, seed=[20260818, cell, rep])

    out = dict(cell=cell, rep=rep, n=2 * n, phi_hat=res["phi_hat"],
               smoke_seen=(cell, rep) in SMOKE_SEEN,
               det_tested=int(res["det_ok"].sum()),
               int_tested=int(res["int_ok"].sum()))
    rejA = {}
    for tag, pv in (("acat", res["p_comb"]), ("chisq", res["p_comb_chisq"])):
        rej = bh_reject(pv, ALPHA)
        rejA[tag] = rej
        nr, fdp, tpr = fdp_tpr(rej, Ut)
        _, _, tprA = fdp_tpr(rej, A)
        _, _, tprP = fdp_tpr(rej, Pt) if Pt.any() else (0, 0.0, np.nan)
        out.update({f"nrej_{tag}": nr, f"fdp_{tag}": fdp, f"tpr_{tag}": tpr,
                    f"tprA_{tag}": tprA, f"tprP_{tag}": tprP})
    rej_det = bh_reject(res["p_det"], ALPHA)
    rej_int = bh_reject(res["p_int"], ALPHA)
    nrD, fdpD, tprD = fdp_tpr(rej_det, Pt if Pt.any() else A)
    nrI, fdpI, tprI = fdp_tpr(rej_int, A)
    nulls = ~Ut
    out.update(nrej_det=nrD, fdp_det=fdpD, tpr_det=tprD,
               nrej_int=nrI, fdp_int=fdpI, tpr_int=tprI,
               typeI_null=float((res["p_comb"][nulls] < 0.05).mean()),
               secs=round(time.time() - t0, 1))

    rej_rows = []
    for j in np.where(rejA["acat"])[0]:
        rej_rows.append(dict(cell=cell, rep=rep, taxon=int(j),
                             p=float(res["p_comb"][j]),
                             channel=str(res["attribution"][j]),
                             is_A=bool(A[j]), is_P=bool(Pt[j])))
    return out, rej_rows


if __name__ == "__main__":
    jobs = [(c, r) for c in CELLS for r in range(R)]
    t0 = time.time()
    ctx = mp.get_context("spawn")     # avoid fork-poisoned BLAS spin locks
    with ctx.Pool(2) as pool:
        results = []
        for i, res in enumerate(pool.imap_unordered(one_run, jobs)):
            results.append(res)
            if (i + 1) % 10 == 0:
                print(f"[{i+1}/160] {time.time()-t0:.0f}s", flush=True)
    rows = [r for r, _ in results]
    rejs = [x for _, rr in results for x in rr]
    df = pd.DataFrame(rows).sort_values(["cell", "rep"])
    df.to_csv("/home/claude/ch_smoke/confirm_detail.csv", index=False)
    pd.DataFrame(rejs).to_csv("/home/claude/ch_smoke/confirm_rejections.csv",
                              index=False)

    def summarize(d, label):
        g = d.groupby("cell")
        s = pd.DataFrame({
            "fdp_acat": g["fdp_acat"].mean(),
            "fdp_mcse": g["fdp_acat"].sem(),
            "tpr_acat": g["tpr_acat"].mean(),
            "tpr_mcse": g["tpr_acat"].sem(),
            "tprA_acat": g["tprA_acat"].mean(),
            "tprP_acat": g["tprP_acat"].mean(),
            "fdp_chisq": g["fdp_chisq"].mean(),
            "tpr_chisq": g["tpr_chisq"].mean(),
            "fdp_det": g["fdp_det"].mean(), "tpr_det": g["tpr_det"].mean(),
            "fdp_int": g["fdp_int"].mean(), "tpr_int": g["tpr_int"].mean(),
            "typeI": g["typeI_null"].mean(), "nrej": g["nrej_acat"].mean(),
            "n_runs": g["rep"].count(),
        }).round(4)
        s["subset"] = label
        return s

    full = summarize(df, "all20")
    untouched = summarize(df[~df["smoke_seen"]], "untouched")
    summ = pd.concat([full, untouched])
    summ.to_csv("/home/claude/ch_smoke/confirm_summary.csv")
    glob = dict(
        global_fdp_acat=df["fdp_acat"].mean(),
        global_fdp_mcse=df["fdp_acat"].sem(),
        global_typeI=df["typeI_null"].mean(),
        total_secs=round(time.time() - t0, 0))
    print(pd.Series(glob).to_string())
    print(full.to_string())
    print("DONE", flush=True)
