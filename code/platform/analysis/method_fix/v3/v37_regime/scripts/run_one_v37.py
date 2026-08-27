"""run_one_v37.py — A1 复测：v3.4 网格每 (cell, rep) 的置换校准检验。

对 v34_full 每个 npz：同种子重放 Y/N/group（与 v3.4/v3.5 逐位一致），
重建 Ŵ，est / placeholder 两臂各跑 calibrated_test（K=20 混合池化 +
LKO FWER），记录校准后 FDP/TPR/FWER、重零尾否决数、池化诊断、耗时。
真值仅用于事后评估。用法: --npz <v34 npz> --out <row csv>
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

SIM_V3 = "/mnt/agents/output/code/simulation_v3"
V36 = "/mnt/agents/output/analysis/method_fix/v3/v36_calibration"
CONFIG_CSV = os.path.join(SIM_V3, "configs", "config_supplementary.csv")
R_REPS = 20
P_TAXA = 100

sys.path.insert(0, V36)
sys.path.insert(1, SIM_V3)
import design  # noqa: E402
import generators  # noqa: E402
from perm_glm import calibrated_test, pool_diagnostic  # noqa: E402


def fdp_tpr(rej, da):
    fp = int((rej & ~da).sum())
    tp = int((rej & da).sum())
    return (fp / (fp + tp) if fp + tp > 0 else 0.0,
            tp / max(int(da.sum()), 1), int(rej.sum()), fp, tp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    cell = int(z["cell_id"]); rep = int(z["rep"])
    cfg = pd.read_csv(CONFIG_CSV)
    row = cfg[cfg.cell_id == cell].iloc[0]
    seeds = np.random.SeedSequence(int(row["seed"])).spawn(R_REPS)
    prm = design.params_for_cell(row)
    prm["effect_mode"] = "absolute"
    Y, truth = generators.generate(row["mechanism"], prm, n=int(row["n"]),
                                   p=P_TAXA, depths=int(row["depth"]),
                                   seed=seeds[rep])
    Y = np.asarray(Y, dtype=float)
    group = np.asarray(truth["group"])
    N = truth["depths"].astype(float)
    da = truth["abs_da_truth"]

    zero = Y == 0
    W = np.zeros(Y.shape)
    W[zero] = z["scores"]
    est_keep = ~((Y == 0) & (W >= 0.5))

    out = dict(file=os.path.basename(args.npz), cell_id=cell, rep=rep,
               mechanism=str(row["mechanism"]),
               grid_group=str(row["grid_group"]))
    sz = truth["structural_zeros"]
    oracle_keep = ~sz
    rng = np.random.default_rng(20260305)
    perms = [rng.permutation(group) for _ in range(20)]  # 三臂共享同一组置换
    for arm, mask in (("est", est_keep.astype(float)), ("plac", None),
                      ("oracle", oracle_keep.astype(float))):
        t0 = time.time()
        r = calibrated_test(Y, group, N, W=mask, perms=perms)
        fdp, tpr, nrej, fp, tp = fdp_tpr(r["reject"], da)
        out.update(**{f"{arm}_fdp": fdp, f"{arm}_tpr": tpr,
                      f"{arm}_n_rej": nrej, f"{arm}_n_fp": fp,
                      f"{arm}_n_tp": tp, f"{arm}_fwer": r["fwer"],
                      f"{arm}_n_heavy": int(r["heavy"].sum()),
                      f"{arm}_t": time.time() - t0})
        if arm == "est":
            mu_hat = Y.sum(0) / np.maximum(N.sum(), 1)
            d = pool_diagnostic(r["null"], mu_hat, r["alpha_hat"])
            out.update(**{f"diag_{k}": v for k, v in d.items()
                          if isinstance(v, (int, float, bool))})
    pd.DataFrame([out]).to_csv(args.out, index=False)
    print(f"saved {args.out} cell={cell} rep={rep} "
          f"est FDP={out['est_fdp']:.2f} TPR={out['est_tpr']:.2f} "
          f"FWER={out['est_fwer']:.2f} | plac FDP={out['plac_fdp']:.2f} "
          f"TPR={out['plac_tpr']:.2f} FWER={out['plac_fwer']:.2f}")


if __name__ == "__main__":
    main()
