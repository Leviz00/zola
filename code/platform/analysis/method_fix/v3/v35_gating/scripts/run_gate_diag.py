"""run_gate_diag.py — 对 v34_full 的每个 npz 计算门控诊断（data-only）。

每个 (cell, rep)：按种子协议重新生成 Y/N/group（同种子 ⇒ 同数据），
从 npz 读拟合输出（φ̂、veto、Ŵ 逐零分），计算 gate.py 三组件诊断。
真值（abs_da_truth）仅用于副臂评估（est@α=0.01 的 FDP/TPR），
**不进入任何门控特征**。

用法: python3 run_gate_diag.py --npz <v34 npz> --out <row csv>
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

SIM_V3 = "/mnt/agents/output/code/simulation_v3"
GATE_DIR = "/mnt/agents/output/analysis/method_fix/v3/v35_gating"
CONFIG_CSV = os.path.join(SIM_V3, "configs", "config_supplementary.csv")
R_REPS = 20
P_TAXA = 100

sys.path.insert(0, GATE_DIR)
sys.path.insert(1, SIM_V3)
import design  # noqa: E402
import generators  # noqa: E402
from abs_glm import abs_nb_glm  # noqa: E402
import gate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    cell = int(z["cell_id"])
    rep = int(z["rep"])
    phi_hat = float(z["phi_hat"])
    veto = bool(z["cnt_veto"])
    est_n_fb = float(z["estimated_n_fallback"])
    plac_n_fb = float(z["placeholder_n_fallback"])

    cfg = pd.read_csv(CONFIG_CSV)
    row = cfg[cfg.cell_id == cell].iloc[0]
    seeds = np.random.SeedSequence(int(row["seed"])).spawn(R_REPS)
    prm = design.params_for_cell(row)
    prm["effect_mode"] = "absolute"
    Y, truth = generators.generate(
        row["mechanism"], prm, n=int(row["n"]), p=P_TAXA,
        depths=int(row["depth"]), seed=seeds[rep])
    Y = np.asarray(Y, dtype=float)
    group = np.asarray(truth["group"])
    N = truth["depths"].astype(float)

    # 重建全量 Ŵ（npz 只存零细胞；Y>0 处按约定为 0）
    zero = Y == 0
    W = np.zeros(Y.shape)
    sco = z["scores"]
    assert len(sco) == int(zero.sum()), "npz scores 与再生零掩码不一致"
    W[zero] = sco

    diag = gate.basic_diagnostics(
        Y, N, group, phi_hat, veto, W,
        est_n_tested_frac=(P_TAXA - est_n_fb) / P_TAXA)
    diag.update(gate.leakage_precheck(W, group, N))
    diag.update(gate.permutation_calibration(Y, N, group, W, abs_nb_glm,
                                             K=20, alpha=0.05))
    diag["plac_tested_frac"] = (P_TAXA - plac_n_fb) / P_TAXA

    # ---- 副臂（评估用，truth 允许）：est 臂 α=0.01 保守变体的 FDP/TPR ------
    da = truth["abs_da_truth"]
    keep = ~((Y == 0) & (W >= 0.5))
    r01 = abs_nb_glm(Y, group, N=N, W=keep.astype(float), alpha=0.01)
    rej = r01["reject"]
    fp = int((rej & ~da).sum())
    tp = int((rej & da).sum())
    diag["est01_fdp"] = fp / (fp + tp) if (fp + tp) > 0 else 0.0
    diag["est01_tpr"] = tp / max(int(da.sum()), 1)
    diag["est01_n_rej"] = int(rej.sum())

    diag.update(file=os.path.basename(args.npz), cell_id=cell, rep=rep,
                mechanism=str(row["mechanism"]),
                grid_group=str(row["grid_group"]))
    pd.DataFrame([diag]).to_csv(args.out, index=False)
    print(f"saved {args.out} cell={cell} rep={rep} "
          f"perm_est={diag['perm_rej_rate_est']:.3f} "
          f"perm_plac={diag['perm_rej_rate_plac']:.3f} "
          f"tested={diag['est_tested_frac']:.2f} "
          f"phi={phi_hat:.3g} veto={veto}")


if __name__ == "__main__":
    main()
