"""run_v38a.py — v3.8 Stage A：Ŵ 逐零分辨率诊断（仅诊断，不修复）。

A1：真参 Bayes 最优逐零分类器 P(缺席|Y=0; 真 π、真 θ̄/m0、真 φ、真 N)
    —— 逐 rep oracle AUC vs est-Ŵ AUC（v34 npz scores 原样复用）。
    π 精确来源：three_layer/beta_binomial truth["pi"]（掩码处已置 0）；
    zinb/zigdm π=1−truth["omega"]。m0/prev 经子流前缀重放精确复原
    （_spawn(seed,8)[1]/[2] 与生成器内部一致）。零概率 g=(κ/(κ+μ))^κ，
    κ 按机制：three_layer φ/(1−m0_j)，zinb/beta_binomial φ，zigdm c/(1−m0_j)。
A2：消融——π-only（g≡1）、depth-only（π≡0.5）、composition-only
    （π≡0.5、N≡N̄）、oracle-full、est-full 五路 AUC。
A3：|log10(φ̂/φ)| 与 AUC 缺口的关联；cell 1008 单独点名。
仅 0<share<1 格（1004-1008）；share=1.0 格标签无变异，跳过。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

SIM = "/mnt/agents/output/code/simulation_v3"
V34 = "/mnt/agents/output/analysis/method_fix/v3/v34_full"
OUT = "/mnt/agents/output/analysis/method_fix/v3/v38_resolution"
sys.path.insert(0, SIM)
import design  # noqa: E402
import generators  # noqa: E402

CELLS = [1004, 1005, 1006, 1007, 1008]
CFG = pd.read_csv(f"{SIM}/configs/config_supplementary.csv")


def mw_auc(score, label):
    """Mann-Whitney AUC with average ranks for ties."""
    from scipy.stats import rankdata
    r = rankdata(score)
    pos = label == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def replay_params(row, rep_seed, prm, p):
    """精确复原 m0 与 prev（子流前缀与生成器内部一致）。"""
    ss = generators._spawn(rep_seed, 8)
    prm2 = generators._resolve_params(prm)
    m0 = generators._base_composition(p, np.random.default_rng(ss[1]),
                                      prm2["base_sigma"])
    prev = generators._prevalences(p, prm2, np.random.default_rng(ss[2]))
    return m0, prev


def main():
    rows = []
    for cell in CELLS:
        row = CFG[CFG.cell_id == cell].iloc[0]
        mech = row["mechanism"]
        seeds = np.random.SeedSequence(int(row["seed"])).spawn(20)
        prm = design.params_for_cell(row)
        prm["effect_mode"] = "absolute"
        phi_true = float(row["dispersion_value"])
        for rep in range(20):
            z = np.load(f"{V34}/npz/cell{cell}_rep{rep}.npz")
            scores, labels = z["scores"], z["labels"]
            Y, truth = generators.generate(mech, prm, n=int(row["n"]), p=100,
                                           depths=int(row["depth"]),
                                           seed=seeds[rep])
            Y = np.asarray(Y, float)
            zero = (Y == 0)
            assert zero.sum() == len(scores), (cell, rep)
            N = truth["depths"].astype(float)
            group = truth["group"].astype(int)
            da = truth["abs_da_truth"]
            eff = np.where((group == 1)[:, None] & da[None, :],
                           float(row["effect_size"]), 1.0)
            m0, prev = replay_params(row, seeds[rep], prm, Y.shape[1])
            if mech in ("three_layer", "beta_binomial"):
                pi = truth["pi"]
            elif mech == "zinb":
                pi = 1.0 - truth["omega"]
            else:  # zigdm_like：π=prev_j（掩码处 0），truth 未存 omega
                pi = np.where(truth["designated_structural"], 0.0,
                              np.broadcast_to(prev[None, :], Y.shape))
            mu = N[:, None] * m0[None, :] * eff
            if mech in ("three_layer", "zigdm_like"):
                kappa = phi_true / np.maximum(1.0 - m0, 1e-9)[None, :]
            else:
                kappa = np.full((1, Y.shape[1]), phi_true)
            g = (kappa / (kappa + mu)) ** kappa  # (n,p)
            pi_z, g_z = pi[zero], g[zero]
            # oracle 全分类器
            P_full = (1 - pi_z) / np.maximum((1 - pi_z) + pi_z * g_z, 1e-300)
            # 消融
            P_pi = 1 - pi_z                                    # π-only
            P_dep = 1.0 / (1.0 + g_z)                          # depth-only
            gj = (kappa[0] / (kappa[0] + N.mean() * m0)) ** kappa[0]
            P_comp = 1.0 / (1.0 + np.broadcast_to(gj, Y.shape)[zero])
            rec = dict(
                cell_id=cell, rep=rep, mechanism=mech,
                share=float(z["struct_frac"]),
                phi_hat=float(z["phi_hat"]), phi_true=phi_true,
                log10_phi_ratio=abs(np.log10(float(z["phi_hat"]) / phi_true)),
                auc_est=mw_auc(scores, labels),
                auc_oracle=mw_auc(P_full, labels),
                auc_pi_only=mw_auc(P_pi, labels),
                auc_depth_only=mw_auc(P_dep, labels),
                auc_comp_only=mw_auc(P_comp, labels),
            )
            rec["gap"] = rec["auc_oracle"] - rec["auc_est"]
            rows.append(rec)
        print(cell, mech, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/v38a_replevel.csv", index=False)
    summ = df.groupby(["cell_id", "mechanism"]).agg(
        share=("share", "mean"),
        auc_est_mean=("auc_est", "mean"), auc_est_med=("auc_est", "median"),
        auc_oracle_mean=("auc_oracle", "mean"),
        auc_oracle_med=("auc_oracle", "median"),
        gap_mean=("gap", "mean"), gap_med=("gap", "median"),
        pi_only=("auc_pi_only", "mean"), depth_only=("auc_depth_only", "mean"),
        comp_only=("auc_comp_only", "mean"),
        phi_ratio_log10=("log10_phi_ratio", "mean")).reset_index()
    summ.to_csv(f"{OUT}/v38a_cell_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
