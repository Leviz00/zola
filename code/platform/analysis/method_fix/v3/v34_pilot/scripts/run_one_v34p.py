"""run_one_v34p.py — v3.4-PILOT：单 (cell, rep) 端到端流水线。

预注册规格：
  simulation_v3 absolute 生成 -> estimation_v3 v3.1 验收配置拟合
  （计数块 λ=0.44 + 撞界否决回退 λ=1.0；相干后验 phi_known=φ̂_cnt；
  W0 截距-only 权重）-> abs_nb_glm 三臂：
    oracle      : W = 真实结构零（presence 掩码）
    estimated   : W = v3.1 后验 Ŵ，剔除规则 Y==0 & Ŵ>=0.5（与 v3.1 一致）
    placeholder : 不剔除（全零当抽样零）
种子：config seed -> SeedSequence.spawn(R) -> rep 子种子（各臂同 rep 同种子）。

逐 cell-rep 记录：三臂 FDP/TPR/n_rej/fallback、Ŵ 质量（ROC-AUC/Brier vs
真实结构零）、分阶段 wall time、φ̂_cnt/veto。npz 存 Ŵ 逐零分供复核。
用法: python3 run_one_v34p.py --cell 1002 --rep 0 --out out.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit

SIM_V3 = "/mnt/agents/output/code/simulation_v3"
EST_V3 = "/mnt/agents/output/code/estimation_v3"
CONFIG_CSV = os.path.join(SIM_V3, "configs", "config_supplementary.csv")

P_TAXA = 100
COUNT_BLOCK_B = 4
DET_MAXITER = 500
COUNT_MAXITER = 200
COUNT_GTOL = 1e-6
COUNT_FTOL = 1e-10
LOG_PHI_BOUNDS = (np.log(0.05), np.log(1e5))
PRIOR_ETA0 = float(np.log(50.0))
LAM_MAIN = 0.44
LAM_RETRY = 1.0
EXCL_THRESHOLD = 0.5
R_REPS = 5

sys.path.insert(0, EST_V3)
sys.path.insert(1, SIM_V3)
import design  # noqa: E402
import generators  # noqa: E402
import composite_likelihood as est_cl  # noqa: E402
import composite_likelihood_cov as est_clcov  # noqa: E402
import composite_likelihood_count as est_clc  # noqa: E402
import posterior as est_post  # noqa: E402
from abs_glm import abs_nb_glm  # noqa: E402
from weighting import validate_weights  # noqa: E402


def count_refine(Y, N, f_det, lam, p):
    blocks = est_clc.make_blocks(p, COUNT_BLOCK_B)
    psi0 = np.concatenate([
        [np.log(f_det["phi"])],
        logit(np.clip(f_det["pi"], 1e-4, 0.9999)),
        np.log(np.clip(f_det["theta"], 1e-7, 0.9))])
    bounds = ([(np.log(0.05), np.log(1e5))]
              + [(logit(1e-4), logit(0.9999))] * p
              + [(np.log(1e-7), np.log(0.9))] * p)

    def obj(psi):
        ll, g = est_clc.count_loglik_grad(psi, Y, N, blocks, None,
                                          PRIOR_ETA0, lam)
        ng = float(np.abs(g).max())
        if ng > 1e3:
            g = g * (1e3 / ng)
        return -ll, -g

    res = minimize(obj, psi0, method="L-BFGS-B", jac=True, bounds=bounds,
                   options={"maxiter": COUNT_MAXITER, "ftol": COUNT_FTOL,
                            "gtol": COUNT_GTOL})
    on_bnd = bool(np.isclose(res.x[0], LOG_PHI_BOUNDS[0], atol=1e-6)
                  or np.isclose(res.x[0], LOG_PHI_BOUNDS[1], atol=1e-6))
    return {"phi": float(np.exp(res.x[0])), "theta": np.exp(res.x[1 + p:]),
            "on_boundary": on_bnd, "lam": lam, "nit": int(res.nit)}


def auc_mw(labels, scores):
    labels = np.asarray(labels, dtype=float)
    order = np.argsort(scores, kind="mergesort")
    sr = scores[order]
    ranks = np.empty(scores.shape[0])
    i = 0
    while i < scores.shape[0]:
        j = i
        while j + 1 < scores.shape[0] and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int, required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = pd.read_csv(CONFIG_CSV)
    row = cfg[cfg.cell_id == args.cell].iloc[0]
    seeds = np.random.SeedSequence(int(row["seed"])).spawn(R_REPS)
    prm = design.params_for_cell(row)
    prm["effect_mode"] = "absolute"

    # ---- 阶段 1：生成 ------------------------------------------------------
    t0 = time.time()
    Y, truth = generators.generate(
        row["mechanism"], prm, n=int(row["n"]), p=P_TAXA,
        depths=int(row["depth"]), seed=seeds[args.rep])
    t_gen = time.time() - t0

    group = np.asarray(truth["group"])
    da = truth["abs_da_truth"]
    sz_true = truth["structural_zeros"]
    Y = np.asarray(Y, dtype=float)
    D = (Y > 0).astype(float)
    N = truth["depths"].astype(float)  # 设计深度（GLM offset 锚）
    n, p = Y.shape

    # ---- 阶段 2：估计器拟合（v3.1 验收配置：相干 + W0） --------------------
    t0 = time.time()
    f_det = est_cl.fit_composite(D, N, phi_known=None, multi_start=False,
                                 maxiter=DET_MAXITER)
    f_cnt = count_refine(Y, N, f_det, LAM_MAIN, p)
    veto = False
    if f_cnt["on_boundary"]:
        f2 = count_refine(Y, N, f_det, LAM_RETRY, p)
        veto = True
        if not f2["on_boundary"]:
            f_cnt = f2
    phi_hat, theta_hat = f_cnt["phi"], f_cnt["theta"]
    Wd0 = np.ones((n, 1))
    f_w0 = est_clcov.fit_composite_cov(D, Wd0, N, phi_known=phi_hat)
    P_w0 = est_post.zero_source_posterior_cov(f_w0["Gamma"], Wd0,
                                              theta_hat, phi_hat, N)
    W_hat = validate_weights(P_w0, Y)
    t_fit = time.time() - t0

    # ---- 阶段 3：abs_nb_glm 三臂 -------------------------------------------
    arms = {
        "oracle": validate_weights(sz_true.astype(float), Y),
        "estimated": W_hat,
        "placeholder": None,
    }
    res = {}
    t_glm = 0.0
    for arm, W in arms.items():
        t0 = time.time()
        if arm == "placeholder":
            r = abs_nb_glm(Y, group, N=N, W=None)
        else:
            keep = ~((Y == 0) & (W >= EXCL_THRESHOLD))
            r = abs_nb_glm(Y, group, N=N, W=keep.astype(float))
        t_glm += time.time() - t0
        rej = r["reject"]
        fp = int((rej & ~da).sum())
        tp = int((rej & da).sum())
        res[arm] = dict(
            fdp=(fp / (fp + tp)) if (fp + tp) > 0 else 0.0,
            tpr=tp / max(int(da.sum()), 1),
            n_rej=int(rej.sum()), n_fp=fp, n_tp=tp,
            n_fallback=int(r["n_fallback"]))

    # ---- Ŵ 质量（逐零，vs 真实结构零） ------------------------------------
    zero = Y == 0
    lab = sz_true[zero].astype(int)
    sco = W_hat[zero]
    w_auc = auc_mw(lab, sco)
    w_brier = float(np.mean((sco - lab) ** 2))

    np.savez_compressed(
        args.out,
        scores=sco, labels=lab, depth=N[np.where(zero)[0]],
        group=group[np.where(zero)[0]],
        cell_id=args.cell, rep=args.rep, mechanism=str(row["mechanism"]),
        grid_group=str(row["grid_group"]), n=int(row["n"]),
        depth_cfg=int(row["depth"]), phi_true=float(row["dispersion_value"]),
        phi_hat=phi_hat, cnt_veto=veto, cnt_lam=f_cnt["lam"],
        t_gen=t_gen, t_fit=t_fit, t_glm=t_glm,
        w_auc=w_auc, w_brier=w_brier,
        n_da=int(da.sum()), struct_frac=float(lab.mean()),
        **{f"{arm}_{k}": v for arm, d in res.items() for k, v in d.items()})
    print(f"saved {args.out} cell={args.cell} rep={args.rep} "
          f"phi_hat={phi_hat:.3g} veto={veto} "
          f"fdp(o/e/p)={res['oracle']['fdp']:.2f}/{res['estimated']['fdp']:.2f}"
          f"/{res['placeholder']['fdp']:.2f} "
          f"tpr(o/e/p)={res['oracle']['tpr']:.2f}/{res['estimated']['tpr']:.2f}"
          f"/{res['placeholder']['tpr']:.2f} "
          f"t={t_gen:.0f}/{t_fit:.0f}/{t_glm:.0f}s")


if __name__ == "__main__":
    main()
