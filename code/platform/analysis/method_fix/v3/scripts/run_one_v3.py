"""run_one_v3.py — v3.1 估计器连贯性修复包：单 (cell, rep) 运行。

R2【后验跨拟合混装】修复：计数块精炼得 φ̂_cnt 后，用 phi_known=φ̂_cnt
重估协变量层（逐类群分解），最终后验出自同一个 φ 的连贯模型。
混装对照直接复用 v2 的 npz（不在此重跑，见 V31_MEMO）。

R3【组条件权重泄漏】三档权重臂：
  W0 截距-only（Wd=[1]）；
  W1 group 条件（Wd=[1,group]，现状对照）；
  W2 留一组出交叉拟合（group0 子样本拟合→预测 group1，反之亦然）。
三臂共用同一次计数块精炼（计数块不含 group，天然 group-free）。
φ 协议：全局 λ=0.44 + 撞界否决（φ̂ 撞界则标记并回退 λ=1.0 重试一次）。

DA 结局（信号筛查，R=3）：exclusion_wilcoxon 的逐 rep FDP/TPR/n_rej。

种子协议与 extract_zero_posteriors.py 一致：config seed → spawn(20) → rep。
用法: python3 run_one_v3.py --cell 6 --rep 0 --out out.npz
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

SIM_DIR = "/mnt/agents/output/code/simulation"
EST_V3 = "/mnt/agents/output/code/estimation_v3"
CONFIG_CSV = os.path.join(SIM_DIR, "configs", "config_fractional.csv")

P_TAXA = 100
COUNT_BLOCK_B = 4
DET_MAXITER = 500
COUNT_MAXITER = 200
COUNT_GTOL = 1e-6
COUNT_FTOL = 1e-10
R_SPAWN = 20
LOG_PHI_BOUNDS = (np.log(0.05), np.log(1e5))
PRIOR_ETA0 = float(np.log(50.0))
LAM_MAIN = 0.44
LAM_RETRY = 1.0


def count_refine(est_clc, Y, N, f_det, lam, p):
    """计数块精炼（_count_refine 等价物 + log-φ 惩罚）。"""
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
    success = bool(res.success)
    if not success and np.max(np.abs(res.jac)) < 1e-3:
        success = True
    on_bnd = bool(np.isclose(res.x[0], LOG_PHI_BOUNDS[0], atol=1e-6)
                  or np.isclose(res.x[0], LOG_PHI_BOUNDS[1], atol=1e-6))
    return {"phi": float(np.exp(res.x[0])), "theta": np.exp(res.x[1 + p:]),
            "success": success, "nit": int(res.nit), "on_boundary": on_bnd,
            "lam": lam}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int, required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, EST_V3)
    sys.path.insert(1, SIM_DIR)
    import design  # noqa: E402
    import generators  # noqa: E402
    import metrics  # noqa: E402
    import composite_likelihood as est_cl  # noqa: E402
    import composite_likelihood_cov as est_clcov  # noqa: E402
    import composite_likelihood_count as est_clc  # noqa: E402
    import posterior as est_post  # noqa: E402
    from weighting import exclusion_wilcoxon, validate_weights  # noqa: E402

    cfg = pd.read_csv(CONFIG_CSV)
    row = cfg[cfg.cell_id == args.cell].iloc[0]
    child_seeds = np.random.SeedSequence(int(row["seed"])).spawn(R_SPAWN)
    params = design.params_for_cell(row)
    Y, truth = generators.generate(
        row["mechanism"], params, n=int(row["n"]), p=P_TAXA,
        depths=int(row["depth"]), seed=child_seeds[args.rep])
    group = np.asarray(truth["group"])
    da = truth["da_taxa"]

    Y = np.asarray(Y, dtype=float)
    D = (Y > 0).astype(float)
    N = Y.sum(axis=1).astype(float)
    n, p = Y.shape
    diag = {}

    # ---- 计数块（group-free；λ=0.44 + 撞界否决回退 λ=1.0） ---------------
    t0 = time.time()
    f_det = est_cl.fit_composite(D, N, phi_known=None, multi_start=False,
                                 maxiter=DET_MAXITER)
    diag["det_phi"] = float(f_det["phi"])
    f_cnt = count_refine(est_clc, Y, N, f_det, LAM_MAIN, p)
    diag["cnt_veto_retry"] = False
    if f_cnt["on_boundary"]:
        f_cnt2 = count_refine(est_clc, Y, N, f_det, LAM_RETRY, p)
        diag["cnt_veto_retry"] = True
        diag["cnt_phi_first"] = f_cnt["phi"]
        if not f_cnt2["on_boundary"]:
            f_cnt = f_cnt2
    diag["cnt_phi"] = f_cnt["phi"]
    diag["cnt_phi_on_boundary"] = f_cnt["on_boundary"]
    diag["cnt_lam"] = f_cnt["lam"]
    diag["cnt_nit"] = f_cnt["nit"]
    diag["cnt_success"] = f_cnt["success"]
    diag["t_count"] = time.time() - t0
    phi_hat, theta_hat = f_cnt["phi"], f_cnt["theta"]

    # ---- 协变量层三臂（全部 phi_known=φ̂_cnt 连贯重估） -------------------
    t0 = time.time()
    # W1: group 条件（现状对照）
    Wd1 = np.column_stack([np.ones(n), group.astype(float)])
    f_w1 = est_clcov.fit_composite_cov(D, Wd1, N, phi_known=phi_hat)
    P_w1 = est_post.zero_source_posterior_cov(f_w1["Gamma"], Wd1,
                                              theta_hat, phi_hat, N)
    diag["t_w1"] = time.time() - t0
    diag["w1_success"] = bool(f_w1["success"])
    diag["w1_abs_gamma1_mean"] = float(np.abs(f_w1["Gamma"][:, 1]).mean())

    t0 = time.time()
    # W0: 截距-only
    Wd0 = np.ones((n, 1))
    f_w0 = est_clcov.fit_composite_cov(D, Wd0, N, phi_known=phi_hat)
    P_w0 = est_post.zero_source_posterior_cov(f_w0["Gamma"], Wd0,
                                              theta_hat, phi_hat, N)
    diag["t_w0"] = time.time() - t0
    diag["w0_success"] = bool(f_w0["success"])

    t0 = time.time()
    # W2: 留一组出交叉拟合
    pi_cross = np.zeros((n, p))
    for g_fit, g_pred in ((0, 1), (1, 0)):
        m_fit = group == g_fit
        m_pred = group == g_pred
        f_sub = est_clcov.fit_composite_cov(
            D[m_fit], np.ones((int(m_fit.sum()), 1)), N[m_fit],
            phi_known=phi_hat)
        pi_cross[m_pred] = 1.0 / (1.0 + np.exp(-f_sub["Gamma"][:, 0]))[None, :]
        diag[f"w2_success_g{g_fit}"] = bool(f_sub["success"])
    P_w2 = est_post.zero_source_posterior(pi_cross, theta_hat, phi_hat, N)
    diag["t_w2"] = time.time() - t0

    # ---- DA 结局（exclusion_wilcoxon，信号筛查） --------------------------
    da_rows = {}
    for arm, P in (("w0", P_w0), ("w1", P_w1), ("w2", P_w2)):
        W = validate_weights(P, Y)
        rej = exclusion_wilcoxon(Y, group, W)["reject"]
        fdp_r, n_rej = metrics.fdp(rej, da)
        tpr_r, _ = metrics.tpr(rej, da)
        da_rows[f"{arm}_fdp"] = float(fdp_r)
        da_rows[f"{arm}_tpr"] = float(tpr_r)
        da_rows[f"{arm}_n_rej"] = int(n_rej)

    diag["t_est"] = diag["t_count"] + diag["t_w1"] + diag["t_w0"] + diag["t_w2"]

    zero = Y == 0
    labels = truth["structural_zeros"][zero].astype(int)
    ii, jj = np.where(zero)
    np.savez_compressed(
        args.out,
        scores_w0=P_w0[zero], scores_w1=P_w1[zero], scores_w2=P_w2[zero],
        labels=labels, depth=N[ii], group=group[ii], taxon=jj, sample=ii,
        cell_id=args.cell, rep=args.rep,
        mechanism=str(row["mechanism"]),
        informative=bool(row["informative_zeros"]),
        sz=float(row["structural_zero_rate"]),
        n=int(row["n"]), depth_cfg=int(row["depth"]),
        phi_true=float(row["dispersion_value"]),
        **{k: v for k, v in diag.items()},
        **da_rows)
    print(f"saved {args.out} zeros={int(zero.sum())} "
          f"struct={labels.mean():.3f} cnt_phi={phi_hat:.3g} "
          f"veto={diag['cnt_veto_retry']} "
          f"fdp(w0/w1/w2)={da_rows['w0_fdp']:.3f}/{da_rows['w1_fdp']:.3f}/"
          f"{da_rows['w2_fdp']:.3f} t_est={diag['t_est']:.0f}s")


if __name__ == "__main__":
    main()
