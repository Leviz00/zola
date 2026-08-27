"""run_one.py — method_fix 开发级评估：单 (cell, rep, arm, λ) 提取逐零后验。

种子协议与 code/simulation/extract_zero_posteriors.py 完全一致：
  config_fractional.csv 的 seed → SeedSequence.spawn(R_spawn=20) → rep 索引。

arm=v1：旧估计目录（budget 臂：cov multi_start=False，无惩罚）——与线上
        48 格运行同一协议，严格配对基线。
arm=v2：estimation_v2（log-φ 弱信息先验 + cov multi_start=True 恢复）。

用法:
  python3 run_one.py --cell 6 --rep 0 --arm v2 --lam 0.44 --out out.npz
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
EST_V1 = "/mnt/agents/output/code/estimation"
EST_V2 = "/mnt/agents/output/code/estimation_v2"
CONFIG_CSV = os.path.join(SIM_DIR, "configs", "config_fractional.csv")

P_TAXA = 100
COUNT_BLOCK_B = 4
COV_MAXITER = 500
DET_MAXITER = 500
COUNT_MAXITER = 200
COUNT_GTOL = 1e-6
COUNT_FTOL = 1e-10
R_SPAWN = 20
LOG_PHI_BOUNDS = (np.log(0.05), np.log(1e5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int, required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--arm", choices=["v1", "v2"], required=True)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--eta0", type=float, default=float(np.log(50.0)))
    ap.add_argument("--out", required=True)
    ap.add_argument("--cov-multistart", type=int, default=-1,
                    help="覆盖 cov 多起点（默认 v1=0, v2=1）")
    args = ap.parse_args()

    est_dir = EST_V1 if args.arm == "v1" else EST_V2
    lam = args.lam if args.arm == "v2" else 0.0
    cov_ms = (args.arm == "v2") if args.cov_multistart < 0 else bool(
        args.cov_multistart)

    sys.path.insert(0, est_dir)
    sys.path.insert(1, SIM_DIR)
    import design  # noqa: E402
    import generators  # noqa: E402
    import composite_likelihood as est_cl  # noqa: E402
    import composite_likelihood_cov as est_clcov  # noqa: E402
    import composite_likelihood_count as est_clc  # noqa: E402
    import posterior as est_post  # noqa: E402

    cfg = pd.read_csv(CONFIG_CSV)
    row = cfg[cfg.cell_id == args.cell].iloc[0]
    child_seeds = np.random.SeedSequence(int(row["seed"])).spawn(R_SPAWN)
    params = design.params_for_cell(row)

    Y, truth = generators.generate(
        row["mechanism"], params, n=int(row["n"]), p=P_TAXA,
        depths=int(row["depth"]), seed=child_seeds[args.rep])
    group = truth["group"]

    Y = np.asarray(Y, dtype=float)
    D = (Y > 0).astype(float)
    N = Y.sum(axis=1).astype(float)
    Wd = np.column_stack([np.ones(group.shape[0]), group.astype(float)])
    p = P_TAXA
    diag = {}

    # (a) 协变量层检出指示 CL
    t0 = time.time()
    kw = dict(phi_known=None, multi_start=cov_ms, maxiter=COV_MAXITER)
    if args.arm == "v2":
        kw.update(prior_eta0=args.eta0, prior_lam=lam)
    f_cov = est_clcov.fit_composite_cov(D, Wd, N, **kw)
    diag["t_cov"] = time.time() - t0
    diag["cov_phi"] = float(f_cov["phi"])
    diag["cov_success"] = bool(f_cov["success"])
    diag["cov_nit"] = int(f_cov["n_iter"])
    diag["cov_msg"] = str(f_cov["message"])[:80]
    if args.arm == "v2":
        diag["cov_converged"] = bool(f_cov["converged"])
        diag["cov_phi_on_boundary"] = bool(f_cov["phi_on_boundary"])
    else:
        msg = str(f_cov["message"]).upper()
        diag["cov_converged"] = bool(f_cov["success"]
                                     and "ITERATIONS" not in msg
                                     and int(f_cov["n_iter"]) < COV_MAXITER)
        diag["cov_phi_on_boundary"] = bool(f_cov["on_boundary"][0])

    # (b1) 无协变量检出 CL 暖启动
    t0 = time.time()
    f_det = est_cl.fit_composite(D, N, phi_known=None, multi_start=False,
                                 maxiter=DET_MAXITER)

    # (b2) 计数块精炼（_count_refine 等价物，v2 加惩罚）
    blocks = est_clc.make_blocks(p, COUNT_BLOCK_B)
    psi0 = np.concatenate([
        [np.log(f_det["phi"])],
        logit(np.clip(f_det["pi"], 1e-4, 0.9999)),
        np.log(np.clip(f_det["theta"], 1e-7, 0.9))])
    bounds = ([(np.log(0.05), np.log(1e5))]
              + [(logit(1e-4), logit(0.9999))] * p
              + [(np.log(1e-7), np.log(0.9))] * p)

    def obj(psi):
        if args.arm == "v2":
            ll, g = est_clc.count_loglik_grad(psi, Y, N, blocks, None,
                                              args.eta0, lam)
        else:
            ll, g = est_clc.count_loglik_grad(psi, Y, N, blocks, None)
        ng = float(np.abs(g).max())
        if ng > 1e3:
            g = g * (1e3 / ng)
        return -ll, -g

    res = minimize(obj, psi0, method="L-BFGS-B", jac=True, bounds=bounds,
                   options={"maxiter": COUNT_MAXITER, "ftol": COUNT_FTOL,
                            "gtol": COUNT_GTOL})
    cnt_success = bool(res.success)
    if not cnt_success and np.max(np.abs(res.jac)) < 1e-3:
        cnt_success = True
    msg = str(res.message).upper()
    diag["t_count"] = time.time() - t0
    diag["cnt_phi"] = float(np.exp(res.x[0]))
    diag["cnt_success"] = cnt_success
    diag["cnt_nit"] = int(res.nit)
    diag["cnt_msg"] = str(res.message)[:80]
    diag["cnt_converged"] = bool(cnt_success and "ITERATIONS" not in msg
                                 and int(res.nit) < COUNT_MAXITER)
    diag["cnt_phi_on_boundary"] = bool(
        np.isclose(res.x[0], LOG_PHI_BOUNDS[0], atol=1e-6)
        or np.isclose(res.x[0], LOG_PHI_BOUNDS[1], atol=1e-6))
    cnt_theta = np.exp(res.x[1 + p:])

    # (c) 逐细胞后验
    P = est_post.zero_source_posterior_cov(f_cov["Gamma"], Wd, cnt_theta,
                                           diag["cnt_phi"], N)
    diag["t_est"] = diag["t_cov"] + diag["t_count"]

    zero = Y == 0
    labels = truth["structural_zeros"][zero].astype(int)
    scores = P[zero]
    ii, jj = np.where(zero)
    np.savez_compressed(
        args.out,
        scores=scores, labels=labels, depth=N[ii],
        group=group[ii], taxon=jj, sample=ii,
        cell_id=args.cell, rep=args.rep, arm=args.arm,
        prior_lam=lam, prior_eta0=args.eta0,
        cov_multistart=int(cov_ms),
        mechanism=str(row["mechanism"]),
        informative=bool(row["informative_zeros"]),
        sz=float(row["structural_zero_rate"]),
        n=int(row["n"]), depth_cfg=int(row["depth"]),
        phi_true=float(row["dispersion_value"]),
        phi_hat=diag["cnt_phi"], cov_phi=diag["cov_phi"],
        cov_phi_on_boundary=diag["cov_phi_on_boundary"],
        cnt_phi_on_boundary=diag["cnt_phi_on_boundary"],
        cov_converged=diag["cov_converged"],
        cnt_converged=diag["cnt_converged"],
        cov_success=diag["cov_success"], cnt_success=diag["cnt_success"],
        cov_nit=diag["cov_nit"], cnt_nit=diag["cnt_nit"],
        t_cov=diag["t_cov"], t_count=diag["t_count"], t_est=diag["t_est"])
    print(f"saved {args.out} zeros={int(zero.sum())} "
          f"struct={labels.mean():.3f} post_mean={scores.mean():.3f} "
          f"cov_phi={diag['cov_phi']:.3g} cnt_phi={diag['cnt_phi']:.3g} "
          f"t_est={diag['t_est']:.0f}s cov_ms={cov_ms} lam={lam}")


if __name__ == "__main__":
    main()
