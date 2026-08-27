"""预算臂（budget arm）保真度与耗时检验。

对照全保真拟合（fit_composite_cov 多起点 + fit_count_composite b=4）：
  预算臂 = fit_composite_cov(multi_start=False)          # 公开 API，跳过暖启动
         + 自驱动 count 块精炼（clc.count_loglik_grad 公开目标，
           从协变量拟合的 (θ̂,φ̂) 与汇总检出率 π₀ 单起点 L-BFGS-B，无 Godambe）
比较：耗时、φ̂、后验矩阵相关、后验 AUC（对模拟真值，仅诊断）。
格子：cell 16（n=600, three_layer, high disp, off）、cell 19（n=200, zinb, on）。
"""
import os, sys, time, json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit, expit

SIM = "/mnt/agents/output/code/simulation"
EST = "/mnt/agents/output/code/estimation"
sys.path.insert(0, SIM); sys.path.insert(0, EST)

import design, generators
import composite_likelihood_cov as clcov
import composite_likelihood_count as clc
import posterior as post

cfg = pd.read_csv(os.path.join(SIM, "configs", "config_fractional.csv"))

def gen_cell(cell_id, rep=0):
    row = cfg[cfg.cell_id == cell_id].iloc[0]
    params = design.params_for_cell(row)
    seeds = np.random.SeedSequence(int(row["seed"])).spawn(100)
    Y, truth = generators.generate(row["mechanism"], params, n=int(row["n"]),
                                   p=100, depths=int(row["depth"]), seed=seeds[rep])
    return Y, truth

def count_refine(Y, N, pi0, theta0, phi0, b=4, maxiter=300):
    """自驱动计数块精炼：总 CL = 检出指示 CL + 零截断 DM 块幅度项。

    ψ 布局（φ 未知）：(logφ, logit π_1..p, log θ_1..p)，见 cl._pack 文档。
    仅用 clc.count_loglik_grad 公开目标 + 单起点 L-BFGS-B。
    """
    p = Y.shape[1]
    blocks = clc.make_blocks(p, b)
    psi0 = np.concatenate([[np.log(phi0)], logit(np.clip(pi0, 1e-4, 0.9999)),
                           np.log(np.clip(theta0, 1e-7, 0.9))])
    bounds = ([(np.log(0.05), np.log(1e5))]
              + [(logit(1e-4), logit(0.9999))] * p
              + [(np.log(1e-7), np.log(0.9))] * p)

    def obj(psi):
        ll, g = clc.count_loglik_grad(psi, Y, N, blocks, phi_known=None)
        ng = float(np.abs(g).max())
        if ng > 1e3:  # 与 fit_count_composite 相同的梯度裁剪
            g = g * (1e3 / ng)
        return -ll, -g

    res = minimize(obj, psi0, method="L-BFGS-B", jac=True, bounds=bounds,
                   options={"maxiter": maxiter, "ftol": 1e-12, "gtol": 1e-8})
    phi = np.exp(res.x[0])
    pi = expit(res.x[1:1 + p])
    theta = np.exp(res.x[1 + p:])
    return {"phi": phi, "pi": pi, "theta": theta, "success": bool(res.success),
            "nit": int(res.nit)}

def budget_pipeline(Y, group):
    D = (Y > 0).astype(float)
    N = Y.sum(axis=1).astype(float)
    W = np.column_stack([np.ones(len(group)), group.astype(float)])
    t0 = time.time()
    f_cov = clcov.fit_composite_cov(D, W, N, phi_known=None,
                                    multi_start=False, maxiter=500)
    t_cov = time.time() - t0
    # 计数块起点：π₀ = 逐类群汇总检出率的 logit（截距），θ̂/φ̂ 用协变量拟合值
    t0 = time.time()
    pi0 = expit(f_cov["Gamma"][:, 0])
    f_cnt = count_refine(Y, N, pi0, f_cov["theta"], f_cov["phi"])
    t_cnt = time.time() - t0
    P = post.zero_source_posterior_cov(f_cov["Gamma"], W, f_cnt["theta"],
                                       f_cnt["phi"], N)
    return f_cov, f_cnt, P, {"t_cov": t_cov, "t_cnt": t_cnt}

for cid in [16, 19]:
    Y, truth = gen_cell(cid)
    t0 = time.time()
    f_cov, f_cnt, P, tt = budget_pipeline(Y, truth["group"])
    zero = Y == 0
    labels = truth["structural_zeros"][zero].astype(float)
    rec = {"cell_id": cid, **{k: round(v, 1) for k, v in tt.items()},
           "t_total": round(time.time() - t0, 1),
           "cov_success": bool(f_cov["success"]), "cov_phi": round(float(f_cov["phi"]), 1),
           "cnt_success": f_cnt["success"], "cnt_phi": round(float(f_cnt["phi"]), 1),
           "cnt_nit": f_cnt["nit"],
           "auc": round(post.auc_score(labels, P[zero]), 4),
           "post_mean": round(float(P[zero].mean()), 4),
           "struct_frac": round(float(labels.mean()), 4)}
    print(json.dumps(rec), flush=True)

print("DONE")
