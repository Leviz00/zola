"""预算臂 v2：det-CL 暖启动 + 放宽收敛的计数块精炼；并量化计数块的贡献。

变体（同数据对照）：
  A. cov(multi_start=False, maxiter=300) + 无计数块（后验用 cov 的 θ̂,φ̂）
  B. cov + count_refine(从 det-CL multi_start=False 暖启动, maxiter=200,
                        gtol=1e-6, ftol=1e-10)
  C. cov + count_refine(从 cov 的 θ̂,φ̂ 启动, maxiter=300, gtol=1e-8)  ← v1 臂
格子：16（n=600 off 难格）、19（n=200 on）、8（n=600 on depth 100k）、
      33（n=100 off 小样本）。
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
import composite_likelihood as cl
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

def count_refine(Y, N, pi0, theta0, phi0, b=4, maxiter=200,
                 gtol=1e-6, ftol=1e-10):
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
        if ng > 1e3:
            g = g * (1e3 / ng)
        return -ll, -g

    t0 = time.time()
    res = minimize(obj, psi0, method="L-BFGS-B", jac=True, bounds=bounds,
                   options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol})
    return {"phi": float(np.exp(res.x[0])), "pi": expit(res.x[1:1 + p]),
            "theta": np.exp(res.x[1 + p:]), "success": bool(res.success),
            "nit": int(res.nit), "t": time.time() - t0}

def diag(P, Y, truth):
    zero = Y == 0
    labels = truth["structural_zeros"][zero].astype(float)
    return {"auc": round(post.auc_score(labels, P[zero]), 4),
            "post_mean": round(float(P[zero].mean()), 4),
            "struct_frac": round(float(labels.mean()), 4)}

for cid in [16, 19, 8, 33]:
    Y, truth = gen_cell(cid)
    group = truth["group"]
    D = (Y > 0).astype(float)
    N = Y.sum(axis=1).astype(float)
    W = np.column_stack([np.ones(len(group)), group.astype(float)])
    rec = {"cell_id": cid, "n_rows": int(Y.shape[0])}

    t0 = time.time()
    f_cov = clcov.fit_composite_cov(D, W, N, phi_known=None,
                                    multi_start=False, maxiter=300)
    rec["t_cov"] = round(time.time() - t0, 1)
    rec["cov_phi"] = round(float(f_cov["phi"]), 1)

    # 臂 A：无计数块
    PA = post.zero_source_posterior_cov(f_cov["Gamma"], W, f_cov["theta"],
                                        f_cov["phi"], N)
    rec["A"] = diag(PA, Y, truth)

    # det-CL 暖启动（multi_start=False 的廉价联合拟合）
    t0 = time.time()
    f_det = cl.fit_composite(D, N, phi_known=None, multi_start=False,
                             maxiter=500)
    rec["t_det"] = round(time.time() - t0, 1)
    rec["det_phi"] = round(float(f_det["phi"]), 1)

    # 臂 B：det-CL 暖启动 + 放宽收敛
    rB = count_refine(Y, N, f_det["pi"], f_det["theta"], f_det["phi"])
    PB = post.zero_source_posterior_cov(f_cov["Gamma"], W, rB["theta"],
                                        rB["phi"], N)
    rec["B"] = {**diag(PB, Y, truth), "t": round(rB["t"], 1),
                "nit": rB["nit"], "phi": round(rB["phi"], 1),
                "success": rB["success"]}

    # 臂 C（v1）：cov θ,φ 启动、严收敛
    rC = count_refine(Y, N, expit(f_cov["Gamma"][:, 0]), f_cov["theta"],
                      f_cov["phi"], maxiter=300, gtol=1e-8, ftol=1e-12)
    PC = post.zero_source_posterior_cov(f_cov["Gamma"], W, rC["theta"],
                                        rC["phi"], N)
    rec["C"] = {**diag(PC, Y, truth), "t": round(rC["t"], 1),
                "nit": rC["nit"], "phi": round(rC["phi"], 1),
                "success": rC["success"]}

    # 后验矩阵两两相关（零单元）
    z = (Y == 0)
    rec["corr_B_C"] = round(float(np.corrcoef(PB[z], PC[z])[0, 1]), 4)
    rec["corr_A_B"] = round(float(np.corrcoef(PA[z], PB[z])[0, 1]), 4)
    rec["t_total"] = round(rec["t_cov"] + rec["t_det"] + rB["t"], 1)
    print(json.dumps(rec), flush=True)

print("DONE")
