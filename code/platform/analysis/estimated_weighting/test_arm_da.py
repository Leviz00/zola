"""关键决策实验：各预算臂的 Ŵ 对【下游 DA 结局】的影响 + placeholder AUC 参照。

格子 16/19 rep 0。输出每臂：后验 AUC、weighted_welch_t / exclusion_wilcoxon
的 FDP/TPR/拒绝数，对照 oracle / placeholder / unweighted。
臂：A=cov only；B=det 暖启动+mi200 松收敛；C=cov 启动+mi300 严收敛；
    F=全驱动 fit_count_composite（仅 cell 19，参照）。
"""
import os, sys, time, json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit, expit

SIM = "/mnt/agents/output/code/simulation"
EST = "/mnt/agents/output/code/estimation"
sys.path.insert(0, SIM); sys.path.insert(0, EST)

import design, generators, metrics
import composite_likelihood as cl
import composite_likelihood_cov as clcov
import composite_likelihood_count as clc
import posterior as post
from weighting import (oracle_weights, placeholder_weights, weighted_welch_t,
                       exclusion_wilcoxon)
from baselines_py import naive_welch_t, tss_wilcoxon

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

def da_outcomes(Y, truth, W, tag):
    group = truth["group"]
    out = {}
    for name, rej in [
        (f"{tag}_welch", weighted_welch_t(Y, group, W)["reject"]),
        (f"{tag}_excl", exclusion_wilcoxon(Y, group, W)["reject"]),
    ]:
        fdp_r, n_rej = metrics.fdp(rej, truth["da_taxa"])
        tpr_r, _ = metrics.tpr(rej, truth["da_taxa"])
        out[name] = {"fdp": round(fdp_r, 3), "tpr": None if np.isnan(tpr_r) else round(tpr_r, 3),
                     "n_rej": n_rej}
    return out

def auc_of(P, Y, truth):
    z = Y == 0
    return round(post.auc_score(truth["structural_zeros"][z].astype(float), P[z]), 4)

for cid in [19, 16]:
    Y, truth = gen_cell(cid)
    group = truth["group"]
    D = (Y > 0).astype(float)
    N = Y.sum(axis=1).astype(float)
    Wd = np.column_stack([np.ones(len(group)), group.astype(float)])
    rec = {"cell_id": cid}

    # 参照档
    W_or = oracle_weights(Y, truth)
    W_ph = placeholder_weights(Y, group)
    rec["auc_oracle"] = auc_of(W_or, Y, truth)
    rec["auc_placeholder"] = auc_of(W_ph, Y, truth)
    rec["oracle"] = da_outcomes(Y, truth, W_or, "oracle")
    rec["placeholder"] = da_outcomes(Y, truth, W_ph, "ph")
    for nm, rr in [("naive_welch", naive_welch_t(Y, group)),
                   ("tss_wilcoxon", tss_wilcoxon(Y, group))]:
        fdp_r, n_rej = metrics.fdp(rr["reject"], truth["da_taxa"])
        tpr_r, _ = metrics.tpr(rr["reject"], truth["da_taxa"])
        rec[nm] = {"fdp": round(fdp_r, 3),
                   "tpr": None if np.isnan(tpr_r) else round(tpr_r, 3), "n_rej": n_rej}

    # 协变量层（各臂共用）
    f_cov = clcov.fit_composite_cov(D, Wd, N, phi_known=None,
                                    multi_start=False, maxiter=500)
    f_det = cl.fit_composite(D, N, phi_known=None, multi_start=False,
                             maxiter=500)
    rec["cov_phi"] = round(float(f_cov["phi"]), 1)
    rec["det_phi"] = round(float(f_det["phi"]), 1)

    arms = {}
    PA = post.zero_source_posterior_cov(f_cov["Gamma"], Wd, f_cov["theta"],
                                        f_cov["phi"], N)
    arms["A_cov_only"] = PA
    rB = count_refine(Y, N, f_det["pi"], f_det["theta"], f_det["phi"])
    arms["B_det_mi200"] = post.zero_source_posterior_cov(
        f_cov["Gamma"], Wd, rB["theta"], rB["phi"], N)
    rec["B_info"] = {"t": round(rB["t"], 1), "nit": rB["nit"],
                     "phi": round(rB["phi"], 1), "success": rB["success"]}
    rB2 = count_refine(Y, N, f_det["pi"], f_det["theta"], f_det["phi"],
                       maxiter=1000, gtol=1e-8, ftol=1e-12)
    arms["B2_det_mi1000"] = post.zero_source_posterior_cov(
        f_cov["Gamma"], Wd, rB2["theta"], rB2["phi"], N)
    rec["B2_info"] = {"t": round(rB2["t"], 1), "nit": rB2["nit"],
                      "phi": round(rB2["phi"], 1), "success": rB2["success"]}
    if False:  # 全驱动参照已取消（>15min/rep，不可行；B2 为实用高保真参照）
        t0 = time.time()
        f_full = clc.fit_count_composite(Y, N, b=4, phi_known=None)
        rec["F_info"] = {"t": round(time.time() - t0, 1),
                         "phi": round(float(f_full["phi"]), 1),
                         "success": bool(f_full["success"])}
        arms["F_full_driver"] = post.zero_source_posterior_cov(
            f_cov["Gamma"], Wd, f_full["theta"], f_full["phi"], N)

    for tag, P in arms.items():
        rec[tag] = {"auc": auc_of(P, Y, truth),
                    **da_outcomes(Y, truth, P, tag)}
    print(json.dumps(rec), flush=True)

print("DONE")
