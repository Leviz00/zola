"""分阶段剖析：cell 19 (zinb, n=100/组) 上协变量 CL / 计数块 CL 各自耗时。

变体对照：joint cov 全多起点 vs maxiter=300；profile(φ=1000) 臂耗时；
计数块 b=4 全拟合耗时。用于决定 48 格的算力配置。
"""
import os, sys, time, json
import numpy as np
import pandas as pd

SIM = "/mnt/agents/output/code/simulation"
EST = "/mnt/agents/output/code/estimation"
sys.path.insert(0, SIM); sys.path.insert(0, EST)

import design, generators
import composite_likelihood_cov as clcov
import composite_likelihood_count as clc
import composite_likelihood as cl
import posterior as post

cfg = pd.read_csv(os.path.join(SIM, "configs", "config_fractional.csv"))
row = cfg[cfg.cell_id == 19].iloc[0]
params = design.params_for_cell(row)
seeds = np.random.SeedSequence(int(row["seed"])).spawn(100)
Y, truth = generators.generate(row["mechanism"], params, n=int(row["n"]),
                               p=100, depths=int(row["depth"]), seed=seeds[0])
group = truth["group"]
D = (Y > 0).astype(float)
N = Y.sum(axis=1).astype(float)
W = np.column_stack([np.ones(len(group)), group.astype(float)])
out = {}

t0 = time.time()
f_prof = clcov.fit_composite_cov(D, W, N, phi_known=1000.0)
out["cov_profile_phi1000"] = round(time.time() - t0, 1)
print(json.dumps({"cov_profile_phi1000": out["cov_profile_phi1000"]}), flush=True)

t0 = time.time()
f_joint_fast = clcov.fit_composite_cov(D, W, N, phi_known=None, maxiter=300)
out["cov_joint_maxiter300"] = round(time.time() - t0, 1)
print(json.dumps({"cov_joint_maxiter300": out["cov_joint_maxiter300"],
                  "success": bool(f_joint_fast["success"])}), flush=True)

t0 = time.time()
f_det = cl.fit_composite(D, N, phi_known=None, multi_start=True)
out["det_cl_joint"] = round(time.time() - t0, 1)
print(json.dumps({"det_cl_joint": out["det_cl_joint"]}), flush=True)

t0 = time.time()
f_cnt = clc.fit_count_composite(Y, N, b=4, phi_known=None)
out["count_b4_joint"] = round(time.time() - t0, 1)
print(json.dumps({"count_b4_joint": out["count_b4_joint"],
                  "success": bool(f_cnt["success"])}), flush=True)

# 后验质量对照：maxiter300 vs 全拟合 的 Γ/θ 差异（用同一 θ,φ 源）
P_fast = post.zero_source_posterior_cov(f_joint_fast["Gamma"], W,
                                        f_cnt["theta"], f_cnt["phi"], N)
zero = Y == 0
labels = truth["structural_zeros"][zero].astype(float)
out["auc_fast"] = round(post.auc_score(labels, P_fast[zero]), 4)
print(json.dumps(out), flush=True)
print("DONE")
