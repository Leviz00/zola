"""基准计时：单个重复上跑完整 ZOLA 估计管线（协变量层 CL + 计数块 + 后验）。

在两个代表性 fractional 格子上各跑 1 个重复：
  cell 16 = three_layer, n=300/组, depth 20000, sz=0.3, informative off（最大 n）
  cell 19 = zinb,        n=100/组, depth 5000,  sz=0.1, informative on
记录各阶段墙钟、收敛状态、φ̂、后验 AUC（对照真值，仅 sanity 用）。
"""
import os, sys, time, json
import numpy as np
import pandas as pd

SIM = "/mnt/agents/output/code/simulation"
EST = "/mnt/agents/output/code/estimation"
sys.path.insert(0, SIM)
sys.path.insert(0, EST)

import design, generators
import composite_likelihood_cov as clcov
import composite_likelihood_count as clc
import posterior as post

OUT = "/mnt/agents/output/analysis/estimated_weighting"
cfg = pd.read_csv(os.path.join(SIM, "configs", "config_fractional.csv"))

def gen_cell(cell_id, rep=0):
    row = cfg[cfg.cell_id == cell_id].iloc[0]
    params = design.params_for_cell(row)
    seeds = np.random.SeedSequence(int(row["seed"])).spawn(100)
    Y, truth = generators.generate(row["mechanism"], params, n=int(row["n"]),
                                   p=100, depths=int(row["depth"]), seed=seeds[rep])
    return Y, truth

def bench(cell_id, rep=0):
    Y, truth = gen_cell(cell_id, rep)
    group = truth["group"]
    D = (Y > 0).astype(float)
    N = Y.sum(axis=1).astype(float)
    W = np.column_stack([np.ones(len(group)), group.astype(float)])
    rec = {"cell_id": cell_id, "rep": rep, "n_rows": Y.shape[0],
           "zero_frac": float((Y == 0).mean())}
    t0 = time.time()
    f_cov = clcov.fit_composite_cov(D, W, N, phi_known=None)
    rec["t_cov"] = time.time() - t0
    rec["cov_success"] = bool(f_cov["success"])
    rec["cov_phi"] = float(f_cov["phi"])
    t0 = time.time()
    f_cnt = clc.fit_count_composite(Y, N, b=4, phi_known=None)
    rec["t_count"] = time.time() - t0
    rec["cnt_success"] = bool(f_cnt["success"])
    rec["cnt_phi"] = float(f_cnt["phi"])
    t0 = time.time()
    P = post.zero_source_posterior_cov(f_cov["Gamma"], W, f_cnt["theta"],
                                       f_cnt["phi"], N)
    rec["t_post"] = time.time() - t0
    # sanity：后验对真值结构零的 AUC（仅基准用，不进正式指标）
    zero = Y == 0
    labels = truth["structural_zeros"][zero].astype(float)
    rec["post_auc"] = post.auc_score(labels, P[zero])
    rec["post_mean"] = float(P[zero].mean())
    rec["struct_frac"] = float(labels.mean())
    return rec

rows = []
for cid in [16, 19]:
    t0 = time.time()
    r = bench(cid)
    r["t_total"] = time.time() - t0
    print(json.dumps(r), flush=True)
    rows.append(r)

pd.DataFrame(rows).to_csv(os.path.join(OUT, "bench_estimation.csv"), index=False)
print("DONE")
