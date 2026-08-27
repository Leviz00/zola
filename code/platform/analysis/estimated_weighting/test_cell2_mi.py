"""cell 2 迭代预算检验：count refine mi200 vs mi600 vs mi1200 对后验与 welch FDP 的影响。"""
import os, sys, time, json
import numpy as np
import pandas as pd

SIM = "/mnt/agents/output/code/simulation"
sys.path.insert(0, SIM)
from run_estimated_weighting import (_count_refine, est_cl, est_clcov,
                                     est_post, validate_weights,
                                     weighted_welch_t, exclusion_wilcoxon,
                                     metrics, design, generators)

cfg = pd.read_csv(os.path.join(SIM, "configs", "config_fractional.csv"))
row = cfg[cfg.cell_id == 2].iloc[0]
params = design.params_for_cell(row)
seeds = np.random.SeedSequence(int(row["seed"])).spawn(100)

for rep in [0, 1]:
    Y, truth = generators.generate(row["mechanism"], params, n=int(row["n"]),
                                   p=100, depths=int(row["depth"]), seed=seeds[rep])
    group = truth["group"]
    D = (Y > 0).astype(float)
    N = Y.sum(axis=1).astype(float)
    Wd = np.column_stack([np.ones(len(group)), group.astype(float)])
    f_cov = est_clcov.fit_composite_cov(D, Wd, N, phi_known=None,
                                        multi_start=False, maxiter=500)
    f_det = est_cl.fit_composite(D, N, phi_known=None, multi_start=False,
                                 maxiter=500)
    rec = {"rep": rep}
    psi = (f_det["pi"], f_det["theta"], f_det["phi"])
    for mi in [200, 600, 1200]:
        t0 = time.time()
        r = _count_refine(Y, N, *psi, maxiter=mi)
        P = est_post.zero_source_posterior_cov(f_cov["Gamma"], Wd, r["theta"],
                                               r["phi"], N)
        W = validate_weights(P, Y)
        z = Y == 0
        labels = truth["structural_zeros"][z].astype(float)
        rej_w = weighted_welch_t(Y, group, W)["reject"]
        rej_e = exclusion_wilcoxon(Y, group, W)["reject"]
        fdp_w, _ = metrics.fdp(rej_w, truth["da_taxa"])
        fdp_e, _ = metrics.fdp(rej_e, truth["da_taxa"])
        rec[f"mi{mi}"] = {"nit": r["nit"], "phi": round(r["phi"], 1),
                          "auc": round(metrics.auc(W[z], labels.astype(bool)), 3),
                          "post_mean": round(float(W[z].mean()), 3),
                          "fdp_welch": round(fdp_w, 3), "fdp_excl": round(fdp_e, 3),
                          "t": round(time.time() - t0, 0)}
        psi = (r["pi"], r["theta"], r["phi"])  # 续跑（热启动链）
    print(json.dumps(rec), flush=True)
print("DONE")
