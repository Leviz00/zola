"""按 cell_tasks 相同种子协议重跑指定 (cell, rep)，截取逐零后验与真值标签。

用法: python3 extract_zero_posteriors.py <cell_id> <rep> <R_spawn> <out.npz>
输出: scores (ZOLA 后验, 仅 Y=0 处), labels (oracle 结构零真值),
      N (样本深度, 与零位一一对应), group, da_taxa, truth 概要
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design, generators  # noqa: E402
import run_estimated_weighting as rew  # noqa: E402

cell_id, rep, R = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
out = sys.argv[4]

cfg = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "configs", "config_fractional.csv"))
row = cfg[cfg.cell_id == cell_id].iloc[0]
child_seeds = np.random.SeedSequence(int(row["seed"])).spawn(R)
params = design.params_for_cell(row)

Y, truth = generators.generate(row["mechanism"], params, n=int(row["n"]),
                               p=rew.P_TAXA, depths=int(row["depth"]),
                               seed=child_seeds[rep])
W_est, diag = rew.estimated_weights(Y, truth["group"])

zero = Y == 0
labels = truth["structural_zeros"][zero].astype(int)
scores = W_est[zero]
ii, jj = np.where(zero)
N = Y.sum(axis=1).astype(float)
np.savez_compressed(out, scores=scores, labels=labels, depth=N[ii],
                    group=truth["group"][ii], taxon=jj, sample=ii,
                    cell_id=cell_id, rep=rep,
                    mechanism=str(row["mechanism"]),
                    informative=bool(row["informative_zeros"]),
                    sz=float(row["structural_zero_rate"]),
                    cov_phi=float(diag.get("cov_phi", np.nan)),
                    cnt_phi=float(diag.get("cnt_phi", np.nan)),
                    cov_success=bool(diag.get("cov_success", False)))
print("saved", out, "zeros:", zero.sum(), "struct frac:", labels.mean(),
      "post mean:", scores.mean(), "cov_phi:", diag.get("cov_phi"))
