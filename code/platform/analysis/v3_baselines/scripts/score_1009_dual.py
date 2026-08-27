"""score_1009_dual.py — Amendment 2：cell 1009 双真值打分。

4 外部方法（raw 文件重读）+ 3 参考臂（abs_nb_glm 渐近 raw 口径逐 rep
重跑，arm 定义与 run_one_v34f 155-175 行一致：oracle=~sz、
estimated=npz 分数≥0.5 剔除、placeholder=None）。
真值 A = abs_da_truth；真值 B = abs_da ∪ 缺席 taxa（逐 rep truthB CSV）。
各出一套 emp_fdr_uncond/emp_fdr_cond/TPR(±MCSE)。
输出：scores_1009_dual.csv。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

SIM = "/mnt/agents/output/code/simulation_v3"
V34 = "/mnt/agents/output/analysis/method_fix/v3/v34_full"
BASE = "/mnt/agents/output/analysis/v3_baselines"
sys.path.insert(0, SIM)
import design  # noqa: E402
import generators  # noqa: E402
from abs_glm import abs_nb_glm  # noqa: E402

CFG = pd.read_csv(f"{SIM}/configs/config_supplementary.csv")
METHODS = ["linda", "ancombc2", "deseq2", "tss_wilcoxon"]


def mcse(x):
    x = np.asarray(x, float)
    return x.std() / np.sqrt(len(x)) if len(x) > 1 else np.nan


def agg(rows):
    df = pd.DataFrame(rows)
    fdp0 = df.fdp.fillna(0.0).values
    rr = df[df.n_rej >= 1]
    return dict(n_rej_reps=len(rr),
                emp_fdr_uncond=fdp0.mean(), emp_fdr_mcse=mcse(fdp0),
                emp_fdr_cond=rr.fdp.mean() if len(rr) else np.nan,
                tpr=df.tpr.mean(), tpr_mcse=mcse(df.tpr.values))


def score_rej(rej, da):
    fp, tp = int((rej & ~da).sum()), int((rej & da).sum())
    return dict(n_rej=fp + tp,
                fdp=fp / (fp + tp) if fp + tp else np.nan,
                tpr=tp / max(int(da.sum()), 1))


def ref_arm_rejects(cell, rep):
    """重跑三臂渐近 GLM，返回 {arm: reject 布尔向量}。"""
    row = CFG[CFG.cell_id == cell].iloc[0]
    seeds = np.random.SeedSequence(int(row["seed"])).spawn(20)
    prm = design.params_for_cell(row)
    prm["effect_mode"] = "absolute"
    Y, truth = generators.generate(row["mechanism"], prm, n=int(row["n"]),
                                   p=100, depths=int(row["depth"]),
                                   seed=seeds[rep])
    Y = np.asarray(Y, float)
    N = truth["depths"].astype(float)
    group = truth["group"].astype(float)
    sz = np.asarray(truth["structural_zeros"], dtype=bool)
    z = np.load(f"{V34}/npz/cell{cell}_rep{rep}.npz")
    W_hat = np.zeros(Y.shape)
    W_hat[Y == 0] = z["scores"]
    arms = {}
    keep_o = ~((Y == 0) & (sz.astype(float) >= 0.5))
    arms["oracle"] = abs_nb_glm(Y, group, N=N, W=keep_o.astype(float))["reject"]
    keep_e = ~((Y == 0) & (W_hat >= 0.5))
    arms["estimated"] = abs_nb_glm(Y, group, N=N, W=keep_e.astype(float))["reject"]
    arms["placeholder"] = abs_nb_glm(Y, group, N=N, W=None)["reject"]
    return arms


def main():
    out_rows = []
    overlap = []
    for rep in range(20):
        tb = pd.read_csv(f"{BASE}/exchange/truth/cell1009_rep{rep}_truthB.csv")
        da_a = tb.abs_da_truth.values.astype(bool)
        da_b = tb.truth_b.values.astype(bool)
        overlap.append(int((tb.abs_da_truth & tb.absent_truth).sum()))
        for m in METHODS:
            try:
                d = pd.read_csv(f"{BASE}/raw/{m}/cell1009_rep{rep}.csv")
                rej = d["rejected"].fillna(0).astype(int).values.astype(bool)
            except FileNotFoundError:
                continue
            for truth_name, da in (("A", da_a), ("B", da_b)):
                out_rows.append(dict(method=m, arm=None, truth=truth_name,
                                     rep=rep, **score_rej(rej, da)))
        for arm, rej in ref_arm_rejects(1009, rep).items():
            for truth_name, da in (("A", da_a), ("B", da_b)):
                out_rows.append(dict(method=f"ref_{arm}", arm=arm,
                                     truth=truth_name, rep=rep,
                                     **score_rej(rej, da)))
        print(f"rep {rep} done", flush=True)
    rep_df = pd.DataFrame(out_rows)
    rep_df.to_csv(f"{BASE}/scores_1009_dual_replevel.csv", index=False)
    summ = (rep_df.groupby(["method", "truth"])
            .apply(lambda g: pd.Series(agg(g.to_dict("records"))),
                   include_groups=False).reset_index())
    summ.to_csv(f"{BASE}/scores_1009_dual.csv", index=False)
    print("overlap abs_da ∩ absent per rep:", overlap)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
