"""trigger_v38b.py — Stage B part 1：data-only 饱和触发统计 + est-new AUC。

触发判据（data-only，v3.5 门控精神）：逐 rep 计算 Ŵ 分数与
"存在率信号" s_ij = 1 − π̂_emp_j（π̂_emp_j = 类群 j 的经验检出率，
= mean(Y[:,j]>0)，纯数据）在零细胞上的 Pearson 相关 corr_s。
直觉：健康后验（真饱和时也应）按 1−π 排序 → corr_s 高；ĝ 塌缩/变异
淹没 π̂ 时 corr_s≈0/负 → 触发退化，Ŵ_new ← 1−π̂_emp_j（零细胞处）。
阈值 τ 由 LOCO 在五格（有 Stage A π-缺口标签）上导出/验证。
输出：v38b_trigger.csv（8 格×20 rep：corr_s、fired@候选 τ、
est-old/new AUC、π-only 参考）。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

SIM = "/mnt/agents/output/code/simulation_v3"
V34 = "/mnt/agents/output/analysis/method_fix/v3/v34_full"
OUT = "/mnt/agents/output/analysis/method_fix/v3/v38_resolution"
sys.path.insert(0, SIM)
import design  # noqa: E402
import generators  # noqa: E402

CELLS = [1000, 1002, 1004, 1005, 1006, 1007, 1008, 1009]
AUC_CELLS = [1004, 1005, 1006, 1007, 1008]
CFG = pd.read_csv(f"{SIM}/configs/config_supplementary.csv")


def mw_auc(score, label):
    from scipy.stats import rankdata
    r = rankdata(score)
    pos = label == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def rep_stats(cell, rep, tau):
    row = CFG[CFG.cell_id == cell].iloc[0]
    seeds = np.random.SeedSequence(int(row["seed"])).spawn(20)
    prm = design.params_for_cell(row)
    prm["effect_mode"] = "absolute"
    z = np.load(f"{V34}/npz/cell{cell}_rep{rep}.npz")
    scores, labels = z["scores"], z["labels"]
    Y, truth = generators.generate(row["mechanism"], prm, n=int(row["n"]),
                                   p=100, depths=int(row["depth"]),
                                   seed=seeds[rep])
    Y = np.asarray(Y, float)
    zero = Y == 0
    assert zero.sum() == len(scores)
    pi_emp = (Y > 0).mean(axis=0)          # 经验检出率（data-only）
    s_zero = 1.0 - np.broadcast_to(pi_emp[None, :], Y.shape)[zero]
    corr_s = np.corrcoef(scores, s_zero)[0, 1] if scores.std() > 0 else 0.0
    fired = corr_s < tau
    new_scores = np.where(fired, s_zero, scores)
    rec = dict(cell_id=cell, rep=rep, mechanism=row["mechanism"],
               share=float(z["struct_frac"]), corr_s=corr_s, fired=fired,
               phi_hat=float(z["phi_hat"]),
               auc_old=mw_auc(scores, labels),
               auc_new=mw_auc(new_scores, labels))
    return rec


def main(tau=0.15):
    rows = [rep_stats(c, r, tau) for c in CELLS for r in range(20)]
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/v38b_trigger.csv", index=False)
    print(f"--- fired rate @ tau={tau} ---")
    print(df.groupby("cell_id").fired.mean().round(2).to_string())
    a = df[df.cell_id.isin(AUC_CELLS)]
    print("--- AUC old vs new (fired reps use fallback) ---")
    print(a.groupby("cell_id")[["auc_old", "auc_new"]]
          .agg(["mean", "median"]).round(3).to_string())
    # LOCO：在其余 4 个 AUC 格上选 tau（最大化与 Stage A π-缺口>0.1 标签的
    # 一致率），在留出格上评估
    gap_label = {1004: 1, 1005: 0, 1006: 1, 1007: 1, 1008: 1}  # Stage A 结论
    loco = []
    grid = np.round(np.arange(0.0, 0.31, 0.05), 2)
    for hold in AUC_CELLS:
        tr = a[a.cell_id != hold]
        best_t, best_acc = 0.15, -1
        for t in grid:
            pred = (tr.corr_s < t).astype(int)
            lab = tr.cell_id.map(gap_label).values
            acc = (pred == lab).mean()
            if acc > best_acc:
                best_acc, best_t = acc, t
        te = a[a.cell_id == hold]
        acc_te = ((te.corr_s < best_t).astype(int)
                  == te.cell_id.map(gap_label).values).mean()
        loco.append(dict(holdout=hold, tau=best_t, train_acc=best_acc,
                         test_acc=acc_te))
    L = pd.DataFrame(loco)
    L.to_csv(f"{OUT}/v38b_loco.csv", index=False)
    print("--- LOCO ---")
    print(L.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
