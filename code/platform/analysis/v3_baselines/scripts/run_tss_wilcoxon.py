"""run_tss_wilcoxon.py — 第 4 个 baseline：TSS+Wilcoxon（SPEC 附录 A）。

逻辑逐字复用 code/simulation/baselines_py.py（只读 import）：
tss_relative_abundance / filter_empty_samples / bh_reject /
mannwhitneyu(two-sided) 调用方式与 ALPHA=0.05 锁定。

v3 适配差异（相对老实现，记入 raw/tss_wilcoxon/README.md）：
1. 老实现把"全恒定 taxon"p 值置 1.0 并留在 BH 分母；v3 按 SPEC §2 把
   **全零 taxon** 记 p_value=NA、filtered=TRUE、剔除出 BH（其余恒定非零
   taxon 保持老行为 p=1.0）。两口径在深度高格几乎无差异（全零 taxon
   稀少），在低深度格更接近 R 基线的 prevalence-filter 惯例。
2. 老实现只出 reject 掩码；v3 另出 BH 校正 q_value（tested 类群内，
   statsmodels multipletests fdr_bh；filtered=NA）。
3. 老实现对固定 8 个 alpha 出矩阵；v3 仅 alpha=0.05 一档。

输出：raw/tss_wilcoxon/cell{cid}_rep{r}.csv
      列 taxon,p_value,q_value,rejected,filtered（100 行）。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

BASE = "/mnt/agents/output/analysis/v3_baselines"
sys.path.insert(0, "/mnt/agents/output/code/simulation")
from baselines_py import (ALPHA, bh_reject, filter_empty_samples,  # noqa: E402
                          tss_relative_abundance)
from scipy import stats  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

CELLS = [1000, 1002, 1004, 1005, 1006, 1007, 1008, 1009]
R_REPS = 20


def run_one(cell, rep):
    Y = pd.read_csv(f"{BASE}/exchange/counts/cell{cell}_rep{rep}.csv",
                    index_col=0).values
    meta = pd.read_csv(f"{BASE}/exchange/meta/cell{cell}_rep{rep}.csv")
    group = meta["group"].values.astype(int)
    taxa = [f"taxon_{j+1}" for j in range(Y.shape[1])]

    Y, group, _, _ = filter_empty_samples(Y, group)
    rel = tss_relative_abundance(Y)
    a, b = rel[group == 0], rel[group == 1]
    p = Y.shape[1]
    pvals = np.full(p, np.nan)
    allzero = (Y.sum(axis=0) == 0)
    for j in range(p):
        if allzero[j]:
            continue  # 全零 → p=NA, filtered（SPEC §2）
        x, y = b[:, j], a[:, j]
        if np.all(x == x[0]) and np.all(y == y[0]) and x[0] == y[0]:
            pvals[j] = 1.0  # 恒定非零：保持老实现行为
            continue
        try:
            _, pv = stats.mannwhitneyu(x, y, alternative="two-sided")
        except ValueError:
            pv = 1.0
        pvals[j] = pv if np.isfinite(pv) else 1.0
    tested = np.isfinite(pvals)
    qvals = np.full(p, np.nan)
    reject = np.zeros(p, dtype=bool)
    if tested.any():
        pt = pvals[tested]
        qvals[tested] = multipletests(pt, method="fdr_bh")[1]
        reject[tested] = bh_reject(pt, ALPHA)
    pd.DataFrame(dict(taxon=taxa, p_value=pvals, q_value=qvals,
                      rejected=reject.astype(int),
                      filtered=(~tested))).to_csv(
        f"{BASE}/raw/tss_wilcoxon/cell{cell}_rep{rep}.csv", index=False)
    return int((~tested).sum())


def main():
    import os
    os.makedirs(f"{BASE}/raw/tss_wilcoxon", exist_ok=True)
    tot = 0
    for c in CELLS:
        nf = [run_one(c, r) for r in range(R_REPS)]
        tot += sum(nf)
        print(f"cell {c}: 20 reps, filtered mean {np.mean(nf):.2f}",
              flush=True)
    print("DONE", tot)


if __name__ == "__main__":
    main()
