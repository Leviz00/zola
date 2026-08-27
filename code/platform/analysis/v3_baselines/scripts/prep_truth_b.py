"""prep_truth_b.py — Amendment 2：cell 1009 真值 B 构造（逐 rep）。

主口径：逐 rep 种子重放（与 run_one_v34f 第 104-135 行一致）取
truth["designated_structural"]，缺席 taxon 集合 = 任一样本被指定的列；
断言 informative 方向（缺席集中在 case）。
交叉验证（脚本注释要求的反推规则）：用 counts+meta 反推
"该 taxon 在全部 case 样本为 0 且在 control 不全为 0"，与重放集合
逐 rep 比对一致才放行。
输出：exchange/truth/cell1009_rep{r}_truthB.csv
      （taxon, abs_da_truth, absent_truth, truth_b）。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

SIM = "/mnt/agents/output/code/simulation_v3"
BASE = "/mnt/agents/output/analysis/v3_baselines"
sys.path.insert(0, SIM)
import design  # noqa: E402
import generators  # noqa: E402

CFG = pd.read_csv(f"{SIM}/configs/config_supplementary.csv")


def main():
    row = CFG[CFG.cell_id == 1009].iloc[0]
    seeds = np.random.SeedSequence(int(row["seed"])).spawn(20)
    prm = design.params_for_cell(row)
    prm["effect_mode"] = "absolute"
    for rep in range(20):
        Y, truth = generators.generate("zinb", prm, n=int(row["n"]), p=100,
                                       depths=int(row["depth"]),
                                       seed=seeds[rep])
        Y = np.asarray(Y, float)
        group = truth["group"].astype(int)
        ds = np.asarray(truth["designated_structural"], dtype=bool)
        absent = ds.any(axis=0)
        # informative 方向断言：指定缺席集中在 case（group==1）
        assert ds[group == 0].sum() == 0, f"rep{rep}: 缺席出现在 control"
        assert absent.sum() == 30, f"rep{rep}: 缺席 taxon 数 {absent.sum()}"
        # 反推规则交叉验证：全 case 为 0 且 control 不全为 0
        infer = (Y[group == 1].sum(axis=0) == 0) & (Y[group == 0].sum(axis=0) > 0)
        assert np.array_equal(infer, absent), \
            f"rep{rep}: 反推集合 {infer.sum()} != 重放集合 {absent.sum()}"
        da = np.asarray(truth["abs_da_truth"], dtype=bool)
        tb = da | absent
        pd.DataFrame(dict(
            taxon=[f"taxon_{j+1}" for j in range(100)],
            abs_da_truth=da.astype(int), absent_truth=absent.astype(int),
            truth_b=tb.astype(int))).to_csv(
            f"{BASE}/exchange/truth/cell1009_rep{rep}_truthB.csv", index=False)
    print("truthB 20/20 written, replay==inferred for all reps")


if __name__ == "__main__":
    main()
