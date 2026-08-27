"""export_exchange.py — v3_baselines 阶段 1：exchange 导出（SPEC §0/§1）。

确定性重生成 v3.4 的 8 格 × R=20 计数数据；生成调用与
v34_full/scripts/run_one_v34f.py 第 104–135 行逐位一致
（design.params_for_cell + effect_mode="absolute" + generators.generate，
code/simulation_v3 只读）。

硬校验：每 (cell,rep)，零点处 structural_zeros 标签向量与 depth 向量
必须与 v34_full/npz 的 labels/depth 逐元素相等；160 全过才算完成，
任一不等立即报错退出（不放行）。

产物：exchange/{counts,meta,truth}/{cell_id}_rep{rep}.csv + manifest。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

SIM_V3 = "/mnt/agents/output/code/simulation_v3"
CONFIG_CSV = os.path.join(SIM_V3, "configs", "config_supplementary.csv")
P_TAXA = 100
V34_NPZ = "/mnt/agents/output/analysis/method_fix/v3/v34_full/npz"
OUT = "/mnt/agents/output/analysis/v3_baselines/exchange"
CELLS = [1000, 1002, 1004, 1005, 1006, 1007, 1008, 1009]
R_REPS = 20

sys.path.insert(0, SIM_V3)
import design  # noqa: E402
import generators  # noqa: E402


def export_one(cfg, cell, rep):
    row = cfg[cfg.cell_id == cell].iloc[0]
    seeds = np.random.SeedSequence(int(row["seed"])).spawn(R_REPS)
    prm = design.params_for_cell(row)
    prm["effect_mode"] = "absolute"
    Y, truth = generators.generate(
        row["mechanism"], prm, n=int(row["n"]), p=P_TAXA,
        depths=int(row["depth"]), seed=seeds[rep])
    Y = np.asarray(Y, dtype=np.int64)
    N = truth["depths"].astype(float)
    sz = np.asarray(truth["structural_zeros"], dtype=bool)
    zero = Y == 0

    # ---- 硬校验（SPEC §1）：与 npz 逐元素相等 --------------------------
    z = np.load(f"{V34_NPZ}/cell{cell}_rep{rep}.npz")
    lab_reg = sz[zero].astype(int)
    dep_reg = np.broadcast_to(N[:, None], Y.shape)[zero]
    if not (len(lab_reg) == len(z["labels"])
            and np.array_equal(lab_reg, z["labels"])):
        raise SystemExit(
            f"HARD-FAIL labels cell{cell} rep{rep}: "
            f"len {len(lab_reg)} vs {len(z['labels'])}")
    if not (len(dep_reg) == len(z["depth"])
            and np.allclose(dep_reg, z["depth"], rtol=0, atol=0)):
        raise SystemExit(f"HARD-FAIL depth cell{cell} rep{rep}")

    n, p = Y.shape
    counts = pd.DataFrame(Y, columns=[f"taxon_{j+1}" for j in range(p)])
    counts.index = [f"sample_{i+1}" for i in range(n)]
    counts.index.name = "sample"
    counts.to_csv(f"{OUT}/counts/cell{cell}_rep{rep}.csv")
    pd.DataFrame(dict(sample=[f"sample_{i+1}" for i in range(n)],
                      group=np.asarray(truth["group"], dtype=int),
                      depth=N.astype(int))).to_csv(
        f"{OUT}/meta/cell{cell}_rep{rep}.csv", index=False)
    pd.DataFrame(dict(taxon=[f"taxon_{j+1}" for j in range(p)],
                      abs_da_truth=np.asarray(truth["abs_da_truth"],
                                              dtype=int))).to_csv(
        f"{OUT}/truth/cell{cell}_rep{rep}.csv", index=False)
    return dict(cell_id=cell, rep=rep, n=n, p=p,
                zero_rate=float(zero.mean()),
                struct_frac=float(z["struct_frac"]), verified=True)


def main():
    cfg = pd.read_csv(CONFIG_CSV)
    man = []
    for cell in CELLS:
        for rep in range(R_REPS):
            man.append(export_one(cfg, cell, rep))
        print(f"cell {cell}: 20/20 verified", flush=True)
    df = pd.DataFrame(man)
    df.to_csv("/mnt/agents/output/analysis/v3_baselines/"
              "export_manifest.csv", index=False)
    assert len(df) == 160 and df.verified.all()
    print(f"EXPORT COMPLETE: {len(df)} files all verified")


if __name__ == "__main__":
    main()
