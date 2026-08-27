"""v3.0 探针：oracle 权重下假阳性的来源解剖（不需要估计，秒级）。
对每个探针格跑 R 个 rep，报告每个被拒绝的 FP 类群：
- 是否结构类群（真值结构零 >0 的类群）
- 其两组相对丰度中位数（组成挤压方向：病例组被压低？）
- 其 θ̄ 真值是否两组相同（绝对丰度定义下的非 DA）
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import design, generators  # noqa: E402
from weighting import oracle_weights, exclusion_wilcoxon  # noqa: E402
from baselines_py import tss_relative_abundance  # noqa: E402

cell_id, R = int(sys.argv[1]), int(sys.argv[2])
cfg = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "configs", "config_fractional.csv"))
row = cfg[cfg.cell_id == cell_id].iloc[0]
child = np.random.SeedSequence(int(row["seed"])).spawn(max(R, 20))
params = design.params_for_cell(row)

records = []
for r in range(R):
    Y, truth = generators.generate(row["mechanism"], params, n=int(row["n"]),
                                   p=100, depths=int(row["depth"]), seed=child[r])
    g = truth["group"]; da = truth["da_taxa"]
    sz = truth["structural_zeros"]
    W = oracle_weights(Y, truth)
    rej = exclusion_wilcoxon(Y, g, W)["reject"]
    rel = tss_relative_abundance(Y)
    fp = np.setdiff1d(np.where(rej)[0], da)
    for j in fp:
        struct_taxon = bool(sz[:, j].any())
        m1 = float(np.median(rel[g == 1, j])); m0 = float(np.median(rel[g == 0, j]))
        records.append(dict(cell_id=cell_id, rep=r, taxon=int(j),
                            struct_taxon=struct_taxon,
                            med_case=m1, med_ctrl=m0,
                            direction=np.sign(m1 - m0),
                            case_zero_frac=float((Y[g == 1, j] == 0).mean()),
                            ctrl_zero_frac=float((Y[g == 0, j] == 0).mean())))
df = pd.DataFrame(records)
out = f"/mnt/agents/output/analysis/method_fix/v3/probe_cell{cell_id}_oracle_fp.csv"
os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_csv(out, index=False)
print(f"cell {cell_id} ({row['mechanism']}, inf={row['informative_zeros']}, sz={row['structural_zero_rate']}, "
      f"effect={row['effect_size']}, depth={row['depth']}): {R} reps, FP 总数 {len(df)}")
if len(df):
    print(df.groupby('struct_taxon').agg(n=('taxon','count'),
          case_down=('direction', lambda s: (s<0).mean()),
          case_zf=('case_zero_frac','mean'), ctrl_zf=('ctrl_zero_frac','mean')).round(3).to_string())
