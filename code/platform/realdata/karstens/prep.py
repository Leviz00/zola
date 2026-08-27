"""prep.py — Karstens2019 稀释系列：数据整理（任务 A 第 1 步）。

输入：/mnt/agents/output/datasets/karstens2019/ 的 ASV 计数表 + taxonomy + metadata。
输出：karstens/data_karstens_genus.npz（Y/depths/taxa/samples/dilution/is_blank）
      results/prep_summary.csv（逐样本深度、检出属数、DNA 浓度）
      results/funnel_karstens.csv

聚合层级选择：**属级**。理由：
  (1) n=10 极小，ASV 级 1,414 个类群中大部分为稀疏检出，逐 ASV 检出指示
      噪声过大；属级聚合 pooling 同一属内多个 ASV 的检出信号，与 realdata
      主线（ibdmdb/mbqc/agp 均属级）口径一致，估计器假设（属级 θ̄、单一 φ）
      可直接沿用；
  (2) 稀释响应是"真实 mock 成员"的属性，mock 群落的已知成员全部在属级
      可分辨（Lactobacillus、Escherichia/Shigella 等），ASV 级分辨对本任务
      （检出曲线形状验证）无增量信息；
  (3) 代价：稀释末端低丰度属的检出被属内 pooling 抬高（属内任一 ASV 检出
      即属检出）——这是保守方向（偏向检出），在报告中声明。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("/mnt/agents/output/datasets/karstens2019")
OUT = Path("/mnt/agents/output/realdata/karstens")
RES = OUT / "results"
RES.mkdir(parents=True, exist_ok=True)


def main():
    counts = pd.read_csv(DATA / "karstens2019_asv_counts.csv", index_col=0)
    tax = pd.read_csv(DATA / "karstens2019_taxonomy.csv", index_col=0)
    meta = pd.read_csv(DATA / "karstens2019_metadata.csv", index_col=0)

    funnel = [("raw_asv", counts.shape[0], counts.shape[1],
               "10 samples x 1,414 ASV (9 mock dilutions + 1 blank)")]

    # 属级键：Genus 缺失回退 unclassified_<Family>（与 aggregate.py 同口径）
    fam = tax["Family"].fillna("").astype(str).str.strip()
    gen = tax["Genus"].fillna("").astype(str).str.strip()
    keys = np.where(gen.eq(""), "unclassified_" + fam, gen)
    gdf = counts.T.groupby(keys).sum().T          # samples × genus
    funnel.append(("genus_aggregated", gdf.shape[0], gdf.shape[1],
                   "genus from taxonomy Genus column, missing->unclassified_<Family>"))

    Y_all = gdf.to_numpy(dtype=np.int64)
    genera = gdf.columns.to_numpy()
    samples = gdf.index.to_numpy()

    # 与 metadata 对齐
    meta = meta.loc[samples]
    dilution_label = meta["SampleDescription"].to_numpy()
    is_blank = (meta["SampleType"] == "Blank").to_numpy()
    # 稀释倍数（名义 3 倍梯度）：D_k 对应 3^{-k}
    dil_level = np.array([int(s[1:]) if s.startswith("D") else -1
                          for s in meta.index])
    dil_factor = np.where(is_blank, np.nan, 3.0 ** (-dil_level))
    dna_conc = meta["DNA_conc"].to_numpy(dtype=float)

    depths = Y_all.sum(axis=1)
    # 过滤：总计数 >= 10 的属（剔除纯噪声单读数属；检出矩阵的实质信息不变）
    keep = Y_all.sum(axis=0) >= 10
    funnel.append(("genera_totalcount_ge10", Y_all.shape[0], int(keep.sum()),
                   f"dropped {int((~keep.sum()))} genera with <10 total reads"))
    Y = Y_all[:, keep]
    genera_kept = genera[keep]

    np.savez_compressed(
        OUT / "data_karstens_genus.npz",
        Y=Y.astype(np.int32), depths=depths.astype(np.int64),
        taxa=genera_kept.astype(str), samples=samples.astype(str),
        dil_factor=dil_factor, dna_conc=dna_conc, is_blank=is_blank,
        dilution_label=dilution_label.astype(str))

    # 逐样本摘要
    D = (Y > 0)
    det_frac = D[:, ].mean(axis=1)
    rows = []
    for i, s in enumerate(samples):
        rows.append({
            "sample": s, "label": dilution_label[i], "is_blank": bool(is_blank[i]),
            "depth": int(depths[i]), "dna_conc": dna_conc[i],
            "nominal_dilution": dil_factor[i],
            "n_genera_detected": int(D[i].sum()),
            "frac_genera_detected": float(det_frac[i]),
        })
    pd.DataFrame(rows).to_csv(RES / "prep_summary.csv", index=False)
    pd.DataFrame(funnel, columns=["step", "n_samples", "n_taxa", "note"]
                 ).to_csv(RES / "funnel_karstens.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"genera kept: {keep.sum()} / {len(genera)}")


if __name__ == "__main__":
    main()
