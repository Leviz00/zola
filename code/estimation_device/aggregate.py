"""aggregate.py — 真实微生物组数据解析与属级聚合（任务 1）。

输出统一格式 /mnt/agents/output/realdata/data/{name}_genus.npz：
    Y      : (n, p) int32  属级计数（样本 × 类群）
    depths : (n,) int64    逐样本测序深度
    taxa   : (p,) str      属名
    samples: (n,) str      样本 ID
并写漏斗表 results/funnel_{name}.csv。

决策（详见 REPORT.md）：
  - ibdmdb：HMP2Data IBD16S 982 OTU × 178 活检样本，taxonomy 带 Genus 列，
    直接聚合；无 mock/空白，全部样本入模。
  - mbqc  ：28,357 个 Greengenes OTU（行名 k__|..|g__|..|OTUid），流式聚合；
    specimen_type_collapsed ∈ {Fresh, Freeze-dried} 为生物样本子集（本次落模用），
    {blank, Fecal/Oral artificial colony, Robogut} 为 mock/空白子集（另存，留待校准），
    Unknown/NaN 剔除。
  - agp   ：deblur BIOM 实测含 observation/metadata/taxonomy（Greengenes 风格
    k__..g__），故按计划 A 做属级聚合（非流行率过滤备选）；BIOM 内 9,511
    样本全为 Stool（空白对照仅存于全量 mapping，不在计数表内），全部入模。
属名缺失（g__ 空/unclassified）时回退为 "unclassified_<family>"。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import h5py
from pathlib import Path

DATA = Path("/mnt/agents/output/datasets")
OUT = Path("/mnt/agents/output/realdata/data")
RES = Path("/mnt/agents/output/realdata/results")
OUT.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

P_MIN, P_MAX = 100, 500          # 目标维度窗口
PREV_GRID = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]


def genus_from_path(fields, prefix="g__", fam_prefix="f__"):
    """从 k__|..|g__ 风格字段列表提取属名；缺失回退 unclassified_<科>。"""
    gen = fam = ""
    for f in fields:
        f = str(f).strip()
        if f.startswith(prefix):
            gen = f[len(prefix):].strip()
        elif f.startswith(fam_prefix):
            fam = f[len(fam_prefix):].strip()
    if gen in ("", "unclassified", "Unclassified", "-"):
        return f"unclassified_{fam}" if fam else "unclassified_unknown"
    return gen


def pick_prev_threshold(prev, p_lo=P_MIN, p_hi=P_MAX):
    """在 PREV_GRID 上选使保留类群数落在 [p_lo, p_hi] 的阈值；偏向 ~250。"""
    best = None
    for t in PREV_GRID:
        k = int((prev >= t).sum())
        if p_lo <= k <= p_hi:
            if best is None or abs(k - 250) < abs(best[1] - 250):
                best = (t, k)
    if best is None:
        # 无阈值命中窗口：选最接近窗口的
        cand = [(t, int((prev >= t).sum())) for t in PREV_GRID]
        best = min(cand, key=lambda x: min(abs(x[1] - p_lo), abs(x[1] - p_hi)))
    return best


def finalize(name, Y, depths, taxa, samples, funnel, note=""):
    """深度过滤 + 流行率过滤 + 保存 npz + 漏斗表。"""
    funnel.append(("genus_aggregated", Y.shape[0], Y.shape[1], note))
    # 1) 深度 > 0
    keep_s = depths > 0
    funnel.append(("samples_depth_gt0", int(keep_s.sum()), Y.shape[1],
                   f"dropped {int((~keep_s).sum())} zero-depth samples"))
    Y = Y[keep_s]; depths = depths[keep_s]; samples = samples[keep_s]
    # 2) 全零类群
    keep_t = Y.sum(axis=0) > 0
    funnel.append(("taxa_totalcount_gt0", Y.shape[0], int(keep_t.sum()), ""))
    Y = Y[:, keep_t]; taxa = taxa[keep_t]
    # 3) 流行率过滤
    prev = (Y > 0).mean(axis=0)
    thr, p_keep = pick_prev_threshold(prev)
    keep = prev >= thr
    funnel.append((f"taxa_prevalence_ge_{thr}", Y.shape[0], int(keep.sum()),
                   f"threshold chosen from grid {PREV_GRID} to land p in "
                   f"[{P_MIN},{P_MAX}]"))
    Y = Y[:, keep]; taxa = taxa[keep]
    funnel.append(("final", Y.shape[0], Y.shape[1],
                   f"depth min/median/max = {depths.min()}/"
                   f"{int(np.median(depths))}/{depths.max()}"))
    np.savez_compressed(OUT / f"{name}_genus.npz", Y=Y.astype(np.int32),
                        depths=depths.astype(np.int64),
                        taxa=np.asarray(taxa, dtype=str),
                        samples=np.asarray(samples, dtype=str))
    pd.DataFrame(funnel, columns=["step", "n_samples", "n_taxa", "note"]
                 ).to_csv(RES / f"funnel_{name}.csv", index=False)
    print(f"[{name}] final n={Y.shape[0]} p={Y.shape[1]} "
          f"prev_thr={thr} depth[{depths.min()},{depths.max()}]")
    return Y, depths, taxa, samples


# ---------------------------------------------------------------------------
def agg_ibdmdb():
    c = pd.read_csv(DATA / "ibdmdb/ibdmdb_16S_otu_counts.csv", index_col=0)
    t = pd.read_csv(DATA / "ibdmdb/ibdmdb_16S_taxonomy.csv", index_col=0)
    funnel = [("raw", c.shape[1], c.shape[0], "178 biopsy samples, all used")]
    fam = t["Family"].fillna("").astype(str)
    gen = t["Genus"].fillna("").astype(str)
    keys = np.where(gen.str.strip().eq(""),
                    "unclassified_" + fam.str.strip(), gen.str.strip())
    gdf = c.groupby(keys).sum()                      # genus × samples
    Y = gdf.to_numpy(dtype=np.int64).T               # samples × genus
    depths = Y.sum(axis=1)
    finalize("ibdmdb", Y, depths,
             gdf.index.to_numpy(), c.columns.to_numpy(), funnel,
             "genus from IBD16S taxonomy Genus column")


def _mbqc_stream(col_sets):
    """流式读 mbqc 大表，按 col_sets dict {name: cols} 分别属级聚合。"""
    tax_rows = pd.read_csv(DATA / "mbqc/mbqc_otu_taxonomy_rows.txt",
                           header=None, names=["rowname"])
    rownames = tax_rows["rowname"].astype(str)
    gkeys = rownames.map(lambda s: genus_from_path(s.split("|"))).to_numpy()

    header = pd.read_csv(DATA / "mbqc/mbqc_merged_mothur_OTU.txt.gz",
                         sep="\t", nrows=0)
    all_cols = header.columns.tolist()
    want = {c: name for name, cols in col_sets.items() for c in cols
            if c in all_cols}
    usecols = [all_cols[0]] + sorted(want)
    # pandas 按文件顺序（而非 usecols 顺序）返回列：vals 的列序 = 文件顺序
    present = [c for c in all_cols[1:] if c in want]
    ppos = {c: k for k, c in enumerate(present)}
    acc = {name: {} for name in col_sets}            # genus -> np.array
    colpos = {name: [ppos[c] for c in cols if c in want]
              for name, cols in col_sets.items()}
    # 跳过头 78 行元数据（header 为第 0 行，元数据第 1–78 行，数据自第 79 行）
    skip = list(range(1, 79))
    reader = pd.read_csv(DATA / "mbqc/mbqc_merged_mothur_OTU.txt.gz",
                         sep="\t", skiprows=skip, usecols=usecols,
                         chunksize=1000, low_memory=False)
    row0 = 0
    for chunk in reader:
        rn = chunk.iloc[:, 0].astype(str).to_numpy()
        idx = pd.Index(rownames).get_indexer(rn)     # 对齐到 taxonomy 行序
        gk = gkeys[idx]
        vals = chunk.iloc[:, 1:].to_numpy(dtype=np.float64)
        df = pd.DataFrame(vals).groupby(gk)
        s = df.sum()
        for name, pos in colpos.items():
            sub = s.iloc[:, pos].to_numpy()
            a = acc[name]
            for i, g in enumerate(s.index):
                a[g] = a[g] + sub[i] if g in a else sub[i].copy()
        row0 += len(chunk)
        print(f"  mbqc chunk rows={row0}", flush=True)
    out = {}
    for name, a in acc.items():
        gdf = pd.DataFrame.from_dict(a, orient="index")
        cols = [c for c in col_sets[name] if c in want]
        gdf.columns = cols
        out[name] = gdf
    return out


def agg_mbqc():
    m = pd.read_csv(DATA / "mbqc/mbqc_sample_metadata.csv",
                    usecols=["Unnamed: 0", "specimen_type_collapsed"])
    m = m.rename(columns={"Unnamed: 0": "sample"})
    bio = m.loc[m["specimen_type_collapsed"].isin(
        ["Fresh", "Freeze-dried"]), "sample"].tolist()
    # 注意：blank 在 specimen_type_collapsed 中标注为 "other"
    # （specimen_type 原始列才是 "blank"，401 个，已核实一一对应）。
    mock = m.loc[m["specimen_type_collapsed"].isin(
        ["other", "Fecal artificial colony", "Oral artificial colony",
         "Robogut"]), "sample"].tolist()
    funnel = [("raw", 18365, 28357,
               "18,365 samples (9 BL x 14 HL labs), 28,357 Greengenes OTUs"),
              ("bio_subset_Fresh+FreezeDried", len(bio), 28357,
               "biological fecal specimens (model fitting uses this subset)"),
              ("mock_blank_subset", len(mock), 28357,
               "blank/Fecal-artificial/Oral-artificial/Robogut; saved for "
               "later calibration, NOT used in fitting"),
              ("dropped_Unknown_or_NaN", 18365 - len(bio) - len(mock), 28357,
               "specimen_type_collapsed Unknown/NaN excluded from both")]
    tabs = _mbqc_stream({"bio": bio, "mockblank": mock})
    # mock/空白子集另存（属级，未过滤），供后续校准分析
    gmock = tabs["mockblank"]
    np.savez_compressed(OUT / "mbqc_mockblank_genus.npz",
                        Y=gmock.T.to_numpy(dtype=np.int32),
                        taxa=gmock.index.to_numpy(),
                        samples=gmock.columns.to_numpy())
    gbio = tabs["bio"]
    Y = gbio.T.to_numpy(dtype=np.int64)
    depths = Y.sum(axis=1)
    finalize("mbqc", Y, depths, gbio.index.to_numpy(),
             gbio.columns.to_numpy(), funnel,
             "genus parsed from OTU row-name taxonomy path (g__ field)")


def agg_agp():
    f = h5py.File(DATA / "agp/deblur_125nt_no_blooms.biom", "r")
    tax = f["observation/metadata/taxonomy"][:]
    gkeys = np.array([genus_from_path(
        [x.decode() if isinstance(x, bytes) else str(x) for x in row])
        for row in tax])
    data = f["observation/matrix/data"][:].astype(np.int64)
    indices = f["observation/matrix/indices"][:]     # sample idx (CSR rows=obs)
    indptr = f["observation/matrix/indptr"][:]
    n_obs, n_smp = f.attrs["shape"]
    sample_ids = np.array([s.decode() for s in f["sample/ids"][:]])
    # 属级聚合：对每个 observation 行，累加到其属
    uniq, inv = np.unique(gkeys, return_inverse=True)
    Y = np.zeros((n_smp, len(uniq)), dtype=np.int64)
    for o in range(n_obs):
        sl = slice(indptr[o], indptr[o + 1])
        if sl.start < sl.stop:
            np.add.at(Y[:, inv[o]], indices[sl], data[sl])
        if o % 8000 == 0:
            print(f"  agp obs {o}/{n_obs}", flush=True)
    funnel = [("raw", n_smp, n_obs,
               "9,511 fecal samples (BIOM contains stool only; control blanks "
               "exist in full AGP mapping but not in this count table)")]
    depths = Y.sum(axis=1)
    # 与官方深度表交叉核对
    try:
        dref = pd.read_csv(DATA / "agp/agp_sample_depths.csv")
        dref = dref.set_index(dref.columns[0])["total_reads"]
        common = pd.Index(sample_ids).intersection(dref.index)
        agree = np.allclose(dref.loc[common].to_numpy(),
                            pd.Series(depths, index=sample_ids).loc[common])
        funnel.append(("depth_crosscheck_vs_agp_sample_depths_csv",
                       len(common), -1, f"match={agree}"))
    except Exception as e:  # noqa
        funnel.append(("depth_crosscheck", -1, -1, f"skipped: {e}"))
    finalize("agp", Y, depths, uniq, sample_ids, funnel,
             "genus from BIOM observation/metadata/taxonomy g__ field")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["ibdmdb", "mbqc", "agp"]
    for w in which:
        print(f"=== {w} ===", flush=True)
        {"ibdmdb": agg_ibdmdb, "mbqc": agg_mbqc, "agp": agg_agp}[w]()
