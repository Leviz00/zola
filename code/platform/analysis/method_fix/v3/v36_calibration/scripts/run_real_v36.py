"""run_real_v36.py — A2：三真实数据集的置换校准检验 + DA 清单。

置换设计（数据分层，2026-03 既定）：
  ibdmdb：178 样本 / 81 subjects（纵向）→ **subject 层聚类置换**（诊断是
          subject 级标签，聚类内保持）；
  mbqc   ：lab(4 vs 6) × 14 个 field-3 处理批次交叉 → **field-3 内分层置换**；
  agp    ：9511 样本 / 9511 hosts（无重复）→ 普通标签置换。
与 v3.5 门控示意同 prep（top-100 流行率类群；ibdmdb 全 178 行、mbqc/agp
平衡子样 ≤350/组）。输出 real_calibrated.csv（数据集级）+ 每数据集
real_da_<name>.csv（校准 q 值 DA 清单）。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

SIM_V3 = "/mnt/agents/output/code/simulation_v3"
V36 = "/mnt/agents/output/analysis/method_fix/v3/v36_calibration"
RD = "/mnt/agents/output/realdata"
sys.path.insert(0, V36)
sys.path.insert(1, SIM_V3)
from perm_glm import calibrated_test  # noqa: E402

K = 20
SEED = 20260305


def prep(Y, depths, names, taxa, group_map, max_per_group=350, top=100,
         group_arr=None):
    df = pd.DataFrame({"name": names})
    df["group"] = df["name"].map(group_map)
    kr = df.group.notna().values
    Y, N, df = Y[kr], depths[kr], df[kr].reset_index(drop=True)
    prev = (Y > 0).mean(axis=0)
    topj = np.argsort(-prev)[:top]
    Y, taxa = Y[:, topj], taxa[topj]
    rng = np.random.default_rng(SEED)
    idx = []
    for gv in sorted(df.group.unique()):
        ii = np.where(df.group.values == gv)[0]
        if len(ii) > max_per_group:
            ii = rng.choice(ii, max_per_group, replace=False)
        idx.append(ii)
    idx = np.concatenate(idx)
    return (Y[idx].astype(float), df.group.values[idx].astype(int), N[idx],
            taxa, df.iloc[idx])


def main():
    summaries = []

    # ---- ibdmdb：subject 聚类置换 -------------------------------------------
    z = np.load(f"{RD}/data/ibdmdb_genus.npz", allow_pickle=True)
    meta = pd.read_csv("/mnt/agents/output/datasets/ibdmdb/"
                       "ibdmdb_16S_sample_metadata.csv")
    meta["g"] = (meta.diagnosis != "nonIBD").astype(int)
    gmap = {str(k): int(v) for k, v in
            meta.set_index("sample_id").g.to_dict().items()}
    smap = {str(k): v for k, v in
            meta.set_index("sample_id").subject_id.to_dict().items()}
    Y, g, N, taxa, df = prep(z["Y"], z["depths"], z["samples"],
                             z["taxa"], gmap, max_per_group=10**9)
    subj = df.name.map(smap).values
    rng = np.random.default_rng(SEED)
    perms = []
    for _ in range(K):
        us = pd.unique(subj)
        g_of_subj = pd.Series([g[np.where(subj == s)[0][0]] for s in us],
                              index=us)
        perm_g = pd.Series(rng.permutation(g_of_subj.values), index=us)
        perms.append(pd.Series(subj).map(perm_g).values.astype(float))
    r = calibrated_test(Y, g, N, W=None, perms=perms)
    summaries.append(dict(dataset="ibdmdb", n=len(g), perm="subject-cluster",
                          fwer=r["fwer"], n_rej=int(r["reject"].sum()),
                          n_heavy=int(r["heavy"].sum())))
    pd.DataFrame(dict(taxon=taxa, b1=r["b1"], p=r["pvals"], q=r["qvals"],
                      reject=r["reject"])).sort_values("p").to_csv(
        f"{V36}/real_da_ibdmdb.csv", index=False)
    print("ibdmdb", summaries[-1])

    # ---- mbqc：field-3 内分层置换 --------------------------------------------
    z = np.load(f"{RD}/data/mbqc_genus.npz", allow_pickle=True)
    names = z["samples"]
    labs = np.array([n.split(".")[1] for n in names])
    f3 = np.array([n.split(".")[3] for n in names])
    two = pd.Series(labs).value_counts().index[:2]
    gmap = {n: (0 if l == two[0] else 1) for n, l in zip(names, labs)
            if l in set(two)}
    Y, g, N, taxa, df = prep(z["Y"], z["depths"], names, z["taxa"], gmap)
    strata = np.array([n.split(".")[3] for n in df.name])
    rng = np.random.default_rng(SEED)
    perms = []
    for _ in range(K):
        gp = g.copy()
        for s in np.unique(strata):
            m = strata == s
            gp[m] = rng.permutation(g[m])
        perms.append(gp.astype(float))
    r = calibrated_test(Y, g, N, W=None, perms=perms)
    summaries.append(dict(dataset="mbqc", n=len(g),
                          perm="stratified-within-field3",
                          fwer=r["fwer"], n_rej=int(r["reject"].sum()),
                          n_heavy=int(r["heavy"].sum())))
    pd.DataFrame(dict(taxon=taxa, b1=r["b1"], p=r["pvals"], q=r["qvals"],
                      reject=r["reject"])).sort_values("p").to_csv(
        f"{V36}/real_da_mbqc.csv", index=False)
    print("mbqc", summaries[-1])

    # ---- agp：普通标签置换 ----------------------------------------------------
    z = np.load(f"{RD}/data/agp_genus.npz", allow_pickle=True)
    meta = pd.read_csv("/mnt/agents/output/datasets/agp/"
                       "agp_sample_metadata_9511fecal.csv")
    pos = meta.ibd.str.contains("Diagnosed|Self-diagnosed", na=False)
    neg = meta.ibd.str.contains("do not have", na=False)
    gmap = dict(zip(meta.sample_name[pos], [1] * pos.sum()))
    gmap.update(dict(zip(meta.sample_name[neg], [0] * neg.sum())))
    Y, g, N, taxa, df = prep(z["Y"], z["depths"], z["samples"], z["taxa"], gmap)
    r = calibrated_test(Y, g, N, W=None, K=K, seed=SEED)
    summaries.append(dict(dataset="agp", n=len(g), perm="plain-label",
                          fwer=r["fwer"], n_rej=int(r["reject"].sum()),
                          n_heavy=int(r["heavy"].sum())))
    pd.DataFrame(dict(taxon=taxa, b1=r["b1"], p=r["pvals"], q=r["qvals"],
                      reject=r["reject"])).sort_values("p").to_csv(
        f"{V36}/real_da_agp.csv", index=False)
    print("agp", summaries[-1])

    pd.DataFrame(summaries).to_csv(f"{V36}/real_calibrated.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
