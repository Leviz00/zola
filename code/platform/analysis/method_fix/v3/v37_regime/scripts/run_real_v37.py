"""run_real_v37.py — M3a 权重臂补齐（解析 Ŵ，免重拟合）+ M3b spike-in 阳性对照。

M3a：W0 截距-only ⇒ π̂_ij=π̂_j；Ŵ_ij = P(结构零|Y=0)
  = (1−π̂_j) / ((1−π̂_j) + π̂_j · g(N_i; θ̂_j, φ̂))，g 用 estimation_v3
  model.log_g(N, θ̂_j, φ̂) 闭式（Y>0 处 Ŵ=0，阈值 0.5 剔除——与模拟口径
  逐位一致）。输入：fit_*_pertaxon.csv（π̂_j、θ̂_j）+ fit_*_summary.csv（φ̂）。
  对三数据集跑 est 臂校准置换检验，与 v3.6 placeholder 臂对照。

M3b：ibdmdb / mbqc spike-in——从 top-100 集选 K=15 个类群（高/中/低流行率
  各 5），case 组注入 ×2/×4（Y→round(Y×fold)，仅 >0 细胞受影响，缺席细胞
  保持零——与 absolute 语义一致；N 保持原始实测总库容作绝对锚，如实记录）。
  报告注入回收率（按 fold/流行率档）与非注入类群 FDP，两臂同跑。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

SIM_V3 = "/mnt/agents/output/code/simulation_v3"
EST_V3 = "/mnt/agents/output/code/estimation_v3"
V36 = "/mnt/agents/output/analysis/method_fix/v3/v36_calibration"
V37 = "/mnt/agents/output/analysis/method_fix/v3/v37_regime"
RD = "/mnt/agents/output/realdata"
sys.path.insert(0, EST_V3)
sys.path.insert(1, V36)
sys.path.insert(2, SIM_V3)
import model  # noqa: E402
from perm_glm import calibrated_test  # noqa: E402

K_PERM = 20
SEED = 20260305


def analytic_what(Y, depths, name):
    """解析 Ŵ（W0 截距-only 相干口径）。"""
    pt = pd.read_csv(f"{RD}/results/fit_{name}_pertaxon.csv")
    summ = pd.read_csv(f"{RD}/results/fit_{name}_summary.csv").iloc[0]
    phi = float(summ.phi_hat)
    z = np.load(f"{RD}/data/{name}_genus.npz", allow_pickle=True)
    taxa = z["taxa"]
    pt = pt.set_index("taxon").reindex(taxa)
    pi = pt.pi_hat.values.clip(1e-6, 1 - 1e-6)
    theta = pt.theta_hat.values.clip(1e-12, None)
    W = np.zeros(Y.shape, dtype=float)
    zero = Y == 0
    Ni = depths.astype(float)
    for j in range(Y.shape[1]):
        lg = model.log_g(Ni, theta[j], phi)
        g = np.exp(lg)
        P = (1 - pi[j]) / ((1 - pi[j]) + pi[j] * g)
        W[zero[:, j], j] = P[zero[:, j]]
    return W


def prep_with_W(Y, depths, names, taxa, W, group_map, max_per_group=350,
                top=100):
    df = pd.DataFrame({"name": names})
    df["group"] = df["name"].map(group_map)
    kr = df.group.notna().values
    Y, N, W, df = Y[kr], depths[kr], W[kr], df[kr].reset_index(drop=True)
    prev = (Y > 0).mean(axis=0)
    topj = np.argsort(-prev)[:top]
    Y, W, taxa = Y[:, topj], W[:, topj], taxa[topj]
    rng = np.random.default_rng(SEED)
    idx = []
    for gv in sorted(df.group.unique()):
        ii = np.where(df.group.values == gv)[0]
        if len(ii) > max_per_group:
            ii = rng.choice(ii, max_per_group, replace=False)
        idx.append(ii)
    idx = np.concatenate(idx)
    return (Y[idx].astype(float), df.group.values[idx].astype(int), N[idx],
            W[idx], taxa, df.iloc[idx])


def make_perms(name, g, df):
    """与 v3.6 相同的分层置换设计。"""
    rng = np.random.default_rng(SEED)
    perms = []
    if name == "ibdmdb":
        meta = pd.read_csv("/mnt/agents/output/datasets/ibdmdb/"
                           "ibdmdb_16S_sample_metadata.csv")
        smap = {str(k): v for k, v in
                meta.set_index("sample_id").subject_id.to_dict().items()}
        subj = df.name.map(smap).values
        for _ in range(K_PERM):
            us = pd.unique(subj)
            gs = pd.Series([g[np.where(subj == s)[0][0]] for s in us],
                           index=us)
            pg = pd.Series(rng.permutation(gs.values), index=us)
            perms.append(pd.Series(subj).map(pg).values.astype(float))
    elif name == "mbqc":
        strata = np.array([n.split(".")[3] for n in df.name])
        for _ in range(K_PERM):
            gp = g.copy()
            for s in np.unique(strata):
                m = strata == s
                gp[m] = rng.permutation(g[m])
            perms.append(gp.astype(float))
    else:
        for _ in range(K_PERM):
            perms.append(rng.permutation(g).astype(float))
    return perms


def run_arm(Y, g, N, keep, perms):
    return calibrated_test(Y, g, N, W=keep, perms=perms)


def load_ds(name):
    z = np.load(f"{RD}/data/{name}_genus.npz", allow_pickle=True)
    return z["Y"].astype(float), z["depths"].astype(float), z["samples"], \
        z["taxa"]


def group_map_ibdmdb():
    meta = pd.read_csv("/mnt/agents/output/datasets/ibdmdb/"
                       "ibdmdb_16S_sample_metadata.csv")
    return {str(k): int(v) for k, v in
            (meta.assign(g=(meta.diagnosis != "nonIBD").astype(int))
             .set_index("sample_id").g.to_dict()).items()}


def group_map_mbqc(names):
    labs = np.array([n.split(".")[1] for n in names])
    two = pd.Series(labs).value_counts().index[:2]
    return {n: (0 if l == two[0] else 1) for n, l in zip(names, labs)
            if l in set(two)}


def group_map_agp():
    meta = pd.read_csv("/mnt/agents/output/datasets/agp/"
                       "agp_sample_metadata_9511fecal.csv")
    pos = meta.ibd.str.contains("Diagnosed|Self-diagnosed", na=False)
    neg = meta.ibd.str.contains("do not have", na=False)
    gmap = dict(zip(meta.sample_name[pos], [1] * pos.sum()))
    gmap.update(dict(zip(meta.sample_name[neg], [0] * neg.sum())))
    return gmap


def main():
    out = []
    for name, gmap_fn, maxg in (("ibdmdb", group_map_ibdmdb, 10**9),
                                ("mbqc", "mbqc", 350),
                                ("agp", group_map_agp, 350)):
        Y, N, names, taxa = load_ds(name)
        gmap = group_map_mbqc(names) if gmap_fn == "mbqc" else gmap_fn()
        W = analytic_what(Y, N, name)
        Y2, g, N2, W2, taxa2, df = prep_with_W(Y, N, names, taxa, W, gmap,
                                              max_per_group=maxg)
        keep_est = ~((Y2 == 0) & (W2 >= 0.5))
        perms = make_perms(name, g, df)
        for arm, keep in (("est", keep_est.astype(float)), ("plac", None)):
            r = run_arm(Y2, g, N2, keep, perms)
            out.append(dict(dataset=name, arm=arm, n=len(g),
                            fwer=r["fwer"], n_rej=int(r["reject"].sum()),
                            n_heavy=int(r["heavy"].sum()),
                            masked_frac=float((~keep_est).mean())
                            if arm == "est" else 0.0))
            pd.DataFrame(dict(taxon=taxa2, b1=r["b1"], p=r["pvals"],
                              q=r["qvals"], reject=r["reject"])
                         ).sort_values("p").to_csv(
                f"{V37}/real_da_{name}_{arm}.csv", index=False)
            print(name, arm, out[-1], flush=True)
    pd.DataFrame(out).to_csv(f"{V37}/real_m3a.csv", index=False)


if __name__ == "__main__":
    main()
