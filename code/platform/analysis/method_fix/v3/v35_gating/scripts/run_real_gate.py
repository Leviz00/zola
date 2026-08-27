"""run_real_gate.py — 真实数据门控示意（ibdmdb / mbqc / agp）。

data-only：Y/depths/组标签（元数据）+ realdata 已有拟合汇总（φ̂/撞界）。
无 Ŵ 后验（realdata 管线未存），故 R1 用 **placeholder 臂**置换校准
（K=10，LRT，top-100 流行率类群，行子样本平衡两组，≤350/组）——这是门控
C2 在无估计权重时的保守近似。R2 用 placeholder n_tested_frac。
C1 直接取 fit_*_summary.csv（φ̂、撞界、SE、zero_fraction、n、depth）。
输出 real_gate.csv（每数据集一行 + gate 裁决）。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

SIM_V3 = "/mnt/agents/output/code/simulation_v3"
sys.path.insert(0, SIM_V3)
from abs_glm import abs_nb_glm  # noqa: E402

RD = "/mnt/agents/output/realdata"
K = 10
SEED = 20260305
OUT = "/mnt/agents/output/analysis/method_fix/v3/v35_gating/real_gate.csv"


def glm_perm(Y, group, N):
    rng = np.random.default_rng(SEED)
    p = Y.shape[1]
    fdr, rate, tested = [], [], []
    for _ in range(K):
        g = rng.permutation(group)
        r = abs_nb_glm(Y, g, N=N, W=None)
        fdr.append(float(r["reject"].sum() > 0))
        rate.append(r["reject"].sum() / p)
        tested.append(1 - r["n_fallback"] / p)
    r = abs_nb_glm(Y, group, N=N, W=None)
    return dict(perm_fdr=float(np.mean(fdr)), perm_rate=float(np.mean(rate)),
                tested_frac=float(np.mean(tested)),
                real_n_rej=int(r["reject"].sum()))


def prep(Y, depths, names, group_map, max_per_group=350, top=100):
    df = pd.DataFrame({"name": names})
    df["group"] = df["name"].map(group_map)
    kr = df.group.notna().values
    Y, N, df = Y[kr], depths[kr], df[kr].reset_index(drop=True)
    prev = (Y > 0).mean(axis=0)
    Y = Y[:, np.argsort(-prev)[:top]]
    rng = np.random.default_rng(SEED)
    idx = []
    for gv in sorted(df.group.unique()):
        ii = np.where(df.group.values == gv)[0]
        if len(ii) > max_per_group:
            ii = rng.choice(ii, max_per_group, replace=False)
        idx.append(ii)
    idx = np.concatenate(idx)
    return Y[idx].astype(float), df.group.values[idx].astype(int), N[idx]


def summarize_fit(name):
    r = pd.read_csv(f"{RD}/results/fit_{name}_summary.csv").iloc[0]
    return dict(phi_hat=float(r.phi_hat), phi_bnd=bool(r.phi_on_boundary),
                se_logphi=float(r.se_gamma_logphi),
                zero_fraction=float(r.zero_fraction), n=int(r.n),
                N_med=float(r.N_median))


def decide(row, perm_fdr_max=0.25, tested_min=0.5):
    """与 v3.4 固化规则同阈值；R3(φ̂ 撞下界) 在真实数据不出现，按规则列。"""
    reasons = []
    if row["perm_fdr"] > perm_fdr_max:
        reasons.append("R1_permFDR")
    if row["tested_frac"] < tested_min:
        reasons.append("R2_tested")
    if row["phi_hat"] <= 0.06:
        reasons.append("R3_phi_lower")
    return ("OFF:" + ";".join(reasons)) if reasons else "ON"


def main():
    out = []

    # ibdmdb: IBD (CD+UC) vs nonIBD
    z = np.load(f"{RD}/data/ibdmdb_genus.npz", allow_pickle=True)
    meta = pd.read_csv("/mnt/agents/output/datasets/ibdmdb/"
                       "ibdmdb_16S_sample_metadata.csv")
    gmap = {str(k): int(v) for k, v in
            (meta.assign(g=(meta.diagnosis != "nonIBD").astype(int))
             .set_index("sample_id").g.to_dict()).items()}
    Y, g, N = prep(z["Y"], z["depths"], z["samples"], gmap)
    d = glm_perm(Y, g, N)
    out.append(dict(dataset="ibdmdb", group_desc="IBD vs nonIBD",
                    n_used=len(g), **summarize_fit("ibdmdb"), **d))
    print("ibdmdb", d, "n=", len(g))

    # mbqc: 两大 handling lab（样本名第 2 字段）
    z = np.load(f"{RD}/data/mbqc_genus.npz", allow_pickle=True)
    names = z["samples"]
    labs = np.array([n.split(".")[1] for n in names])
    two = pd.Series(labs).value_counts().index[:2]
    gmap = {n: (0 if l == two[0] else 1) for n, l in zip(names, labs)
            if l in set(two)}
    Y, g, N = prep(z["Y"], z["depths"], names, gmap)
    d = glm_perm(Y, g, N)
    out.append(dict(dataset="mbqc", group_desc=f"lab {two[0]} vs {two[1]}",
                    n_used=len(g), **summarize_fit("mbqc"), **d))
    print("mbqc", d, "n=", len(g))

    # agp: 自述确诊 IBD vs 无
    z = np.load(f"{RD}/data/agp_genus.npz", allow_pickle=True)
    meta = pd.read_csv("/mnt/agents/output/datasets/agp/"
                       "agp_sample_metadata_9511fecal.csv")
    pos = meta.ibd.str.contains("Diagnosed|Self-diagnosed", na=False)
    neg = meta.ibd.str.contains("do not have", na=False)
    gmap = dict(zip(meta.sample_name[pos], [1] * pos.sum()))
    gmap.update(dict(zip(meta.sample_name[neg], [0] * neg.sum())))
    Y, g, N = prep(z["Y"], z["depths"], z["samples"], gmap)
    d = glm_perm(Y, g, N)
    out.append(dict(dataset="agp", group_desc="IBD(self/doctor) vs none",
                    n_used=len(g), **summarize_fit("agp"), **d))
    print("agp", d, "n=", len(g))

    df = pd.DataFrame(out)
    df["gate"] = df.apply(decide, axis=1)
    df.to_csv(OUT, index=False)
    print(df[["dataset", "phi_hat", "phi_bnd", "zero_fraction", "perm_fdr",
              "tested_frac", "real_n_rej", "gate"]].to_string(index=False))


if __name__ == "__main__":
    main()
