"""realdata_recompute.py — 用校准后的判据重算三数据集可识别类群比例。

读取 realdata/results/fit_{name}_pertaxon.csv（最终管线状态）与
data/{name}_genus.npz（深度），逐类群计算 I_j（剖面 Fisher 信息，(c)）与
稳健 e_j 变体（(b1) N_q10、(b2) 均值），按校准阈值分类，与现行 SE 判据
（74.5%/46.6%/83.5%）对比。

注意：ibdmdb/agp 的 φ̂ 撞 1e5 上界；q 随 φ 单增 ⇒ I_j(φ=1e5) 是真实信息的
保守下界（真实 φ 更大只会增加检出信息）。

输出：results/realdata_pertaxon_scored.csv, results/realdata_proportions.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/agents/output/analysis/ej_criterion")
import criteria  # noqa: E402

RES = Path("/mnt/agents/output/realdata/results")
DATA = Path("/mnt/agents/output/realdata/data")
OUT = Path(__file__).parent / "results"

PHI_HAT = {"ibdmdb": 1e5, "mbqc": 1453.585027, "agp": 1e5}
SE_CUT = 1.0  # 现行判据 (a)


def fisher_vec(pi, theta, phi, depths):
    """向量化逐类群 I_j：pi, theta 为 (p,)，depths (n,)。返回 (p,) I_j。"""
    from model import log_g, dlogg_dtheta
    N = np.asarray(depths, float)[:, None]          # (n,1)
    th = np.asarray(theta, float)[None, :]          # (1,p)
    pv = np.asarray(pi, float)[None, :]
    lg = log_g(N, th, phi)
    g = np.exp(lg)
    q = np.clip(pv * (1.0 - g), 1e-12, 1 - 1e-12)
    dqa = (1.0 - g) * pv * (1.0 - pv)
    dqb = -pv * g * dlogg_dtheta(N, th, phi) * th
    w = 1.0 / (q * (1.0 - q))
    Aaa = (w * dqa * dqa).sum(0)
    Aab = (w * dqa * dqb).sum(0)
    Abb = (w * dqb * dqb).sum(0)
    return Aaa - Aab * Aab / np.maximum(Abb, 1e-300), Aab / np.sqrt(
        np.maximum(Aaa * Abb, 1e-300))


def run(name, t_info=1.0, t_eq10=None, t_emean=None):
    df = pd.read_csv(RES / f"fit_{name}_pertaxon.csv")
    dep = np.load(DATA / f"{name}_genus.npz", allow_pickle=True)["depths"]
    phi = PHI_HAT[name]
    df["on_boundary"] = df["on_boundary_pi"] | df["on_boundary_theta"]
    I_j, rho_ab = fisher_vec(df["pi_hat"].to_numpy(),
                             df["theta_hat"].to_numpy(), phi, dep)
    df["I_j"] = I_j
    df["rho_ab_fisher"] = rho_ab
    df["e_j_q10"] = [criteria.ej_quantile(t, phi, dep, 0.10)
                     for t in df["theta_hat"]]
    df["e_j_mean"] = [criteria.ej_mean(t, phi, dep) for t in df["theta_hat"]]
    df["n_e_j_mean"] = len(dep) * df["e_j_mean"]
    # 判据分类
    df["ident_a_se"] = (df["se_logit_pi"] < SE_CUT) & ~df["on_boundary"]
    df["ident_c_info"] = (df["I_j"] >= t_info) & ~df["on_boundary"]
    if t_eq10 is not None:
        df["ident_b1_eq10"] = df["e_j_q10"] >= t_eq10
    if t_emean is not None:
        df["ident_b2_emean"] = df["e_j_mean"] >= t_emean
    df["dataset"] = name
    return df


def main(t_info=1.0, t_eq10=None, t_emean=None):
    all_df = pd.concat([run(n, t_info, t_eq10, t_emean)
                        for n in ("ibdmdb", "mbqc", "agp")],
                       ignore_index=True)
    all_df.to_csv(OUT / "realdata_pertaxon_scored.csv", index=False)
    agg = {"dataset": [], "p": []}
    cols = [c for c in all_df.columns if c.startswith("ident_")]
    for c in cols:
        agg[c] = []
    for name, sub in all_df.groupby("dataset"):
        agg["dataset"].append(name)
        agg["p"].append(len(sub))
        for c in cols:
            agg[c].append(float(sub[c].mean()))
    prop = pd.DataFrame(agg)
    prop.to_csv(OUT / "realdata_proportions.csv", index=False)
    print(prop.to_string(index=False))
    # 一致率：判据 (c) vs 判据 (a)
    for name, sub in all_df.groupby("dataset"):
        agree = (sub["ident_a_se"] == sub["ident_c_info"]).mean()
        print(f"[{name}] (c) 与 (a) 逐类群一致率 {agree:.3f}; "
              f"(a) {sub['ident_a_se'].sum()} 个, (c) {sub['ident_c_info'].sum()} 个")


if __name__ == "__main__":
    main()
