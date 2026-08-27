"""validate_cov.py — 里程碑 1 验证：存在层协变量的模拟自校准（Exp 1 同规格扩展）。

设计（与 validate.py Exp 1 对齐：n=200、φ=3000、深度 [1e3,1e5] 跨 3φ、
θ̄∈[7e-4,1.05e-3]、加 π≈1 bulk 类群压 E1）：
  W_i = (1, group_i, cont_i)，group 平衡二值，cont 标准化连续（与 N 独立）；
  logit π_ij = γ_j'W_i，组间效应 γ_group：类群 0–3 为 0.8、4–7 为 0；
  连续效应 γ_cont：类群 0,1,4,5 为 0.5、2,3,6,7 为 0；
  截距取 logit(PI_CAL_j) − 0.5·γ_group_j（组平均 π 回到 Exp 1 的 π 网格）。

检验（结果写入 results/cov_*.csv，全部数字来自实际运行）：
  1. γ̂ 逐系数偏差、sd、Godambe 平均 SE、cov95（自然尺度 ±1.96·SE）；
  2. 存在性关联 Wald 检验：逐 rep 对 16 个斜率做 BH(0.05)，FDR = 逐 rep
     FDP 均值；功效 = 非零系数的 BH 拒绝率；另报边际 z 检验的逐零假设
     type-I 率；
  3. φ 已知 profile 臂 + φ 未知联合臂。

运行：python3 validate_cov.py --R 300 --cores 2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import norm

import model_cov
import composite_likelihood_cov as clc

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

PHI_CAL = 3000.0
PI_CAL = np.array([0.85, 0.87, 0.89, 0.91, 0.93, 0.94, 0.95, 0.95])
TH_CAL = np.array([7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5]) * 1e-4
G_GROUP = np.array([1.2] * 4 + [0.0] * 4)
G_CONT = np.array([0.6, 0.6, 0.0, 0.0, 0.6, 0.6, 0.0, 0.0])
N_SAMP = 200
P = 8
Z95 = 1.959964
GAMMA_TRUE = np.column_stack([logit(PI_CAL) - 0.5 * G_GROUP,
                              G_GROUP, G_CONT])       # (8,3)


def _design(rng, n=N_SAMP):
    W = np.column_stack([np.ones(n),
                         rng.integers(0, 2, n),
                         rng.standard_normal(n)])
    N = np.exp(rng.uniform(np.log(1e3), np.log(1e5), n)).astype(int)
    return W, N


def _sim(rng):
    W, N = _design(rng)
    theta_bar = np.concatenate([TH_CAL, [1.0 - TH_CAL.sum()]])
    G_full = np.vstack([GAMMA_TRUE, [20.0, 0.0, 0.0]])   # bulk π≈1
    Y, Z = model_cov.simulate_three_layer_cov(G_full, W, theta_bar,
                                              PHI_CAL, N, rng)
    return clc.detection_indicators(Y)[:, :P], W, N


def _bh_reject(pvals, q=0.05):
    """Benjamini–Hochberg：返回拒绝掩码。"""
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    k = np.where(ranked <= q * (np.arange(m) + 1) / m)[0]
    mask = np.zeros(m, dtype=bool)
    if k.size:
        mask[order[:k.max() + 1]] = True
    return mask


def _rep(seed):
    rng = np.random.default_rng(seed)
    D, W, N = _sim(rng)
    row = {"seed": seed}
    for arm, ph in [("k", PHI_CAL), ("j", None)]:
        f = clc.fit_composite_cov(D, W, N, phi_known=ph)
        row[f"succ_{arm}"] = f["success"]
        for j in range(P):
            for c in range(3):
                row[f"g_{arm}_{j}_{c}"] = f["Gamma"][j, c]
                row[f"sg_{arm}_{j}_{c}"] = f["se_Gamma"][j, c]
            row[f"lt_{arm}_{j}"] = np.log(f["theta"][j])
            row[f"sb_{arm}_{j}"] = f["se_beta"][j]
        if ph is None:
            row["phi_j"] = f["phi"]
            row["sg_phi_j"] = f["se_gamma_phi"]
    return row


def _summarize(df, R):
    rows = []
    for arm, tag in [("k", "phi_known"), ("j", "joint")]:
        succ = float(df[f"succ_{arm}"].mean())
        for j in range(P):
            for c in range(3):
                gh = df[f"g_{arm}_{j}_{c}"].values
                sg = df[f"sg_{arm}_{j}_{c}"].values
                gt = GAMMA_TRUE[j, c]
                z = (gh - gt) / np.maximum(sg, 1e-300)
                rows.append({
                    "arm": tag, "taxon": j,
                    "coef": ["intercept", "group", "cont"][c],
                    "gamma_true": gt,
                    "gamma_bias": float(gh.mean() - gt),
                    "gamma_median_bias": float(np.median(gh) - gt),
                    "sd_gamma": float(gh.std(ddof=1)),
                    "iqr_gamma": float(np.quantile(gh, 0.75)
                                       - np.quantile(gh, 0.25)),
                    "mean_se": float(sg.mean()),
                    "median_se": float(np.median(sg)),
                    "frac_se_gt5": float((sg > 5).mean()),
                    "cov95": float((np.abs(z) <= Z95).mean()),
                    "mean_z": float(z.mean()),
                    "type1_marginal_05": float((2 * norm.sf(np.abs(
                        gh / np.maximum(sg, 1e-300))) < 0.05).mean())
                    if c > 0 and gt == 0.0 else np.nan,
                    "success_rate": succ,
                })
            lt = df[f"lt_{arm}_{j}"].values
            sb = df[f"sb_{arm}_{j}"].values
            rows.append({
                "arm": tag, "taxon": j, "coef": "log_theta",
                "gamma_true": np.log(TH_CAL[j]),
                "gamma_bias": float(lt.mean() - np.log(TH_CAL[j])),
                "gamma_median_bias": float(np.median(lt) - np.log(TH_CAL[j])),
                "sd_gamma": float(lt.std(ddof=1)),
                "iqr_gamma": np.nan,
                "mean_se": float(sb.mean()),
                "median_se": float(np.median(sb)),
                "frac_se_gt5": float((sb > 5).mean()),
                "cov95": float((np.abs((lt - np.log(TH_CAL[j]))
                                       / np.maximum(sb, 1e-300)) <= Z95).mean()),
                "mean_z": np.nan, "type1_marginal_05": np.nan,
                "success_rate": succ,
            })
    # 联合臂 φ
    lg = np.log(df["phi_j"].values)
    rows.append({
        "arm": "joint", "taxon": "phi", "coef": "log_phi",
        "gamma_true": np.log(PHI_CAL),
        "gamma_bias": float(lg.mean() - np.log(PHI_CAL)),
        "sd_gamma": float(lg.std(ddof=1)),
        "mean_se": float(df["sg_phi_j"].mean()),
        "cov95": float((np.abs((lg - np.log(PHI_CAL))
                               / df["sg_phi_j"].values) <= Z95).mean()),
        "mean_z": np.nan, "type1_marginal_05": np.nan,
        "success_rate": float(df["succ_j"].mean()),
    })
    return pd.DataFrame(rows)


def _summarize_wald(df):
    """逐 rep BH(0.05)（16 个斜率）：FDR、逐系数功效。"""
    is_null = np.array([GAMMA_TRUE[j, c] == 0.0
                        for j in range(P) for c in (1, 2)])
    fdps, rej_rates = [], np.zeros(16)
    for _, r in df.iterrows():
        pv = np.array([2 * norm.sf(abs(r[f"g_k_{j}_{c}"]
                                       / max(r[f"sg_k_{j}_{c}"], 1e-300)))
                       for j in range(P) for c in (1, 2)])
        rej = _bh_reject(pv, 0.05)
        if rej.any():
            fdps.append(float((rej & is_null).sum() / rej.sum()))
        else:
            fdps.append(0.0)
        rej_rates += rej
    rej_rates /= len(df)
    rows = [{"scope": "overall", "coef": "all",
             "fdr_bh05": float(np.mean(fdps)),
             "power_bh05_nonnull": float(rej_rates[~is_null].mean()),
             "rejection_null_mean": float(rej_rates[is_null].mean())}]
    for j in range(P):
        for c, cn in [(1, "group"), (2, "cont")]:
            idx = j * 2 + (c - 1)
            rows.append({"scope": f"taxon_{j}", "coef": cn,
                         "gamma_true": GAMMA_TRUE[j, c],
                         "power_or_rejrate_bh05": float(rej_rates[idx]),
                         "is_null": bool(is_null[idx])})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=300)
    ap.add_argument("--cores", type=int, default=2)
    args = ap.parse_args()
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(args.cores) as pool:
        rows = pool.map(_rep, range(50_000, 50_000 + args.R), chunksize=4)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cov_calibration_perrep.csv", index=False)
    summ = _summarize(df, args.R)
    summ.to_csv(OUT / "cov_calibration_summary.csv", index=False)
    wald = _summarize_wald(df)
    wald.to_csv(OUT / "cov_wald_summary.csv", index=False)

    print("== 里程碑 1：存在层协变量自校准（R=%d, n=%d, φ=%.0f, %.0fs）=="
          % (args.R, N_SAMP, PHI_CAL, time.time() - t0))
    for tag in ["phi_known", "joint"]:
        s = summ[(summ["arm"] == tag)
                 & (summ["coef"].isin(["group", "cont"]))]
        print("  [%s] 斜率偏差 [%.3f, %.3f]（中位 [%.3f, %.3f]），"
              "sd [%.3f, %.3f]，median_se [%.3f, %.3f]，"
              "frac(se>5) [%.3f, %.3f]，cov95 [%.3f, %.3f]"
              % (tag, s["gamma_bias"].min(), s["gamma_bias"].max(),
                 s["gamma_median_bias"].min(), s["gamma_median_bias"].max(),
                 s["sd_gamma"].min(), s["sd_gamma"].max(),
                 s["median_se"].min(), s["median_se"].max(),
                 s["frac_se_gt5"].min(), s["frac_se_gt5"].max(),
                 s["cov95"].min(), s["cov95"].max()))
        si = summ[(summ["arm"] == tag) & (summ["coef"] == "intercept")]
        print("  [%s] 截距偏差 [%.3f, %.3f]，cov95 [%.3f, %.3f]"
              % (tag, si["gamma_bias"].min(), si["gamma_bias"].max(),
                 si["cov95"].min(), si["cov95"].max()))
    t1 = summ[(summ["coef"].isin(["group", "cont"]))
              & summ["type1_marginal_05"].notna()]
    print("  边际 z 检验 type-I(0.05) 范围 [%.3f, %.3f]"
          % (t1["type1_marginal_05"].min(), t1["type1_marginal_05"].max()))
    print("  BH(0.05)：FDR=%.3f，非零系数平均功效=%.3f，零系数平均拒绝率=%.3f"
          % (wald.loc[0, "fdr_bh05"], wald.loc[0, "power_bh05_nonnull"],
             wald.loc[0, "rejection_null_mean"]))
    prow = summ[summ["coef"] == "log_phi"].iloc[0]
    print("  [joint] log φ̂ 偏差 %.3f，sd %.2f，cov95 %.3f"
          % (prow["gamma_bias"], prow["sd_gamma"], prow["cov95"]))


if __name__ == "__main__":
    main()
