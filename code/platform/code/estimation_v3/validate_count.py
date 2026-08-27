"""validate_count.py — 里程碑 2 验证：计数幅度 DM 块复合似然 + 逐单元后验。

实验（结果写入 results/count_*.csv、posterior_*.csv、block_complexity.csv，
全部数字来自实际运行）：

  C1 主校准（Exp 1 同规格：n=200、φ=3000、深度 [1e3,1e5]、θ̄ 同 Exp 1、
     π≈1 bulk）：每 rep 拟合 b∈{1,4,8}（φ 已知 profile）+ b=4（φ 未知
     联合）。b=1 即检出指示 CL（magnitude≡0 严格退化），与 b≥2 同数据
     对照：θ̂ 精度（sd(log θ̂)）、π–θ 脊相关 corr(log π̂, log θ̂)、覆盖率。
  C2 脊区对照（Exp 2 同规格：φ=10 已知、θ̄=1e-3、e_j=0.047）：b=1 vs
     b=4，检验计数幅度是否部分断开脊（sd(log θ̂)、边界率、相关）。
  C3 逐单元零来源后验（eq:posterior 逐单元近似）：用 C1 b=4 拟合的
     π̂,θ̂,φ̂ 计算所有零单元的 Pr(Z=0|Y=0)，对照模拟真值 Z：逐 rep AUC、
     池化可靠性分箱（预测均值 vs 经验结构零比例）、总体零构成校准
     （平均后验 vs 实际结构零占比）。
  C4 复杂度实测：magnitude 评估耗时随块大小 b 与 p 的标度
     （对照 fix_N4 的 O(np)/O(np·b) 声明），给出默认 b 建议。

运行：python3 validate_count.py --R 200 --cores 2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import model
import composite_likelihood as cl
import composite_likelihood_count as clc
import posterior as post

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

Z95 = 1.959964

# 主校准（与 Exp 1 对齐）
PHI_CAL = 3000.0
PI_CAL = np.array([0.85, 0.87, 0.89, 0.91, 0.93, 0.94, 0.95, 0.95])
TH_CAL = np.array([7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5]) * 1e-4
N_SAMP = 200
P = 8

# 脊区（与 Exp 2 对齐）
PHI_RIDGE = 10.0
TH_RIDGE = 0.001
PI_RIDGE = np.array([0.85, 0.90, 0.95, 0.97])


def _depths(rng, n=N_SAMP, lo=1e3, hi=1e5):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n)).astype(int)


def _sim_main(rng):
    theta_bar = np.concatenate([TH_CAL, [1.0 - TH_CAL.sum()]])
    pi_all = np.concatenate([PI_CAL, [1.0]])
    N = _depths(rng)
    Y, Z = model.simulate_three_layer(pi_all, theta_bar, PHI_CAL, N, rng)
    return Y[:, :P], Z[:, :P], N


def _sim_ridge(rng):
    p = len(PI_RIDGE)
    theta_bar = np.concatenate([np.full(p, TH_RIDGE), [1.0 - p * TH_RIDGE]])
    pi_all = np.concatenate([PI_RIDGE, [1.0]])
    N = _depths(rng)
    Y, Z = model.simulate_three_layer(pi_all, theta_bar, PHI_RIDGE, N, rng)
    return Y[:, :p], Z[:, :p], N


# ---------------------------------------------------------------------------
# C1 主校准 + C3 后验（同一批拟合内完成）
# ---------------------------------------------------------------------------

def _rep_main(seed):
    rng = np.random.default_rng(seed)
    Y, Z, N = _sim_main(rng)
    row = {"seed": seed}
    fits = {}
    for tag, b, ph in [("b1k", 1, PHI_CAL), ("b4k", 4, PHI_CAL),
                       ("b8k", 8, PHI_CAL), ("b4j", 4, None)]:
        f = clc.fit_count_composite(Y, N, b=b, phi_known=ph)
        fits[tag] = f
        row[f"succ_{tag}"] = f["success"]
        for j in range(P):
            row[f"la_{tag}_{j}"] = np.log(f["pi"][j]
                                          / (1 - f["pi"][j]))
            row[f"lt_{tag}_{j}"] = np.log(f["theta"][j])
            row[f"sa_{tag}_{j}"] = f["se_alpha"][j]
            row[f"sb_{tag}_{j}"] = f["se_beta"][j]
        if ph is None:
            row["lphi_b4j"] = np.log(f["phi"])
            row["sgphi_b4j"] = f["se_gamma_phi"]
    # C3：后验（b=4 φ 已知拟合）
    f = fits["b4k"]
    pm = post.zero_source_posterior(f["pi"], f["theta"], f["phi"], N)
    zero = Y == 0
    labels = (Z == 0)[zero].astype(float)
    scores = pm[zero]
    row["post_auc"] = post.auc_score(labels, scores)
    row["post_mean"] = float(scores.mean())
    row["struct_frac"] = float(labels.mean())
    row["n_zero"] = int(zero.sum())
    # 池化校准分箱用：保存分箱均值×计数（10 箱等宽）
    mp, fp, cnt = post.calibration_bins(labels, scores, n_bins=10)
    for k in range(10):
        row[f"bin{k}_predsum"] = (mp[k] * cnt[k]) if cnt[k] else 0.0
        row[f"bin{k}_possum"] = (fp[k] * cnt[k]) if cnt[k] else 0.0
        row[f"bin{k}_cnt"] = int(cnt[k])
    return row


def _summarize_main(df):
    rows = []
    la_true = np.log(PI_CAL / (1 - PI_CAL))
    lt_true = np.log(TH_CAL)
    for tag in ["b1k", "b4k", "b8k", "b4j"]:
        succ = float(df[f"succ_{tag}"].mean())
        for j in range(P):
            la = df[f"la_{tag}_{j}"].values
            lt = df[f"lt_{tag}_{j}"].values
            sa = df[f"sa_{tag}_{j}"].values
            sb = df[f"sb_{tag}_{j}"].values
            rows.append({
                "arm": tag, "taxon": j,
                "logit_pi_bias": float(la.mean() - la_true[j]),
                "logit_pi_median_bias": float(np.median(la) - la_true[j]),
                "sd_logit_pi": float(la.std(ddof=1)),
                "median_se_alpha": float(np.median(sa)),
                "cov95_logit_pi": float((np.abs((la - la_true[j])
                                        / np.maximum(sa, 1e-300)) <= Z95).mean()),
                "log_theta_bias": float(lt.mean() - lt_true[j]),
                "log_theta_median_bias": float(np.median(lt) - lt_true[j]),
                "sd_log_theta": float(lt.std(ddof=1)),
                "median_se_beta": float(np.median(sb)),
                "cov95_log_theta": float((np.abs((lt - lt_true[j])
                                         / np.maximum(sb, 1e-300)) <= Z95).mean()),
                "corr_logpi_logtheta": float(np.corrcoef(la, lt)[0, 1]),
                "frac_theta_boundary": float((np.exp(lt) > 0.8).mean()),
                "frac_pi_boundary": float((np.abs(la) > 9.1).mean()),
                "success_rate": succ,
            })
    lg = df["lphi_b4j"].values
    sg = df["sgphi_b4j"].values
    rows.append({
        "arm": "b4j", "taxon": "phi",
        "log_phi_bias": float(lg.mean() - np.log(PHI_CAL)),
        "log_phi_median_bias": float(np.median(lg) - np.log(PHI_CAL)),
        "sd_log_phi": float(lg.std(ddof=1)),
        "median_se_gamma_phi": float(np.median(sg)),
        "cov95_log_phi": float((np.abs((lg - np.log(PHI_CAL))
                                       / np.maximum(sg, 1e-300)) <= Z95).mean()),
        "success_rate": float(df["succ_b4j"].mean()),
    })
    return pd.DataFrame(rows)


def _summarize_posterior(df):
    rows = [{
        "metric": "auc", "R": len(df),
        "mean": float(df["post_auc"].mean()),
        "sd": float(df["post_auc"].std(ddof=1)),
        "q05": float(df["post_auc"].quantile(0.05)),
        "q50": float(df["post_auc"].quantile(0.5)),
        "q95": float(df["post_auc"].quantile(0.95)),
    }, {
        "metric": "struct_zero_frac_among_zeros",
        "mean": float(df["struct_frac"].mean()),
        "post_mean_posterior": float(df["post_mean"].mean()),
        "calibration_error": float((df["post_mean"]
                                    - df["struct_frac"]).mean()),
    }]
    summ = pd.DataFrame(rows)
    # 池化可靠性分箱
    bins = []
    for k in range(10):
        cnt = df[f"bin{k}_cnt"].sum()
        if cnt > 0:
            bins.append({"bin": k, "count": int(cnt),
                         "mean_pred": float(df[f"bin{k}_predsum"].sum() / cnt),
                         "frac_structural": float(df[f"bin{k}_possum"].sum()
                                                  / cnt)})
        else:
            bins.append({"bin": k, "count": 0,
                         "mean_pred": np.nan, "frac_structural": np.nan})
    return summ, pd.DataFrame(bins)


# ---------------------------------------------------------------------------
# C2 脊区
# ---------------------------------------------------------------------------

def _rep_ridge(seed):
    rng = np.random.default_rng(seed)
    Y, Z, N = _sim_ridge(rng)
    p = len(PI_RIDGE)
    row = {"seed": seed}
    for tag, b in [("b1", 1), ("b4", 4)]:
        f = clc.fit_count_composite(Y, N, b=b, phi_known=PHI_RIDGE)
        row[f"succ_{tag}"] = f["success"]
        for j in range(p):
            row[f"la_{tag}_{j}"] = np.log(f["pi"][j] / (1 - f["pi"][j]))
            row[f"lt_{tag}_{j}"] = np.log(f["theta"][j])
            row[f"det_{j}"] = int((Y[:, j] > 0).sum())
    return row


def _summarize_ridge(df):
    rows = []
    e_j = model.effective_detection_strength(TH_RIDGE, PHI_RIDGE, 1e3)
    for tag in ["b1", "b4"]:
        for j in range(len(PI_RIDGE)):
            la = df[f"la_{tag}_{j}"].values
            lt = df[f"lt_{tag}_{j}"].values
            rows.append({
                "arm": tag, "taxon": j, "e_j": e_j,
                "mean_det": float(df[f"det_{j}"].mean()),
                "sd_log_theta": float(lt.std(ddof=1)),
                "corr_logpi_logtheta": float(np.corrcoef(la, lt)[0, 1]),
                "theta_median_rel": float(np.median(np.exp(lt)) / TH_RIDGE),
                "frac_theta_boundary": float((np.exp(lt) > 0.8).mean()),
                "frac_pi_boundary": float((np.abs(la) > 9.1).mean()),
                "success_rate": float(df[f"succ_{tag}"].mean()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# C4 复杂度实测
# ---------------------------------------------------------------------------

def exp_complexity():
    rng = np.random.default_rng(7)
    rows = []
    # (a) magnitude_loglik_grad 单次评估耗时 vs b 与 p
    for p in [8, 32, 128]:
        n = 200
        th = np.full(p, 1e-3)
        Y = rng.poisson(3.0, (n, p)) * rng.integers(0, 2, (n, p))
        N = np.full(n, 3000)
        for b in [2, 4, 8, 12]:
            blocks = clc.make_blocks(p, b)
            clc.magnitude_loglik_grad(Y, N, th, 3000.0, blocks)
            t0 = time.perf_counter()
            rep = 20
            for _ in range(rep):
                clc.magnitude_loglik_grad(Y, N, th, 3000.0, blocks)
            dt = (time.perf_counter() - t0) / rep * 1000
            rows.append({"what": "mag_eval_ms", "p": p, "b": b,
                         "n": n, "ms": dt})
    # (b) 全拟合墙钟 vs b（主校准单数据集）
    Y, Z, N = _sim_main(np.random.default_rng(11))
    for b in [1, 2, 4, 8]:
        t0 = time.perf_counter()
        clc.fit_count_composite(Y, N, b=b, phi_known=PHI_CAL)
        rows.append({"what": "fit_wall_s", "p": P, "b": b, "n": N_SAMP,
                     "ms": (time.perf_counter() - t0)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "block_complexity.csv", index=False)
    print("== C4: 复杂度实测 ==")
    print(df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=200)
    ap.add_argument("--cores", type=int, default=2)
    ap.add_argument("--exp", default="all",
                    choices=["all", "main", "ridge", "complexity"])
    args = ap.parse_args()
    from multiprocessing import Pool

    if args.exp in ("all", "main"):
        t0 = time.time()
        with Pool(args.cores) as pool:
            rows = pool.map(_rep_main, range(60_000, 60_000 + args.R),
                            chunksize=4)
        df = pd.DataFrame(rows)
        df.to_csv(OUT / "count_calibration_perrep.csv", index=False)
        summ = _summarize_main(df)
        summ.to_csv(OUT / "count_calibration_summary.csv", index=False)
        psumm, bins = _summarize_posterior(df)
        psumm.to_csv(OUT / "posterior_validation.csv", index=False)
        bins.to_csv(OUT / "posterior_calibration_bins.csv", index=False)
        print("== C1: 计数幅度块 CL 主校准（R=%d, %.0fs）==" % (args.R,
                                                              time.time() - t0))
        for tag in ["b1k", "b4k", "b8k", "b4j"]:
            s = summ[(summ["arm"] == tag)
                     & (summ["taxon"].astype(str) != "phi")]
            print("  [%s] log θ 偏差 [%.3f, %.3f]（中位 [%.3f, %.3f]），"
                  "sd(log θ) [%.2f, %.2f]，cov95 θ [%.3f, %.3f]，"
                  "corr(π,θ) [%.2f, %.2f]"
                  % (tag, s["log_theta_bias"].min(), s["log_theta_bias"].max(),
                     s["log_theta_median_bias"].min(),
                     s["log_theta_median_bias"].max(),
                     s["sd_log_theta"].min(), s["sd_log_theta"].max(),
                     s["cov95_log_theta"].min(), s["cov95_log_theta"].max(),
                     s["corr_logpi_logtheta"].min(),
                     s["corr_logpi_logtheta"].max()))
            print("       logit π 偏差 [%.3f, %.3f]，cov95 π [%.3f, %.3f]，"
                  "θ 撞界率 [%.3f, %.3f]"
                  % (s["logit_pi_bias"].min(), s["logit_pi_bias"].max(),
                     s["cov95_logit_pi"].min(), s["cov95_logit_pi"].max(),
                     s["frac_theta_boundary"].min(),
                     s["frac_theta_boundary"].max()))
        pr = summ[summ["taxon"].astype(str) == "phi"].iloc[0]
        print("  [b4j] log φ̂ 偏差 %.3f（中位 %.3f），sd %.2f，cov95 %.3f"
              % (pr["log_phi_bias"], pr["log_phi_median_bias"],
                 pr["sd_log_phi"], pr["cov95_log_phi"]))
        print("== C3: 逐单元后验 ==")
        print("  AUC 均值 %.3f（sd %.3f，[q05,q95]=[%.3f, %.3f]）"
              % (psumm.loc[0, "mean"], psumm.loc[0, "sd"],
                 psumm.loc[0, "q05"], psumm.loc[0, "q95"]))
        print("  零单元结构零占比 %.3f vs 平均后验 %.3f（校准误差 %+.3f）"
              % (psumm.loc[1, "mean"], psumm.loc[1, "post_mean_posterior"],
                 psumm.loc[1, "calibration_error"]))
        print(bins.to_string(index=False))

    if args.exp in ("all", "ridge"):
        t0 = time.time()
        R2 = min(args.R, 150)
        with Pool(args.cores) as pool:
            rows = pool.map(_rep_ridge, range(70_000, 70_000 + R2),
                            chunksize=4)
        df = pd.DataFrame(rows)
        df.to_csv(OUT / "count_ridge_perrep.csv", index=False)
        summ = _summarize_ridge(df)
        summ.to_csv(OUT / "count_ridge_summary.csv", index=False)
        print("== C2: 脊区 b=1 vs b=4（R=%d, %.0fs）==" % (R2,
                                                          time.time() - t0))
        print(summ.to_string(index=False))

    if args.exp in ("all", "complexity"):
        exp_complexity()


if __name__ == "__main__":
    main()
