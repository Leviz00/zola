"""validate.py — 全部验证实验：g 三方对照、模拟自校准、脊区域、φ 可识别性。

运行（结果写入 results/，全部数字来自实际运行）：
    python3 validate.py --exp all --cores 2

实验设计依据（fix_N1 命题 prop:ident 及 remark）：
  - 可识别区域由有效一阶检出强度 e_j = φθ̄[ψ(φ+N_min)−ψ(φ)] 与 Fisher 信息
    共同标定；主校准取 φ=3000（深度窗口 [1e3,1e5] 跨越 N≈3φ=9e3，满足
    prop (iii) 的形状条件），θ̄∈[7e-4,1.05e-3]（φθ̄∈[2.1,3.15]，
    e_j∈[0.60,0.91]，检出曲线在窗口内不饱和、Fisher sd(log θ̂)≈0.2）。
  - 大体量类群 π=1（恒在场）使归一化耦合误差 E1≈±0.3%（fix_N1 §3.2 的
    "条件于其余类群在场"极限，ZIBB 边际近似精确）。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

import model
import composite_likelihood as cl

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 主校准设计（可识别区域）
# ---------------------------------------------------------------------------
PHI_CAL = 3000.0
PI_CAL = np.array([0.85, 0.87, 0.89, 0.91, 0.93, 0.94, 0.95, 0.95])
TH_CAL = np.array([7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5]) * 1e-4
N_SAMP = 200
Z95, Z90 = 1.959964, 1.644854


def _depths(rng, n=N_SAMP, lo=1e3, hi=1e5):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n)).astype(int)


def _sim_cal(rng):
    theta_bar = np.concatenate([TH_CAL, [1.0 - TH_CAL.sum()]])
    pi_all = np.concatenate([PI_CAL, [1.0]])
    N = _depths(rng)
    Y, Z = model.simulate_three_layer(pi_all, theta_bar, PHI_CAL, N, rng)
    return cl.detection_indicators(Y)[:, :8], N


# ---------------------------------------------------------------------------
# Exp 0：g 闭式 vs 显式乘积 vs 数值积分（fix_N1 §1.3 网格）
# ---------------------------------------------------------------------------

def exp0_g_verification():
    rows = []
    grid = [(N, th, ph) for ph in [0.5, 1.0, 5.0]
            for N in [10, 1000] for th in [0.001, 0.01]]
    for (N, th, ph) in grid:
        gc = model.g_closed(N, th, ph)
        gp = model.g_product(N, th, ph)
        gq = model.g_quad(N, th, ph)
        exact = 1.0 - gc
        o1 = model.one_minus_g_order1(N, th, ph)
        o2 = model.one_minus_g_order2(N, th, ph)
        rows.append({
            "phi": ph, "N": N, "theta": th,
            "g_closed": gc, "g_product": gp, "g_quad": gq,
            "absdiff_closed_product": abs(gc - gp),
            "absdiff_closed_quad": abs(gc - gq),
            "one_minus_g_exact": exact,
            "order1_phi_theta_dpsi": o1,
            "order2_corrected": o2,
            "order1_rel_err": (o1 - exact) / exact,
            "order2_rel_err": (o2 - exact) / exact,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "g_verification.csv", index=False)
    print("== Exp 0: g 三方对照（12 格）==")
    print("  max |closed-product| = %.2e, max |closed-quad| = %.2e"
          % (df["absdiff_closed_product"].max(), df["absdiff_closed_quad"].max()))
    print("  一阶展开相对误差范围 [%.2e, %.2e]，二阶 [%.2e, %.2e]"
          % (df["order1_rel_err"].min(), df["order1_rel_err"].max(),
             df["order2_rel_err"].min(), df["order2_rel_err"].max()))
    return df


# ---------------------------------------------------------------------------
# Exp 1：模拟自校准（R=500）
# ---------------------------------------------------------------------------

def _rep_calibration(seed):
    rng = np.random.default_rng(seed)
    D, N = _sim_cal(rng)
    fk = cl.fit_composite(D, N, phi_known=PHI_CAL)
    fj = cl.fit_composite(D, N)
    row = {"seed": seed, "succ_k": fk["success"], "succ_j": fj["success"]}
    for j in range(8):
        row[f"pi_k_{j}"] = fk["pi"][j]; row[f"th_k_{j}"] = fk["theta"][j]
        row[f"sa_k_{j}"] = fk["se_alpha"][j]; row[f"sb_k_{j}"] = fk["se_beta"][j]
        row[f"saN_k_{j}"] = fk["se_pi_naive"][j] / (fk["pi"][j] * (1 - fk["pi"][j]))
        row[f"sbN_k_{j}"] = fk["se_theta_naive"][j] / fk["theta"][j]
    row["phi_j"] = fj["phi"]; row["sg_j"] = fj["se_gamma"]
    for j in range(8):
        row[f"pi_j_{j}"] = fj["pi"][j]; row[f"th_j_{j}"] = fj["theta"][j]
        row[f"sa_j_{j}"] = fj["se_alpha"][j]; row[f"sb_j_{j}"] = fj["se_beta"][j]
    return row


def _summarize_calibration(df):
    rows = []
    la, lb = logit(PI_CAL), np.log(TH_CAL)
    for arm, tag in [("k", "phi_known"), ("j", "joint")]:
        succ = df[f"succ_{arm}"].mean()
        for j in range(8):
            pi_h, th_h = df[f"pi_{arm}_{j}"].values, df[f"th_{arm}_{j}"].values
            ah, bh = logit(pi_h), np.log(th_h)
            sa, sb = df[f"sa_{arm}_{j}"].values, df[f"sb_{arm}_{j}"].values
            cov_a = float(((ah - Z95 * sa <= la[j]) & (la[j] <= ah + Z95 * sa)).mean())
            cov_b = float(((bh - Z95 * sb <= lb[j]) & (lb[j] <= bh + Z95 * sb)).mean())
            row = {
                "arm": tag, "taxon": j,
                "pi_true": PI_CAL[j], "theta_true": TH_CAL[j],
                "pi_rel_bias": pi_h.mean() / PI_CAL[j] - 1,
                "theta_rel_bias": th_h.mean() / TH_CAL[j] - 1,
                "theta_median_rel_bias": np.median(th_h) / TH_CAL[j] - 1,
                # Jensen/偏度分解：lognormal 预测 mean/θ−1 ≈ log bias + sd²/2
                "theta_jensen_pred": (bh.mean() - lb[j])
                + 0.5 * bh.std(ddof=1) ** 2,
                "logit_pi_bias": ah.mean() - la[j],
                "log_theta_bias": bh.mean() - lb[j],
                "sd_log_theta": bh.std(ddof=1),
                "cov95_logit_pi": cov_a, "cov95_log_theta": cov_b,
                "frac_theta_boundary": float((th_h > 0.8).mean()),
                "success_rate": succ,
            }
            if arm == "k":
                saN, sbN = df[f"saN_k_{j}"].values, df[f"sbN_k_{j}"].values
                row["cov95_logit_pi_naive"] = float(
                    ((ah - Z95 * saN <= la[j]) & (la[j] <= ah + Z95 * saN)).mean())
                row["cov95_log_theta_naive"] = float(
                    ((bh - Z95 * sbN <= lb[j]) & (lb[j] <= bh + Z95 * sbN)).mean())
            rows.append(row)
    # joint 臂的 φ
    ph, sg = df["phi_j"].values, df["sg_j"].values
    lg = np.log(ph)
    rows.append({
        "arm": "joint", "taxon": "phi", "pi_true": np.nan, "theta_true": np.nan,
        "phi_rel_bias": ph.mean() / PHI_CAL - 1,
        "phi_median_rel_bias": np.median(ph) / PHI_CAL - 1,
        "log_phi_bias": lg.mean() - np.log(PHI_CAL),
        "sd_log_phi": lg.std(ddof=1),
        "cov95_log_phi": float(((lg - Z95 * sg <= np.log(PHI_CAL))
                                & (np.log(PHI_CAL) <= lg + Z95 * sg)).mean()),
        "phi_iqr": np.quantile(ph, 0.75) - np.quantile(ph, 0.25),
    })
    return pd.DataFrame(rows)


def exp1_calibration(R, cores):
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(cores) as pool:
        rows = pool.map(_rep_calibration, range(10_000, 10_000 + R), chunksize=8)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "calibration_perrep.csv", index=False)
    summ = _summarize_calibration(df)
    summ.to_csv(OUT / "calibration_summary.csv", index=False)
    print("== Exp 1: 模拟自校准（R=%d, n=%d, φ=%.0f, %.0fs）=="
          % (R, N_SAMP, PHI_CAL, time.time() - t0))
    k = summ[summ["arm"] == "phi_known"]
    print("  [φ 已知 profile 臂]")
    print("  π 相对偏差范围 [%.1f%%, %.1f%%]" % (100 * k["pi_rel_bias"].min(),
                                                100 * k["pi_rel_bias"].max()))
    print("  θ 相对偏差范围 [%.1f%%, %.1f%%]" % (100 * k["theta_rel_bias"].min(),
                                                  100 * k["theta_rel_bias"].max()))
    print("  Godambe cov95 logit-π 范围 [%.3f, %.3f]，log-θ [%.3f, %.3f]"
          % (k["cov95_logit_pi"].min(), k["cov95_logit_pi"].max(),
             k["cov95_log_theta"].min(), k["cov95_log_theta"].max()))
    j = summ[summ["arm"] == "joint"]
    print("  [φ 未知联合臂] θ 相对偏差 [%.1f%%, %.1f%%]，sd(log θ) 中位 %.2f"
          % (100 * j["theta_rel_bias"].min(), 100 * j["theta_rel_bias"].max(),
             j["sd_log_theta"].median()))
    prow = summ[summ["taxon"] == "phi"].iloc[0]
    print("  φ̂：相对偏差 %.1f%%（中位 %.1f%%），sd(log φ̂)=%.2f，cov95=%.3f"
          % (100 * prow["phi_rel_bias"], 100 * prow["phi_median_rel_bias"],
             prow["sd_log_phi"], prow["cov95_log_phi"]))
    return df, summ


# ---------------------------------------------------------------------------
# Exp 2：脊区域对照（θ̄=0.001，e_j 远低于阈值）
# ---------------------------------------------------------------------------

# 脊区域：θ̄=0.001（命题 (ii) 稀有区）。φ=10 已知以隔离 π–θ 脊
# （e_j=0.047 远低于主校准的 0.60–0.91；φ=2 时 e_j=0.013，每数据集检出
# 仅 ~2–9 个，退化为全边界解，附 results/ridge_phi2_perrep.csv 对照）。
PHI_RIDGE = 10.0
TH_RIDGE = 0.001
PI_RIDGE = np.array([0.85, 0.90, 0.95, 0.97])


def _rep_ridge(seed):
    rng = np.random.default_rng(seed)
    p = len(PI_RIDGE)
    theta_bar = np.concatenate([np.full(p, TH_RIDGE), [1.0 - p * TH_RIDGE]])
    pi_all = np.concatenate([PI_RIDGE, [1.0]])
    N = _depths(rng)
    Y, Z = model.simulate_three_layer(pi_all, theta_bar, PHI_RIDGE, N, rng)
    D = cl.detection_indicators(Y)[:, :p]
    f = cl.fit_composite(D, N, phi_known=PHI_RIDGE)
    row = {"seed": seed, "success": f["success"]}
    for j in range(p):
        row[f"pi_{j}"] = f["pi"][j]; row[f"th_{j}"] = f["theta"][j]
        row[f"det_{j}"] = D[:, j].sum()
    return row


def exp2_ridge(R, cores):
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(cores) as pool:
        rows = pool.map(_rep_ridge, range(20_000, 20_000 + R), chunksize=8)
    df = pd.DataFrame(rows)
    tag = "ridge" if PHI_RIDGE == 10.0 else f"ridge_phi{int(PHI_RIDGE)}"
    df.to_csv(OUT / f"{tag}_perrep.csv", index=False)
    e_j = model.effective_detection_strength(TH_RIDGE, PHI_RIDGE, 1e3)
    # 主校准的 sd(log θ̂) 参考值（若已跑 Exp 1），用于方差膨胀倍数
    try:
        cal = pd.read_csv(OUT / "calibration_summary.csv")
        sd_ref = float(cal.loc[cal["arm"] == "phi_known", "sd_log_theta"]
                       .median())
    except Exception:
        sd_ref = np.nan
    rows = []
    for j in range(len(PI_RIDGE)):
        lp, lt = np.log(df[f"pi_{j}"].values), np.log(df[f"th_{j}"].values)
        prod = lp + lt  # log(π̂θ̂)：理论可识别泛函
        interior = (df[f"pi_{j}"].values < 0.999) & (df[f"th_{j}"].values < 0.8)
        rows.append({
            "taxon": j, "pi_true": PI_RIDGE[j], "theta_true": TH_RIDGE,
            "e_j": e_j,
            "mean_det_per_dataset": df[f"det_{j}"].mean(),
            "corr_logpi_logtheta_all": float(np.corrcoef(lp, lt)[0, 1]),
            "corr_logpi_logtheta_interior": float(
                np.corrcoef(lp[interior], lt[interior])[0, 1])
            if interior.sum() > 10 else np.nan,
            "frac_interior": float(interior.mean()),
            "sd_log_theta": lt.std(ddof=1),
            "sd_log_theta_interior": lt[interior].std(ddof=1)
            if interior.sum() > 10 else np.nan,
            "sd_log_pi_theta_product": prod.std(ddof=1),
            "var_inflation_vs_calib": (lt.std(ddof=1) / sd_ref) ** 2
            if sd_ref == sd_ref else np.nan,
            "product_rel_bias": np.exp(prod.mean()) / (PI_RIDGE[j] * TH_RIDGE) - 1,
            "frac_pi_boundary": float(((df[f"pi_{j}"] > 0.999)
                                       | (df[f"pi_{j}"] < 1.1e-4)).mean()),
            "frac_theta_boundary": float((df[f"th_{j}"] > 0.8).mean()),
            "theta_median_rel": np.median(df[f"th_{j}"]) / TH_RIDGE,
        })
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / f"{tag}_summary.csv", index=False)
    print("== Exp 2: 脊区域（θ̄=0.001, φ=%.0f 已知, e_j=%.4f, R=%d, %.0fs）=="
          % (PHI_RIDGE, e_j, R, time.time() - t0))
    print(summ[["taxon", "corr_logpi_logtheta_all",
                "corr_logpi_logtheta_interior", "sd_log_theta",
                "sd_log_pi_theta_product", "var_inflation_vs_calib",
                "frac_theta_boundary", "theta_median_rel"]].to_string(index=False))
    return df, summ


# ---------------------------------------------------------------------------
# Exp 3：φ 已知/未知 与 "深度需跨越 N≈3φ"（定性）
# ---------------------------------------------------------------------------

def exp3_shape_grid():
    """确定性形状残差（fix_N1 §1.4 P3 复现）：S_N(φ) 形状经最优标量放大后
    的相对残差——越小越不可区分。"""
    supports = {
        "cross_10_1e5": np.geomspace(10, 1e5, 15),
        "big_only_1e4_1e5": np.geomspace(1e4, 1e5, 5),
        "small_only_10_100": np.geomspace(10, 100, 5),
    }
    pairs = [(1.0, 2.0), (1.0, 5.0), (0.5, 5.0)]
    rows = []
    for sname, Ns in supports.items():
        for (a, b) in pairs:
            Sa = model.S_N(Ns, a); Sb = model.S_N(Ns, b)
            c = float(Sa @ Sb / (Sb @ Sb))   # 最优标量放大
            resid = float(np.sqrt(np.mean((Sa - c * Sb) ** 2)) / np.mean(Sa))
            rows.append({"support": sname, "phi_a": a, "phi_b": b,
                         "rel_residual": resid})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "phi_shape_grid.csv", index=False)
    print("== Exp 3a: S_N 形状残差（越小越不可区分）==")
    print(df.to_string(index=False))
    return df


PHI_E3 = 30.0
TH_E3 = np.array([7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5]) * 1e-3  # φθ≈0.21–0.32


def _rep_phi(setting):
    seed, tag = setting
    rng = np.random.default_rng(seed)
    if tag == "S2_not_spanned":
        N = _depths(rng, lo=1e3, hi=1e5)     # 3φ=90 ≪ N_min：不跨越
    else:
        N = _depths(rng, lo=1e1, hi=1e5)     # 3φ=90 ∈ [10,1e5]：跨越
    theta_bar = np.concatenate([TH_E3, [1.0 - TH_E3.sum()]])
    pi_all = np.concatenate([PI_CAL, [1.0]])
    Y, Z = model.simulate_three_layer(pi_all, theta_bar, PHI_E3, N, rng)
    D = cl.detection_indicators(Y)[:, :8]
    fj = cl.fit_composite(D, N)
    row = {"seed": seed, "setting": tag, "success": fj["success"],
           "phi": fj["phi"], "se_gamma": fj["se_gamma"]}
    for j in range(8):
        row[f"th_{j}"] = fj["theta"][j]
    return row


def exp3_phi(R, cores):
    from multiprocessing import Pool
    t0 = time.time()
    jobs = ([(30_000 + r, "S2_not_spanned") for r in range(R)]
            + [(40_000 + r, "S3_spanned") for r in range(R)])
    with Pool(cores) as pool:
        rows = pool.map(_rep_phi, jobs, chunksize=8)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "phi_experiment_perrep.csv", index=False)
    rows = []
    for tag, sub in df.groupby("setting"):
        lg = np.log(sub["phi"].values)
        rows.append({
            "setting": tag, "phi_true": PHI_E3, "R": len(sub),
            "log_phi_bias": lg.mean() - np.log(PHI_E3),
            "sd_log_phi": lg.std(ddof=1),
            "iqr_log_phi": np.quantile(lg, 0.75) - np.quantile(lg, 0.25),
            "frac_phi_boundary": float(((sub["phi"] < 0.06)
                                        | (sub["phi"] > 9e4)).mean()),
            "sd_log_theta_median": float(
                np.log(sub[[f"th_{j}" for j in range(8)]].values).std(0, ddof=1)
                .mean()),
            "success_rate": sub["success"].mean(),
        })
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "phi_experiment_summary.csv", index=False)
    print("== Exp 3b: φ=30 联合估计（跨 vs 不跨 N≈3φ，R=%d/组, %.0fs）=="
          % (R, time.time() - t0))
    print(summ.to_string(index=False))
    return df, summ


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all",
                    choices=["all", "g", "cal", "ridge", "phi"])
    ap.add_argument("--R", type=int, default=500)
    ap.add_argument("--cores", type=int, default=2)
    args = ap.parse_args()
    if args.exp in ("all", "g"):
        exp0_g_verification()
    if args.exp in ("all", "cal"):
        exp1_calibration(args.R, args.cores)
    if args.exp in ("all", "ridge"):
        exp2_ridge(min(args.R, 300), args.cores)
    if args.exp in ("all", "phi"):
        exp3_shape_grid()
        exp3_phi(min(args.R, 150), args.cores)


if __name__ == "__main__":
    main()
