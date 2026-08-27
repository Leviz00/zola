"""estimated 档加权 DA 模拟运行器：ZOLA 估计管线 → 48 格 fractional grid。

桥接说明（估计框架 ../estimation/ 与模拟框架的对接；不复制估计代码，只
import 调用）：

  每个重复 = 生成数据 → 跑 ZOLA 估计 → 产出 Ŵ → 注入加权 DA → 指标。

  1. 生成：generators.generate(mechanism, params, n, p=100, depths, seed)，
     种子树与 run_baseline_grid.py 完全一致（cell 种子 → spawn(R) 逐重复
     子流；SeedSequence.spawn 的第 i 个子流与 R 无关，故本运行器 rep r 与
     baseline/其他网格运行的 rep r 是同一数据集，可直接配对）。
  2. 估计（真实后验权重 Ŵ 的来源）：
     a. 协变量层检出指示 CL：composite_likelihood_cov.fit_composite_cov(
        D, W=[1, group], N, phi_known=None) —— 逐细胞 π̂_ij = σ(γ̂_j'W_i)，
        信息性零（缺席与 case 组对齐）由此进入后验；
     b. 计数块精炼：composite_likelihood_count.count_loglik_grad（总复合
        似然 = 检出指示 CL + 零截断 DM 块幅度项，b=4 推荐默认）用自驱动
        单起点 L-BFGS-B 精炼 (π, θ̄, φ)，起点取无协变量检出 CL
        composite_likelihood.fit_composite(multi_start=False) 的解——与
        fit_count_composite 内部"检出 CL 暖启动"同一协议；
     c. 逐细胞后验：posterior.zero_source_posterior_cov(Γ̂, W, θ̂, φ̂, N)，
        即 eq:posterior 的逐单元近似 Pr(Z_ij=0 | Y_ij=0)。
     Ŵ 仅在 Y=0 处被 weighting.py 接口读取（Y>0 处由 validate_weights
     强制置 0）；weighting.py 的接口与既有结果文件均未改动。
  3. 检测（alpha=0.05 锁定，全部在同一份数据上配对）：
     unweighted : naive_welch_t, tss_wilcoxon
     加权两法   : weighted_welch_t, exclusion_wilcoxon
     三档权重   : oracle（上限）/ placeholder（占位）/ estimated（本档）

算力预算与保真度（依据 analysis/estimated_weighting/ 下的基准实验）：
  全保真驱动（fit_composite_cov 多起点 + fit_count_composite 含 Godambe
  三明治）实测 10–17 分钟/重复，48 格不可行。本运行器采用【预算臂】：
  协变量层 multi_start=False（跳过多起点暖启动——实测暖启动占耗时
  90%+，联合 L-BFGS-B 本身仅数秒）+ 自驱动计数块（同一目标函数、同一
  暖启动协议，仅省略与权重无关的 Godambe 协方差与多起点）。预算臂与
  全驱动在后验 AUC 与 DA 结局上的对照见 MEMO；估计器是 M-估计 argmax，
  省略部分不影响估计量定义，只影响局部最优保护与标准误（本用途不需要）。
  未收敛重复保留（与 estimation/README 的做法一致），逐重复记录
  cov_success / cnt_success / cnt_nit 供审计。

指标（预注册，与 run_weighting_check.py 相同）：逐重复 FDP（无拒绝记 0）
→ 实证 FDR = FDP 均值，MC SE = sd(FDP)/√R；功效 = 逐重复 TPR 均值。
诊断列（不进判定）：后验对真值结构零的 AUC、post_mean、struct_frac、
φ̂ 两阶段值、估计耗时。

断点续跑：逐格子写 checkpoints（results/estimated_weighting_ckpt/
cell_{id}_metrics.csv 与 cell_{id}_diag.csv），主进程逐重复追加；
重启时跳过已有 (cell_id, rep)。

输出（results/）：
  estimated_weighting_replicates.csv  每 格子×重复×方法 一行
  estimated_weighting_summary.csv     每 格子×方法 的 FDR/功效 + MC SE
  estimated_weighting_diagnostics.csv 每 格子×重复 的估计诊断
  estimated_weighting_by_informative.csv 按信息性 on/off 汇总

运行：python3 run_estimated_weighting.py [--R 40] [--cells 1,19,...] [--cores 2]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

HERE = os.path.dirname(os.path.abspath(__file__))
EST_DIR = os.path.normpath(os.path.join(HERE, "..", "estimation"))
if EST_DIR not in sys.path:
    sys.path.insert(0, EST_DIR)

import design  # noqa: E402
import generators  # noqa: E402
import metrics  # noqa: E402
from baselines_py import ALPHA, naive_welch_t, tss_wilcoxon  # noqa: E402
from weighting import (  # noqa: E402
    exclusion_wilcoxon,
    oracle_weights,
    placeholder_weights,
    validate_weights,
    weighted_welch_t,
)

# 估计框架（import 使用，不复制）
import composite_likelihood as est_cl  # noqa: E402
import composite_likelihood_cov as est_clcov  # noqa: E402
import composite_likelihood_count as est_clc  # noqa: E402
import posterior as est_post  # noqa: E402

RESULTS_DIR = os.path.join(HERE, "results")
CKPT_DIR = os.path.join(RESULTS_DIR, "estimated_weighting_ckpt")
CONFIG_CSV = os.path.join(HERE, "configs", "config_fractional.csv")

P_TAXA = 100
COUNT_BLOCK_B = 4          # 计数块大小（estimation README 推荐默认）
COV_MAXITER = 500
DET_MAXITER = 500
COUNT_MAXITER = 200        # 计数块精炼迭代上限（全驱动为 1000，见 MEMO 预算论证）
COUNT_GTOL = 1e-6
COUNT_FTOL = 1e-10

UNWEIGHTED_METHODS = ["naive_welch_t", "tss_wilcoxon"]
WEIGHT_TIERS = ["oracle", "placeholder", "estimated"]
WEIGHTED_METHODS = ["weighted_welch_t", "exclusion_wilcoxon"]
ALL_METHODS = (UNWEIGHTED_METHODS
               + [f"{m}_{t}" for m in WEIGHTED_METHODS for t in WEIGHT_TIERS])


# ---------------------------------------------------------------------------
# 估计桥接
# ---------------------------------------------------------------------------

def _count_refine(Y, N, pi0, theta0, phi0, b=COUNT_BLOCK_B,
                  maxiter=COUNT_MAXITER, gtol=COUNT_GTOL, ftol=COUNT_FTOL):
    """计数块精炼：最大化 检出指示 CL + 零截断 DM 块幅度项 的总复合似然。

    仅用估计框架的公开目标 est_clc.count_loglik_grad；ψ 布局
    (logφ, logit π_1..p, log θ_1..p) 见 composite_likelihood._pack 的文档。
    与 fit_count_composite 相同的边界、梯度裁剪与收敛判据思路；
    省略 Godambe 三明治（权重只需要点估计）。
    """
    p = Y.shape[1]
    blocks = est_clc.make_blocks(p, b)
    psi0 = np.concatenate([[np.log(phi0)], logit(np.clip(pi0, 1e-4, 0.9999)),
                           np.log(np.clip(theta0, 1e-7, 0.9))])
    bounds = ([(np.log(0.05), np.log(1e5))]
              + [(logit(1e-4), logit(0.9999))] * p
              + [(np.log(1e-7), np.log(0.9))] * p)

    def obj(psi):
        ll, g = est_clc.count_loglik_grad(psi, Y, N, blocks, phi_known=None)
        ng = float(np.abs(g).max())
        if ng > 1e3:  # 与 fit_count_composite 相同的梯度裁剪
            g = g * (1e3 / ng)
        return -ll, -g

    res = minimize(obj, psi0, method="L-BFGS-B", jac=True, bounds=bounds,
                   options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol})
    success = bool(res.success)
    if not success and np.max(np.abs(res.jac)) < 1e-3:
        success = True  # 与估计框架一致的投影梯度判据
    return {"phi": float(np.exp(res.x[0])), "pi": expit(res.x[1:1 + p]),
            "theta": np.exp(res.x[1 + p:]), "success": success,
            "nit": int(res.nit)}


def estimated_weights(Y, group):
    """真实估计后验权重 Ŵ（接口：仅在 Y=0 处被读取）。

    返回 (W, diag)：W ∈ [0,1]^{n×p}（已经 validate_weights 裁剪掩码）；
    diag 为诊断字典（收敛标志、φ̂、后验汇总、耗时）。
    """
    Y = np.asarray(Y, dtype=float)
    group = np.asarray(group)
    D = (Y > 0).astype(float)
    N = Y.sum(axis=1).astype(float)
    Wd = np.column_stack([np.ones(group.shape[0]), group.astype(float)])
    diag = {}
    t0 = time.time()
    # (a) 协变量层检出指示 CL（联合 φ；multi_start=False 为预算臂，
    #     跳过的多起点暖启动实测占耗时 90%+，见 MEMO）
    f_cov = est_clcov.fit_composite_cov(D, Wd, N, phi_known=None,
                                        multi_start=False,
                                        maxiter=COV_MAXITER)
    diag["cov_success"] = bool(f_cov["success"])
    diag["cov_phi"] = float(f_cov["phi"])
    diag["t_cov"] = time.time() - t0
    t0 = time.time()
    # (b1) 无协变量检出 CL 暖启动（与 fit_count_composite 内部协议一致）
    f_det = est_cl.fit_composite(D, N, phi_known=None, multi_start=False,
                                 maxiter=DET_MAXITER)
    # (b2) 计数块精炼
    f_cnt = _count_refine(Y, N, f_det["pi"], f_det["theta"], f_det["phi"])
    diag["cnt_success"] = bool(f_cnt["success"])
    diag["cnt_nit"] = int(f_cnt["nit"])
    diag["cnt_phi"] = float(f_cnt["phi"])
    diag["t_count"] = time.time() - t0
    # (c) 逐细胞后验：协变量版 π̂_ij + 计数块精炼的 θ̂、φ̂
    P = est_post.zero_source_posterior_cov(f_cov["Gamma"], Wd,
                                           f_cnt["theta"], f_cnt["phi"], N)
    diag["t_est_total"] = diag["t_cov"] + diag["t_count"]
    return validate_weights(P, Y), diag


# ---------------------------------------------------------------------------
# 逐重复
# ---------------------------------------------------------------------------

def run_one_replicate(args):
    """生成一份数据，跑四档检测 + 估计诊断；返回 (指标行列表, 诊断行)。"""
    cell_id, mechanism, params, n, depth, rep_seed, rep = args
    Y, truth = generators.generate(
        mechanism, params, n=n, p=P_TAXA, depths=depth, seed=rep_seed
    )
    group = truth["group"]
    da = truth["da_taxa"]

    rejects = {
        "naive_welch_t": naive_welch_t(Y, group)["reject"],
        "tss_wilcoxon": tss_wilcoxon(Y, group)["reject"],
    }
    W_or = oracle_weights(Y, truth)
    W_ph = placeholder_weights(Y, group)
    est_status = "ok"
    try:
        W_est, diag = estimated_weights(Y, group)
    except Exception as exc:  # 不静默跳过：记录状态，estimated 档记 NaN
        W_est, diag = None, {}
        est_status = f"failed:{type(exc).__name__}"
    for tier, W in [("oracle", W_or), ("placeholder", W_ph)]:
        rejects[f"weighted_welch_t_{tier}"] = weighted_welch_t(
            Y, group, W)["reject"]
        rejects[f"exclusion_wilcoxon_{tier}"] = exclusion_wilcoxon(
            Y, group, W)["reject"]
    if W_est is not None:
        rejects["weighted_welch_t_estimated"] = weighted_welch_t(
            Y, group, W_est)["reject"]
        rejects["exclusion_wilcoxon_estimated"] = exclusion_wilcoxon(
            Y, group, W_est)["reject"]

    rows = []
    for method in ALL_METHODS:
        if method in rejects:
            fdp_r, n_rej = metrics.fdp(rejects[method], da)
            tpr_r, _ = metrics.tpr(rejects[method], da)
        else:  # 估计失败的重复：estimated 档记 NaN，汇总时剔除并计数
            fdp_r, tpr_r, n_rej = float("nan"), float("nan"), -1
        rows.append(dict(cell_id=cell_id, rep=rep, method=method,
                         fdp=fdp_r, tpr=tpr_r, n_rej=n_rej))

    drow = dict(cell_id=cell_id, rep=rep, est_status=est_status, **diag)
    if W_est is not None:
        zero = Y == 0
        labels = truth["structural_zeros"][zero].astype(float)
        scores = W_est[zero]
        drow["post_mean"] = float(scores.mean()) if scores.size else np.nan
        drow["struct_frac"] = float(labels.mean()) if labels.size else np.nan
        try:
            drow["post_auc"] = metrics.auc(scores, labels.astype(bool))
        except ValueError:
            drow["post_auc"] = np.nan  # 单一类别（如高深度格全为结构零）
    return rows, drow


# ---------------------------------------------------------------------------
# 断点续跑主循环
# ---------------------------------------------------------------------------

def cell_tasks(row, R):
    child_seeds = np.random.SeedSequence(int(row["seed"])).spawn(R)
    params = design.params_for_cell(row)
    return [
        (int(row["cell_id"]), row["mechanism"], params, int(row["n"]),
         int(row["depth"]), child_seeds[r], r)
        for r in range(R)
    ]


def _append_csv(df_new, path):
    """追加写（不存在则建文件带表头）。"""
    header = not os.path.exists(path)
    df_new.to_csv(path, mode="a", header=header, index=False)


def run_grid(cfg, R, cores):
    os.makedirs(CKPT_DIR, exist_ok=True)
    t_start = time.time()
    for _, row in cfg.iterrows():
        cid = int(row["cell_id"])
        ckpt_m = os.path.join(CKPT_DIR, f"cell_{cid}_metrics.csv")
        ckpt_d = os.path.join(CKPT_DIR, f"cell_{cid}_diag.csv")
        done = set()
        if os.path.exists(ckpt_m):
            prev = pd.read_csv(ckpt_m)
            done = set(prev["rep"].unique())
        todo = [t for t in cell_tasks(row, R) if t[6] not in done]
        if not todo:
            print(f"cell {cid}: {R}/{R} 已完成，跳过", flush=True)
            continue
        t0 = time.time()
        n_done = 0
        with Pool(processes=cores) as pool:
            for rows, drow in pool.imap_unordered(run_one_replicate, todo,
                                                  chunksize=1):
                _append_csv(pd.DataFrame(rows), ckpt_m)
                _append_csv(pd.DataFrame([drow]), ckpt_d)
                n_done += 1
                if n_done % 5 == 0 or n_done == len(todo):
                    print(f"cell {cid}: {n_done}/{len(todo)} reps, "
                          f"{time.time() - t0:.0f}s "
                          f"(total {time.time() - t_start:.0f}s)", flush=True)


def summarize(cfg, R):
    """汇总四档 FDR/功效；写正式结果 CSV。"""
    def _load(suffix):
        frames = []
        for cid in cfg["cell_id"]:
            path = os.path.join(CKPT_DIR, f"cell_{cid}_{suffix}.csv")
            if os.path.exists(path):
                frames.append(pd.read_csv(path))
        return pd.concat(frames, ignore_index=True) if frames else None

    rep_df = _load("metrics")
    diag_df = _load("diag")
    if rep_df is None:
        print("no checkpoint data found; nothing to summarize")
        return None, None

    summary = []
    for (cid, method), g in rep_df.groupby(["cell_id", "method"]):
        fdr_hat, fdr_se, n_f = metrics.empirical_rate(g["fdp"].to_numpy())
        tpr_hat, tpr_se, _ = metrics.empirical_rate(g["tpr"].to_numpy())
        summary.append(dict(cell_id=cid, method=method, R=len(g),
                            R_valid=n_f, emp_fdr=fdr_hat, fdr_mc_se=fdr_se,
                            power=tpr_hat, power_mc_se=tpr_se,
                            mean_rejections=g.loc[g["n_rej"] >= 0, "n_rej"].mean()))
    out = cfg.merge(pd.DataFrame(summary), on="cell_id")

    rep_path = os.path.join(RESULTS_DIR, "estimated_weighting_replicates.csv")
    sum_path = os.path.join(RESULTS_DIR, "estimated_weighting_summary.csv")
    diag_path = os.path.join(RESULTS_DIR, "estimated_weighting_diagnostics.csv")
    rep_df.to_csv(rep_path, index=False)
    out.to_csv(sum_path, index=False)
    diag_df.to_csv(diag_path, index=False)

    # 按信息性 on/off 汇总（格级均值的均值，各格等权）
    by_inf = (out.groupby(["informative_zeros", "method"])
              [["emp_fdr", "power", "mean_rejections"]]
              .mean().reset_index())
    inf_path = os.path.join(RESULTS_DIR,
                            "estimated_weighting_by_informative.csv")
    by_inf.to_csv(inf_path, index=False)
    print(f"\nwrote {sum_path}\nwrote {rep_path}\nwrote {diag_path}\n"
          f"wrote {inf_path}")
    return out, by_inf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=40,
                    help="每格重复数（estimated 档算力约束；MC SE 随之报告）")
    ap.add_argument("--cells", type=str, default="",
                    help="逗号分隔的 cell_id 子集（默认全部 48 格）")
    ap.add_argument("--cores", type=int, default=2)
    args = ap.parse_args()

    cfg = pd.read_csv(CONFIG_CSV)
    if args.cells:
        keep = {int(x) for x in args.cells.split(",")}
        cfg = cfg[cfg["cell_id"].isin(keep)].reset_index(drop=True)
    print(f"estimated weighting: {len(cfg)} cells x R = {args.R} "
          f"(p={P_TAXA}, alpha={ALPHA}, b={COUNT_BLOCK_B}, "
          f"count_maxiter={COUNT_MAXITER}, cores={args.cores})", flush=True)
    run_grid(cfg, args.R, args.cores)
    summarize(cfg, args.R)


if __name__ == "__main__":
    main()
