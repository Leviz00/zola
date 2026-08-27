"""perm_glm.py — v3.6 置换经验校准检验（主路线：池化零分布）。

流程（对单数据集 + 单掩码臂）：
  1. 观测组：abs_nb_lrt_stats → lrt_obs (p,)
  2. K 次组标签置换：每次得 lrt_k (p,)（护栏随置换重算，fallback 记 NaN）
  3. **池化经验零**：所有非 NaN 的 (K×p) 个置换统计量合成单一零分布
     p_j = (1 + #{null ≥ lrt_obs_j}) / (1 + n_null) → BH
  4. 自洽 FWER（leave-one-out）：逐置换 k 用其余 K−1 个置换的池化零给
     lrt_k 算 p 值并 BH，记录是否有拒绝 → perm_fdr_calibrated
     （LKO 防止"自己的最大值进自己的零分布"造成的伪拒绝：
     含自身时 p_min=1/(Kp+1) 恰好压过 BH rank-1 阈值，FWER=1 的假象）。

池化可交换性诊断（pool_diagnostic，预注册）：
  * 逐类群零均值统计量（χ²₁ 期望 1；K=20 下 MC-SD≈0.32）与 log μ̂_j、
    log α̂_j 的相关——|r|>0.3 且显著 ⇒ 池化假设受质疑，回退逐类群置换；
  * 池化零 99% 分位 vs χ²₁ 0.99（6.63）——整体形状膨胀幅度；
  * 池化 p vs 逐类群 p（K 分辨率内）对观测统计量的 Spearman 相关。
"""
from __future__ import annotations

import sys

import numpy as np
from scipy import stats as sstats

SIM_V3 = "/mnt/agents/output/code/simulation_v3"
sys.path.insert(0, SIM_V3)
from abs_glm import abs_nb_lrt_stats, _bh  # noqa: E402


def pooled_p(stat, null):
    """经验 p：(1 + #{null >= stat}) / (1 + n_null)；stat NaN -> 1。"""
    stat = np.asarray(stat, dtype=float)
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    p = np.ones(stat.shape[0])
    ok = np.isfinite(stat)
    if null.size and ok.any():
        ge = (null[None, :] >= stat[ok][:, None]).sum(axis=1)
        p[ok] = (1.0 + ge) / (1.0 + null.size)
    return p


def heavy_null_taxa(null_mat):
    """重零尾类群识别（混合池化否决）：逐类群零均值统计量
    > max(10, 5 × 全体零中位数) ⇒ 拟合病态（如 1004 taxon18：其全部置换
    统计量 >50 且观测统计量 ~1e5，池化会把病态判为显著）。
    这些类群不进池化零，改用逐类群置换 p（min p = 1/(K+1)，保守、安全）。"""
    K, p = null_mat.shape
    finite = np.isfinite(null_mat)
    pool_med = float(np.median(null_mat[finite])) if finite.any() else 1.0
    thr = max(10.0, 5.0 * pool_med)
    with np.errstate(invalid="ignore"):
        mean_j = np.array([np.mean(null_mat[finite[:, j], j])
                           if finite[:, j].any() else np.nan
                           for j in range(p)])
    heavy = np.isfinite(mean_j) & (mean_j > thr)
    return heavy, thr


def calibrated_test(Y, group, N, W=None, K=20, alpha=0.05, seed=20260305,
                    perms=None):
    """混合池化置换校准检验 + LKO 自洽 FWER。

    Returns dict(reject, pvals, qvals, fwer, lrt_obs, null (K,p), heavy,
                 alpha_hat, n_fallback_obs)
    """
    group = np.asarray(group, dtype=float)
    p = Y.shape[1]
    obs = abs_nb_lrt_stats(Y, group, N=N, W=W)
    lrt_obs = obs["lrt"]
    if perms is None:
        rng = np.random.default_rng(seed)
        perms = [rng.permutation(group) for _ in range(K)]
    K = len(perms)
    null_mat = np.full((K, p), np.nan)
    for k, gperm in enumerate(perms):
        null_mat[k] = abs_nb_lrt_stats(Y, gperm, N=N, W=W)["lrt"]
    heavy, _ = heavy_null_taxa(null_mat)
    pool = null_mat[:, ~heavy]
    null_pool = pool[np.isfinite(pool)]
    pvals = np.ones(p)
    ok = np.isfinite(lrt_obs) & ~heavy
    pvals[ok] = pooled_p(lrt_obs[ok], null_pool)
    # 重零尾类群：逐类群置换 p
    for j in np.where(heavy & np.isfinite(lrt_obs))[0]:
        nj = null_mat[np.isfinite(null_mat[:, j]), j]
        pvals[j] = (1.0 + (nj >= lrt_obs[j]).sum()) / (1.0 + len(nj))
    reject, qvals = _bh(pvals, alpha)
    # LKO 自洽 FWER（同套否决规则逐置换重算）
    fwer = 0.0
    for k in range(K):
        rest = null_mat[np.arange(K) != k]
        h_k, _ = heavy_null_taxa(rest)
        pk = np.ones(p)
        pool_k = rest[:, ~h_k]
        pool_k = pool_k[np.isfinite(pool_k)]
        okk = np.isfinite(null_mat[k]) & ~h_k
        pk[okk] = pooled_p(null_mat[k][okk], pool_k)
        for j in np.where(h_k & np.isfinite(null_mat[k]))[0]:
            nj = rest[np.isfinite(rest[:, j]), j]
            pk[j] = (1.0 + (nj >= null_mat[k, j]).sum()) / (1.0 + len(nj))
        fwer += float(_bh(pk, alpha)[0].any())
    fwer /= K
    return dict(reject=reject, pvals=pvals, qvals=qvals, fwer=float(fwer),
                lrt_obs=lrt_obs, null=null_mat, heavy=heavy,
                alpha_hat=obs["alpha_hat"], b1=obs["b1"],
                n_fallback_obs=int(obs["n_fallback"]))


def pool_diagnostic(null_mat, mu_hat, alpha_hat):
    """池化可交换性诊断（重零尾类群剔除后评估）。

    裁决规则（预注册）：pooling_ok = False 当且仅当
      (a) 逐类群零均值 SD > 1.5 × MC-SD（幅度必须超出 MC 噪声才有实际
          影响），且 (b) |r(零均值, log μ̂)| 或 |r(·, log α̂)| > 0.3
          且 p<0.01。重零尾类群由 heavy_null_taxa 事先否决（逐类群处置）。
    """
    K, p = null_mat.shape
    heavy, thr = heavy_null_taxa(null_mat)
    nm = null_mat.copy()
    nm[:, heavy] = np.nan
    valid = np.isfinite(nm).sum(axis=0) >= max(5, K // 2)
    per_taxon_mean = np.array([np.nanmean(nm[:, j]) if valid[j]
                               else np.nan for j in range(p)])
    out = dict(n_taxa_valid=int(valid.sum()), n_heavy=int(heavy.sum()),
               heavy_thr=float(thr),
               per_taxon_null_mean=float(np.nanmean(per_taxon_mean[valid])),
               per_taxon_null_mean_sd=float(np.nanstd(per_taxon_mean[valid])),
               mc_sd_expected=float(np.sqrt(2.0 / K)))
    v = valid & np.isfinite(mu_hat) & np.isfinite(alpha_hat) & (mu_hat > 0)
    if v.sum() >= 20:
        r_mu, p_mu = sstats.pearsonr(per_taxon_mean[v], np.log(mu_hat[v]))
        r_al, p_al = sstats.pearsonr(per_taxon_mean[v],
                                     np.log(np.clip(alpha_hat[v], 1e-9, None)))
        out.update(corr_nullmean_logmu=float(r_mu), p_logmu=float(p_mu),
                   corr_nullmean_logalpha=float(r_al), p_logalpha=float(p_al))
    pool = nm[np.isfinite(nm)]
    if pool.size:
        out.update(null_q99=float(np.quantile(pool, 0.99)),
                   chi2_q99=6.635, null_mean=float(pool.mean()))
    sd_ratio = out["per_taxon_null_mean_sd"] / out["mc_sd_expected"]
    out["sd_over_mc"] = float(sd_ratio)
    corr_bad = ((abs(out.get("corr_nullmean_logmu", 0)) > 0.3
                 and out.get("p_logmu", 1) < 0.01) or
                (abs(out.get("corr_nullmean_logalpha", 0)) > 0.3
                 and out.get("p_logalpha", 1) < 0.01))
    out["pooling_ok"] = bool(not (sd_ratio > 1.5 and corr_bad))
    return out
