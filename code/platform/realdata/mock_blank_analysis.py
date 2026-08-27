"""mock_blank_analysis.py — MBQC mock 真值验证与 blank 污染谱（任务 2）。

输入：data/mbqc_mockblank_genus.npz（aggregate.py 产物，4,670 样本 × 717 属，
未做流行率过滤）+ mbqc_sample_metadata.csv 的 specimen_type 列。

真值：MOESM11 Supplementary Table 1（pdftotext -layout 解析，本文件内硬编码
并附核对注释）：
  - 口腔人工群落 22 株 → 22 个属，全部在 717 属表中匹配；
  - 肠道人工群落 20 株 → 19 个属，16 个直接匹配；Alistipes /
    Subdoligranulum / Synergistes 在 Greengenes 13.5 行名中无 g__ 标签
    （grep 核实：无 Alistipes/Subdoligranulum；Synergistetes 门下仅
    Pyramidobacter），被聚合进 unclassified_<科> 桶，属级不可验证，
    从 present 标签集剔除（敏感性：同时剔除对应 unclassified 科桶）。

方法：每个 mock 类型单独做共享 φ 的复合似然——φ 剖面网格扫描（复合似然在
φ 固定时按类群可分解为 2 维问题，estimation/README profile 路径），取剖面
对数似然最大者，再做局部加密；最终逐类群 MLE + Godambe SE（φ 固定）。
零来源后验：P(Z_ij=0 | Y_ij=0, N_i) = (1-π_j)/[(1-π_j)+π_j g(N_i;θ_j,φ̂)]。

产出（results/）：
  funnel_mbqc_mockblank.csv
  mbqc_mock_fit_{fecal,oral}.csv     逐类群 (π̂,θ̂,SE,e_j,present 标签,...)
  mbqc_mock_auc.csv                  属级/细胞级 AUC
  mbqc_blank_contamination.csv       blank 污染谱 + 与 mock 交叉
  mbqc_mock_posterior_cells.npz      F4 绘图用细胞级后验（抽样）
  mbqc_mock_depthcurve.npz           F3 mock 检出-深度曲线数据
全链路确定性：无 RNG（绘图抽样用固定 seed=7）。
"""

from __future__ import annotations

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "/mnt/agents/output/code/estimation")
from composite_likelihood import (_fit_single_taxon_phi_known,
                                  godambe_covariance, detection_indicators)
from model import g_closed, effective_detection_strength
from scipy.stats import spearmanr

ROOT = Path("/mnt/agents/output/realdata")
DATA, RES = ROOT / "data", ROOT / "results"
DS = Path("/mnt/agents/output/datasets")

# --- MOESM11 Supplementary Table 1（pdftotext 解析后人工核对，22+20 株） ---
ORAL_GENERA = ["Bacillus", "Bifidobacterium", "Rothia", "Parvimonas",
               "Mogibacterium", "Campylobacter", "Eggerthella", "Slackia",
               "Klebsiella", "Fusobacterium", "Leptotrichia", "Weissella",
               "Eikenella", "Neisseria", "Tannerella", "Prevotella",
               "Capnocytophaga", "Granulicatella", "Streptococcus", "Gemella",
               "Dialister", "Veillonella"]                     # 22/22 匹配
GUT_GENERA = ["Coprobacillus", "Bifidobacterium", "Collinsella", "Bilophila",
              "Escherichia", "Fusobacterium", "Anaerostipes", "Clostridium",
              "Lactobacillus", "Pediococcus", "Ralstonia", "Paenibacillus",
              "Parabacteroides", "Bacteroides", "Propionibacterium",
              "Enterococcus"]  # 16/19 匹配（缺 Alistipes/Subdoligranulum/
                               # Synergistes，见模块 docstring）
GUT_UNVERIFIABLE = ["Alistipes", "Subdoligranulum", "Synergistes"]
# 对应科桶（敏感性分析时从 absent 标签剔除，避免标签噪声）
GUT_AMBIGUOUS_BUCKETS = ["unclassified_Rikenellaceae",
                         "unclassified_Ruminococcaceae",
                         "unclassified_Dethiosulfovibrionaceae"]


def mw_auc(scores, labels):
    """Mann-Whitney AUC：P(score_positive > score_negative)（ties=0.5）。"""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    sp, sn = s[y], s[~y]
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    # 平均秩处理 ties
    ss = s[order]
    i = 0
    r = np.arange(1, len(s) + 1, dtype=float)
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = r[i:j + 1].mean()
        i = j + 1
    rp = ranks[y].sum()
    np_ = y.sum(); nn_ = (~y).sum()
    return float((rp - np_ * (np_ + 1) / 2) / (np_ * nn_))


def _quick_fit(d, N, phi):
    """单起点 2 维拟合（φ 剖面扫描用；最终 MLE 仍用 3 起点版本）。"""
    from scipy.optimize import minimize
    from scipy.special import logit as _logit
    from composite_likelihood import _neg_loglik_grad
    from model import g_closed as _gc
    mean_det = float(d.mean())
    theta0 = 2e-3
    amp = float((1.0 - _gc(N, theta0, phi)).mean())
    pi0 = float(np.clip(mean_det / max(amp, 1e-6), 1e-3, 0.999))
    s = np.array([_logit(pi0), np.log(theta0)])
    r = minimize(_neg_loglik_grad, s, args=(d[:, None], N, phi),
                 method="L-BFGS-B", jac=True,
                 bounds=[(_logit(1e-4), _logit(0.9999)),
                         (np.log(1e-7), np.log(0.9))],
                 options={"maxiter": 100, "ftol": 1e-12, "gtol": 1e-8})
    return r.fun


def fit_mock(D, N, taxa_detected_mask, phi_grid_coarse, tag):
    """共享 φ 剖面扫描（单起点快速）+ 最终逐类群 3 起点 MLE。返回 dict。"""
    t0 = time.time()
    idx_det = np.where(taxa_detected_mask)[0]
    ndet = len(idx_det)

    def profile_ll(phi):
        tot = 0.0
        for j in idx_det:
            tot += -_quick_fit(D[:, j], N, phi)
        return tot

    # 粗扫（若最优在上界则向外扩展探测，直至 1e6 或似然回落）
    grid = list(phi_grid_coarse)
    lls_l = [profile_ll(phi) for phi in grid]
    while int(np.argmax(lls_l)) == len(grid) - 1 and grid[-1] < 1e6:
        new_phi = min(grid[-1] * 3.0, 1e6)
        grid.append(new_phi)
        lls_l.append(profile_ll(new_phi))
        print(f"  [{tag}] phi at grid top, extend probe "
              f"phi={new_phi:.3g} ll={lls_l[-1]:.3f}", flush=True)
    k = int(np.argmax(lls_l))
    phi0 = grid[k]
    phi_on_top = (k == len(grid) - 1) and grid[-1] >= 1e6
    # 局部加密（对数等距 ±半步）
    lo = np.log10(grid[max(k - 1, 0)])
    hi = np.log10(grid[min(k + 1, len(grid) - 1)])
    fine = np.logspace(lo, hi, 5)
    lls_f = np.array([profile_ll(phi) for phi in fine])
    phi_hat = float(fine[int(np.argmax(lls_f))])
    print(f"  [{tag}] phi scan: coarse best={phi0:.3g} "
          f"-> refined phi_hat={phi_hat:.4g} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # 最终逐类群 MLE + Godambe（φ 固定，2p 参数）
    p = D.shape[1]
    pi = np.full(p, 1e-4)          # 未检出类群：π 撞下界（似然最优方向）
    theta = np.full(p, np.nan)
    se_a = np.full(p, np.nan)
    se_b = np.full(p, np.nan)
    boundary_pi = np.ones(p, dtype=bool)   # 未检出者标边界
    boundary_th = np.zeros(p, dtype=bool)
    succ = np.ones(p, dtype=bool)    # 未检出类群不参与拟合，默认 True
    for j in idx_det:
        r = _fit_single_taxon_phi_known(D[:, j], N, phi_hat)
        a, b = r.x
        pi[j] = 1.0 / (1.0 + np.exp(-a))
        theta[j] = np.exp(b)
        boundary_pi[j] = (abs(a - (-9.21024036697707)) < 1e-6
                          or abs(a - 9.21024036697707) < 1e-6)
        boundary_th[j] = (abs(b - np.log(1e-7)) < 1e-6
                          or abs(b - np.log(0.9)) < 1e-6)
        succ[j] = bool(r.success)
    fail_det = int((~succ[idx_det]).sum())
    # Godambe（仅检出类群子矩阵，φ 固定）
    Dd = D[:, idx_det]
    psi = np.concatenate([
        np.log(pi[idx_det] / (1 - pi[idx_det])),
        np.log(theta[idx_det])])
    V_god, V_naive, A, B = godambe_covariance(psi, Dd, N, phi_known=phi_hat)
    sd = np.sqrt(np.maximum(np.diag(V_god), 0.0))
    se_a[idx_det] = sd[:ndet]
    se_b[idx_det] = sd[ndet:]
    print(f"  [{tag}] fitted {ndet} detected taxa, "
          f"fail={fail_det} ({time.time() - t0:.0f}s)", flush=True)
    return dict(phi_hat=phi_hat, pi=pi, theta=theta, se_alpha=se_a,
                se_beta=se_b, boundary_pi=boundary_pi,
                boundary_theta=boundary_th, success=succ,
                n_detected_taxa=ndet, phi_on_grid_top=phi_on_top)


def zero_source_posterior(D, N, pi, theta, phi):
    """对 D=0 的细胞计算 P(structural zero | Y=0, N)（向量化）。"""
    lg = g_closed(N[:, None].astype(float),
                  np.nan_to_num(theta, nan=1e-9)[None, :], phi)
    num = 1.0 - pi[None, :]
    den = num + pi[None, :] * lg
    with np.errstate(divide="ignore", invalid="ignore"):
        post = num / den
    return np.where(D == 0, post, np.nan)


def main():
    z = np.load(DATA / "mbqc_mockblank_genus.npz", allow_pickle=True)
    Y, taxa, samples = z["Y"].astype(np.int64), z["taxa"].astype(str), z["samples"].astype(str)
    meta = pd.read_csv(DS / "mbqc/mbqc_sample_metadata.csv",
                       usecols=["Unnamed: 0", "specimen_type"])
    stype = meta.set_index("Unnamed: 0")["specimen_type"].loc[samples].to_numpy()
    depths = Y.sum(axis=1)

    funnel = [("raw_mockblank", Y.shape[0], Y.shape[1],
               "blank + Fecal/Oral artificial colony + Robogut (Robogut 本分析不用)")]

    # ---------------- blank 污染谱 ----------------
    mb = stype == "blank"
    mb_pos = mb & (depths > 0)
    Yb = Y[mb_pos]
    blank_prev = (Yb > 0).mean(axis=0)
    blank_tot = Yb.sum(axis=0)
    funnel.append(("blank_depth_gt0", int(mb_pos.sum()), Y.shape[1],
                   f"dropped {int((mb & (depths == 0)).sum())} zero-depth blanks; "
                   f"blank depth median={int(np.median(depths[mb_pos]))}"))

    # ---------------- mock 拟合（fecal/oral） ----------------
    present_map = {"Fecal artificial colony": set(GUT_GENERA),
                   "Oral artificial colony": set(ORAL_GENERA)}
    tag_map = {"Fecal artificial colony": "fecal",
               "Oral artificial colony": "oral"}
    phi_grid = np.logspace(np.log10(0.5), 5, 8)
    auc_rows = []
    post_cells = {}
    curve_data = {}
    for st, present in present_map.items():
        tag = tag_map[st]
        m = (stype == st) & (depths > 0)
        Ym, Nm = Y[m], depths[m].astype(float)
        funnel.append((f"mock_{tag}_depth_gt0", int(m.sum()), Y.shape[1],
                       f"dropped {int(((stype == st) & (depths == 0)).sum())} "
                       f"zero-depth; depth min/median/max = "
                       f"{int(Nm.min())}/{int(np.median(Nm))}/{int(Nm.max())}"))
        D = detection_indicators(Ym)
        det_mask = Ym.sum(axis=0) > 0
        fit = fit_mock(D, Nm, det_mask, phi_grid, tag)
        phi_hat = fit["phi_hat"]
        e_j = effective_detection_strength(
            np.nan_to_num(fit["theta"], nan=0.0), phi_hat, Nm.min())
        prev = (Ym > 0).mean(axis=0)
        labels = np.array([t in present for t in taxa])
        df = pd.DataFrame({
            "taxon": taxa, "known_present": labels,
            "pi_hat": fit["pi"], "theta_hat": fit["theta"],
            "se_logit_pi": fit["se_alpha"], "se_log_theta": fit["se_beta"],
            "e_j": e_j, "prevalence": prev,
            "n_detected": (Ym > 0).sum(axis=0),
            "total_count": Ym.sum(axis=0),
            "on_boundary_pi": fit["boundary_pi"],
            "on_boundary_theta": fit["boundary_theta"],
            "ever_detected": det_mask})
        df.to_csv(RES / f"mbqc_mock_fit_{tag}.csv", index=False)

        # ---- 属级 AUC（score = π̂） ----
        sc = df["pi_hat"].to_numpy()
        auc_pi = mw_auc(sc, labels)
        # 敏感性：剔除模糊科桶
        if tag == "fecal":
            keep = ~np.isin(taxa, GUT_AMBIGUOUS_BUCKETS)
            auc_pi_sens = mw_auc(sc[keep], labels[keep])
        else:
            auc_pi_sens = np.nan
        # ---- 零来源后验（细胞级 + 属级均值） ----
        post = zero_source_posterior(D, Nm, fit["pi"], fit["theta"], phi_hat)
        # 属级：score = 1 - mean posterior over zero cells（无零细胞的属→1）
        with np.errstate(invalid="ignore"):
            mpost = np.nanmean(post, axis=0)
        mpost = np.where(np.isnan(mpost), 0.0, mpost)  # 全部检出→无零→score 1
        auc_post = mw_auc(1.0 - mpost, labels)
        # 细胞级 AUC：present 属的零细胞 vs absent 属的零细胞
        cells_p = post[:, labels][~np.isnan(post[:, labels])]
        cells_a = post[:, ~labels][~np.isnan(post[:, ~labels])]
        auc_cell = mw_auc(np.concatenate([cells_p, cells_a]),
                          np.concatenate([np.zeros(len(cells_p), bool),
                                          np.ones(len(cells_a), bool)]))
        # 注：score 为后验本身，present=0 类；AUC = P(post_absent > post_present)
        auc_rows.append({
            "mock": tag, "n_samples": int(m.sum()), "phi_hat": phi_hat,
            "phi_on_grid_top": bool(fit["phi_on_grid_top"]),
            "N_min": Nm.min(), "N_median": float(np.median(Nm)),
            "N_max": Nm.max(),
            "n_present_labeled": int(labels.sum()),
            "n_present_detected": int((labels & det_mask).sum()),
            "auc_pi": auc_pi, "auc_pi_sensitivity_excl_ambiguous": auc_pi_sens,
            "auc_genus_mean_posterior": auc_post,
            "auc_cell_posterior": auc_cell,
            "n_zero_cells_present": len(cells_p),
            "n_zero_cells_absent": len(cells_a),
            "median_posterior_present": float(np.median(cells_p)),
            "median_posterior_absent": float(np.median(cells_a)),
            "n_taxa_fit_failed": int((~fit["success"][det_mask]).sum())})
        # F4 绘图数据：固定 seed 抽样存盘
        rng = np.random.default_rng(7)
        sp = rng.choice(cells_p, min(len(cells_p), 60000), replace=False)
        sa = rng.choice(cells_a, min(len(cells_a), 60000), replace=False)
        post_cells[f"{tag}_present"] = sp
        post_cells[f"{tag}_absent"] = sa

        # ---- F3 mock 检出-深度曲线数据（代表性高丰度 present 属） ----
        rep = (df[df["known_present"] & df["ever_detected"]]
               .sort_values("total_count", ascending=False).head(3)["taxon"]
               .tolist())
        edges = np.unique(np.quantile(np.log10(Nm),
                                      np.linspace(0, 1, 13)))
        cents, emp, cnt = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            mm = (np.log10(Nm) >= a) & (np.log10(Nm) < b)
            if mm.sum() >= 5:
                cents.append(float(10 ** ((a + b) / 2)))
                cnt.append(int(mm.sum()))
                emp.append(D[mm][:, [list(taxa).index(t) for t in rep]]
                           .mean(axis=0).tolist())
        curve_data[tag] = dict(
            rep=rep, centers=np.array(cents), emp=np.array(emp), cnt=np.array(cnt),
            theo=np.column_stack([
                df.set_index("taxon").loc[t, "pi_hat"]
                * (1 - g_closed(np.array(cents),
                                df.set_index("taxon").loc[t, "theta_hat"],
                                phi_hat)) for t in rep]),
            Nm=Nm)

    pd.DataFrame(auc_rows).to_csv(RES / "mbqc_mock_auc.csv", index=False)
    np.savez_compressed(RES / "mbqc_mock_posterior_cells.npz", **post_cells)
    np.savez_compressed(
        RES / "mbqc_mock_depthcurve.npz",
        **{f"{tag}_{k}": v for tag, cd in curve_data.items()
           for k, v in cd.items() if k != "rep"},
        **{f"{tag}_rep": np.array(cd["rep"]) for tag, cd in curve_data.items()})

    # ---------------- blank × mock 交叉 ----------------
    mf = (stype == "Fecal artificial colony") & (depths > 0)
    mo = (stype == "Oral artificial colony") & (depths > 0)
    fec_prev = (Y[mf] > 0).mean(axis=0)
    ora_prev = (Y[mo] > 0).mean(axis=0)
    is_member = np.array([(t in GUT_GENERA) or (t in ORAL_GENERA) for t in taxa])
    rho_f, pv_f = spearmanr(blank_prev, fec_prev)
    rho_o, pv_o = spearmanr(blank_prev, ora_prev)
    bdf = pd.DataFrame({
        "taxon": taxa, "blank_prevalence": blank_prev,
        "blank_total_count": blank_tot,
        "fecal_mock_prevalence": fec_prev, "oral_mock_prevalence": ora_prev,
        "is_table1_member": is_member})
    bdf = bdf.sort_values(["blank_prevalence", "blank_total_count"],
                          ascending=False)
    bdf.to_csv(RES / "mbqc_blank_contamination.csv", index=False)
    funnel.append(("blank_contamination_spectrum", int(mb_pos.sum()),
                   int((blank_prev > 0).sum()),
                   f"genera ever detected in blanks; spearman(blank,fecal mock)="
                   f"{rho_f:.3f} (p={pv_f:.1e}), spearman(blank,oral mock)="
                   f"{rho_o:.3f} (p={pv_o:.1e})"))
    pd.DataFrame(funnel, columns=["step", "n_samples", "n_taxa", "note"]
                 ).to_csv(RES / "funnel_mbqc_mockblank.csv", index=False)
    print(pd.DataFrame(auc_rows).to_string(index=False), flush=True)
    print(funnel[-1], flush=True)


if __name__ == "__main__":
    main()
