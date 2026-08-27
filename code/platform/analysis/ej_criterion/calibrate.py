"""calibrate.py — 用 sweep_perrep.csv 校准三个候选判据并对比 ROC/一致性。

两层评估：
  Tier-1（rep 级，真实数据用法：一次拟合、逐类群判好坏）
    标签 good = 未撞 π 边界 且 |logit π̂ − logit π| ≤ 2（≈SE<1 目标的 95% 半宽）；
    各候选得分做 ROC/AUC + Youden 最优阈值。
  Tier-2（cell 级，模拟设计用法：跨重复可靠性）
    标签 reliable = sd(logit_err)≤1 且 |bias|≤0.5 且 撞界率≤0.1 且 cov95≥0.85；
    阈值可移植性检验：同一阈值在 clean 与 spike 两情景下是否同时最优/可用
    （e_j(N_min) 预期不可移植，I_j 预期可移植）。

另对既有 Exp 1–3 结果（calibration_perrep / ridge_perrep / ridge_phi2_perrep）
做外部验证：这些 cell 的 I_j 应分别大于/小于校准阈值。

输出：results/calibration_roc.csv, results/calibration_thresholds.csv,
      results/calibration_cells.csv, results/exp123_validation.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

sys.path.insert(0, "/mnt/agents/output/analysis/ej_criterion")
import criteria  # noqa: E402

BASE = Path(__file__).parent
OUT = BASE / "results"
EST = Path("/mnt/agents/output/code/estimation/results")


def roc_auc(score, good):
    """Mann–Whitney AUC：score 大 ⇒ good。"""
    s = np.asarray(score, float)
    g = np.asarray(good, bool)
    ok = np.isfinite(s)
    s, g = s[ok], g[ok]
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    # 并列取平均秩
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = r[order[i:j + 1]].mean()
        i = j + 1
    n1, n0 = g.sum(), (~g).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    return float((r[g].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def youden(score, good, grid=None):
    """在 score 网格上找 Youden J = sens + spec − 1 最大的阈值（score≥t ⇒ 判好）。"""
    s = np.asarray(score, float)
    g = np.asarray(good, bool)
    ok = np.isfinite(s)
    s, g = s[ok], g[ok]
    if grid is None:
        grid = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 200)))
    best = None
    P, Nn = g.sum(), (~g).sum()
    for t in grid:
        pred = s >= t
        tp = (pred & g).sum()
        fp = (pred & ~g).sum()
        sens, spec = tp / max(P, 1), 1 - fp / max(Nn, 1)
        J = sens + spec - 1
        if best is None or J > best[1]:
            best = (t, J, sens, spec)
    return best  # (threshold, J, sens, spec)


SCORES = ["se_alpha", "e_j_min", "e_j_q10", "e_j_mean", "n_e_j_mean", "I_j"]


def main():
    df = pd.read_csv(OUT / "sweep_perrep.csv")
    df["good"] = (~df["on_bnd_pi"]) & (df["logit_err"].abs() <= 2.0)

    # ---- Tier-1: rep 级 ROC（分臂 × 分情景 + 合并） ----
    rows = []
    for (arm, scen), sub in df.groupby(["arm", "scenario"]):
        for sc in SCORES:
            ls = (np.log10(sub[sc].clip(lower=1e-12)) if sc != "se_alpha"
                  else -np.log10(sub[sc].clip(lower=1e-12)))
            auc = roc_auc(ls, sub["good"])
            t, J, se, sp = youden(ls, sub["good"])
            rows.append({"tier": "rep", "arm": arm, "scenario": scen,
                         "criterion": sc, "auc": auc,
                         "youden_log10": t if sc != "se_alpha" else -t,
                         "youden": 10 ** t if sc != "se_alpha" else 10 ** (-t),
                         "sens": se, "spec": sp, "J": J})
    for arm, sub in df.groupby("arm"):
        for sc in SCORES:
            ls = (np.log10(sub[sc].clip(lower=1e-12)) if sc != "se_alpha"
                  else -np.log10(sub[sc].clip(lower=1e-12)))
            auc = roc_auc(ls, sub["good"])
            t, J, se, sp = youden(ls, sub["good"])
            rows.append({"tier": "rep", "arm": arm, "scenario": "pooled",
                         "criterion": sc, "auc": auc,
                         "youden_log10": t if sc != "se_alpha" else -t,
                         "youden": 10 ** t if sc != "se_alpha" else 10 ** (-t),
                         "sens": se, "spec": sp, "J": J})
    roc_df = pd.DataFrame(rows)

    # ---- Tier-2: cell 级 ----
    def cell_stats(sub):
        return pd.Series({
            "R": len(sub),
            "sd_logit": sub["logit_err"].std(ddof=1),
            "bias_logit": sub["logit_err"].mean(),
            "frac_bnd_pi": sub["on_bnd_pi"].mean(),
            "cov95": sub["cov95"].mean(),
            "se_alpha_med": sub["se_alpha"].median(),
            "e_j_min_med": sub["e_j_min"].median(),
            "e_j_q10_med": sub["e_j_q10"].median(),
            "e_j_mean_med": sub["e_j_mean"].median(),
            "n_e_j_mean_med": sub["n_e_j_mean"].median(),
            "I_j_med": sub["I_j"].median(),
        })
    cells = (df.groupby(["arm", "scenario", "taxon", "pi_true", "phitheta"])
               .apply(cell_stats, include_groups=False).reset_index())
    cells["reliable"] = ((cells["sd_logit"] <= 1.0)
                         & (cells["bias_logit"].abs() <= 0.5)
                         & (cells["frac_bnd_pi"] <= 0.1)
                         & (cells["cov95"] >= 0.85))
    crows = []
    for (arm, scen), sub in cells.groupby(["arm", "scenario"]):
        for sc in ["se_alpha_med", "e_j_min_med", "e_j_q10_med",
                   "e_j_mean_med", "n_e_j_mean_med", "I_j_med"]:
            ls = (np.log10(sub[sc].clip(lower=1e-15))
                  if sc != "se_alpha_med"
                  else -np.log10(sub[sc].clip(lower=1e-15)))
            auc = roc_auc(ls, sub["reliable"])
            t, J, se, sp = youden(ls, sub["reliable"])
            crows.append({"tier": "cell", "arm": arm, "scenario": scen,
                          "criterion": sc, "auc": auc,
                          "youden_log10": (t if sc != "se_alpha_med" else -t),
                          "youden": (10 ** t if sc != "se_alpha_med"
                                     else 10 ** (-t)),
                          "sens": se, "spec": sp, "J": J})
    croc = pd.DataFrame(crows)
    roc_all = pd.concat([roc_df, croc], ignore_index=True)
    roc_all.to_csv(OUT / "calibration_roc.csv", index=False)
    cells.to_csv(OUT / "calibration_cells.csv", index=False)

    # ---- 阈值可移植性：spike 情景下用 clean 情景的 Youden 阈值的表现 ----
    port = []
    for arm in df["arm"].unique():
        sub = df[df["arm"] == arm]
        for sc in SCORES:
            ls = (np.log10(sub[sc].clip(lower=1e-12)) if sc != "se_alpha"
                  else -np.log10(sub[sc].clip(lower=1e-12)))
            t, J, se, sp = youden(ls[sub["scenario"] == "clean"],
                                  sub.loc[sub["scenario"] == "clean", "good"])
            # 只对有 spike 的臂有意义
            if (sub["scenario"] == "spike").any():
                pred = ls[sub["scenario"] == "spike"] >= t
                gg = sub.loc[sub["scenario"] == "spike", "good"].to_numpy()
                acc = float((pred.to_numpy() == gg).mean())
                port.append({"arm": arm, "criterion": sc,
                             "clean_threshold_log10": t,
                             "spike_acc_at_clean_threshold": acc})
    pd.DataFrame(port).to_csv(OUT / "calibration_portability.csv", index=False)

    # ---- 固定阈值评估（推荐阈值在各情景下的表现；含边界条件） ----
    fixed = [("a_se<1+bnd", (df["se_alpha"] < 1.0) & ~df["on_bnd_pi"]),
             ("c_Ij>=1+bnd", (df["I_j"] >= 1.0) & ~df["on_bnd_pi"]),
             ("c_Ij>=1_nobnd", df["I_j"] >= 1.0),
             ("b1_eq10>=0.45", df["e_j_q10"] >= 0.45),
             ("b2_emean>=1.85", df["e_j_mean"] >= 1.85),
             ("ej_min>=0.34", df["e_j_min"] >= 0.34)]
    frows = []
    for name, pred in fixed:
        for (arm, scen), sub in df.groupby(["arm", "scenario"]):
            pr = pred.loc[sub.index].to_numpy()
            gg = sub["good"].to_numpy()
            P, Nn = gg.sum(), (~gg).sum()
            frows.append({"rule": name, "arm": arm, "scenario": scen,
                          "acc": float((pr == gg).mean()),
                          "sens": float((pr & gg).sum() / max(P, 1)),
                          "spec": float(1 - (pr & ~gg).sum() / max(Nn, 1))})
    fdf = pd.DataFrame(frows)
    fdf.to_csv(OUT / "calibration_fixed_rules.csv", index=False)

    # ---- Exp 1–3 外部验证（既有 CSV 的 cell 级 I_j 与判据通过情况） ----
    rng = np.random.default_rng(7)
    vrows = []
    # Exp 1 cells (φ=3000, 可识别, 应通过)：I_j 用 200 次深度抽样的均值
    cal = pd.read_csv(EST / "calibration_summary.csv")
    cal = cal[(cal["arm"] == "phi_known") & (cal["taxon"] != "phi")]
    for _, r in cal.iterrows():
        Is = []
        for _ in range(50):
            Nd = np.exp(rng.uniform(np.log(1e3), np.log(1e5), 200)).astype(int)
            Is.append(criteria.fisher_per_taxon(r["pi_true"], r["theta_true"],
                                                3000.0, Nd)[1])
        vrows.append({"exp": "Exp1_identifiable", "taxon": int(r["taxon"]),
                      "pi_true": r["pi_true"], "theta_true": r["theta_true"],
                      "phi": 3000.0, "I_j_mean": float(np.mean(Is)),
                      "reliable_observed": bool(
                          r["cov95_logit_pi"] >= 0.9
                          and abs(r["logit_pi_bias"]) <= 0.5),
                      "cov95_obs": r["cov95_logit_pi"]})
    # Exp 2 ridge cells (φ=10 / φ=2, 脊区, 应不通过)
    for tag, phi_r, fname in [("Exp2_ridge_phi10", 10.0, "ridge_perrep.csv"),
                              ("Exp2_ridge_phi2", 2.0, "ridge_phi2_perrep.csv")]:
        rp = pd.read_csv(EST / fname)
        for j, pi_t in enumerate([0.85, 0.90, 0.95, 0.97]):
            lp = logit(np.clip(rp[f"pi_{j}"].values, 1e-9, 1 - 1e-9))
            bnd = float(((rp[f"pi_{j}"] > 0.999)
                         | (rp[f"pi_{j}"] < 1.1e-4)).mean())
            Is = []
            for _ in range(50):
                Nd = np.exp(rng.uniform(np.log(1e3), np.log(1e5), 200)).astype(int)
                Is.append(criteria.fisher_per_taxon(pi_t, 1e-3, phi_r, Nd)[1])
            vrows.append({"exp": tag, "taxon": j, "pi_true": pi_t,
                          "theta_true": 1e-3, "phi": phi_r,
                          "I_j_mean": float(np.mean(Is)),
                          "reliable_observed": bool(
                              lp.std(ddof=1) <= 1.0 and bnd <= 0.1),
                          "cov95_obs": np.nan,
                          "sd_logit_obs": lp.std(ddof=1), "frac_bnd": bnd})
    vdf = pd.DataFrame(vrows)
    vdf.to_csv(OUT / "exp123_validation.csv", index=False)

    print(roc_all.to_string(index=False))
    print("\n== fixed rules ==")
    print(fdf.to_string(index=False))
    print("\n== portability ==")
    print(pd.DataFrame(port).to_string(index=False))
    print("\n== exp123 validation ==")
    print(vdf.to_string(index=False))


if __name__ == "__main__":
    main()
