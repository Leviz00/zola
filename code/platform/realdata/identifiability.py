"""identifiability.py — 可识别性分析（任务 3）。

判据（fix_N1 未给数值阈值，采用 SE 判据并在报告中说明）：
  可识别区：Godambe SE(logit π̂) < 1 且未撞参数边界；
  脊区    ：SE(logit π̂) ≥ 1 或撞边界（此时 π̂–θ̂ 沿 πθ̄=const 脊退化）。
SE<1 对应 logit 尺度上 π̂ 的 95% 区间半宽 <2，即 π 的点估计不被检出指示
噪声淹没的实操门槛。

e_j = φ̂ θ̄_j [ψ(φ̂+N_min)−ψ(φ̂)]（fix_N1 remark 的有效一阶检出强度）。
另拟合 P(可识别) 对 log10 e_j 的 logistic 曲线，给出经验跨界 e*（P=0.5），
作为未来以 e_j 本身为操作阈值的标定。

脊区验证：逐类群 Godambe corr(α̂_j, β̂_j)（模拟中跨重复 corr≈−0.94 的
经验对应量——单次拟合内该参数的抽样相关），报告脊区中位数。

用法：python3 identifiability.py ibdmdb [mbqc agp]
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize

RES = Path("/mnt/agents/output/realdata/results")
SE_CUT = 1.0


def logistic_crossover(x, y):
    """y~Bern(logistic(a+b*x)) MLE，返回 P(y=1)=0.5 的 x*。"""
    def nll(ab):
        z = np.clip(ab[0] + ab[1] * x, -30, 30)
        p = 1 / (1 + np.exp(-z))
        return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    r = minimize(nll, [2.0, 1.0], method="Nelder-Mead")
    a, b = r.x
    return -a / b if abs(b) > 1e-8 else np.nan, a, b


def run(name):
    df = pd.read_csv(RES / f"fit_{name}_pertaxon.csv")
    df["on_boundary"] = df["on_boundary_pi"] | df["on_boundary_theta"]
    df["zone"] = np.where((df["se_logit_pi"] < SE_CUT) & ~df["on_boundary"],
                          "identifiable", "ridge")
    ident = (df["zone"] == "identifiable").astype(int).to_numpy()
    le = np.log10(np.maximum(df["e_j"].to_numpy(), 1e-12))
    xstar, a, b = logistic_crossover(le, ident)
    df["p_identifiable_logistic"] = 1 / (1 + np.exp(-(a + b * le)))
    df.to_csv(RES / f"identifiability_{name}.csv", index=False)

    ridge = df[df["zone"] == "ridge"]
    summ = {
        "dataset": name,
        "p": len(df),
        "n_identifiable": int(ident.sum()),
        "n_ridge": int((1 - ident).sum()),
        "frac_identifiable": float(ident.mean()),
        "n_boundary": int(df["on_boundary"].sum()),
        "e_j_median": float(df["e_j"].median()),
        "e_j_q10": float(df["e_j"].quantile(0.1)),
        "e_j_q90": float(df["e_j"].quantile(0.9)),
        "e_crossover_log10": float(xstar),
        "e_crossover": float(10 ** xstar),
        "ridge_corr_median": float(ridge["godambe_corr_alpha_beta"].median())
        if len(ridge) else np.nan,
        "ridge_corr_q25": float(ridge["godambe_corr_alpha_beta"].quantile(.25))
        if len(ridge) else np.nan,
        "ident_corr_median": float(
            df.loc[df["zone"] == "identifiable",
                   "godambe_corr_alpha_beta"].median()),
        "sim_reference_corr": -0.94,
    }
    pd.DataFrame([summ]).to_csv(RES / f"identifiability_{name}_summary.csv",
                                index=False)
    print(f"[{name}] identifiable {ident.sum()}/{len(df)} "
          f"({ident.mean():.1%}), e*={10**xstar:.3g}, "
          f"ridge corr median={summ['ridge_corr_median']:.3f}", flush=True)


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["ibdmdb"]):
        run(name)
