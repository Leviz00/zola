"""validate_v323.py — v3.2a/v3.2b 验收 + v3.3 冒烟（产物 -> method_fix/v3/）。

Part A v3.2a 绝对效应生成模式验收：
  A1 契约断言：4 机制 × 2 模式 generate()（内部断言结构性零 Y==0）。
  A2 判据(i)：非 DA 类群 E[Y|N]/N 两组差异——精确层（theta_bar 相等）+
     经验层（R=10 × n=300 合并，均值>0.01 类群的相对差，vs legacy 对照）。
  A3 判据(ii)：相对丰度挤压保留（case/ctrl 相对比例中位 < 1，uplift 实测）。
  A4 κ 近似误差：effect=1 时 absolute(BB κ=φ/(1−θ̃)) vs legacy(DM 边缘) 的
     逐类群 var(Y/N) 比（R=20），加理论比值。
Part B v3.2b abs_nb_glm 验收：
  B1 全局 null FDR（R=20，three_layer + zinb）。
  B2 absolute DA 格 oracle 掩码 FDR/TPR（bridge s8 R=20、s9 R=20）。
Part C v3.3 冒烟：10 补充格 × 3 rep 双轨生成 + 每格 1 rep abs_nb_glm。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

SIM_V3 = "/mnt/agents/output/code/simulation_v3"
OUT = "/mnt/agents/output/analysis/method_fix/v3"
sys.path.insert(0, SIM_V3)
import design  # noqa: E402
import generators  # noqa: E402
from abs_glm import abs_nb_glm  # noqa: E402

P = 100


def gen(mech, prm, n, depth, seed):
    return generators.generate(mech, prm, n=n, p=P, depths=depth, seed=seed)


def part_a(log):
    rows = []
    base_prm = dict(effect_size=2.0, structural_zero_rate=0.1,
                    informative_zeros=False, dispersion=15.0, da_fraction=0.1,
                    depth_cv=0.3)
    # A1 契约断言（generate 内部断言会在违规时抛错）
    for mech in ["three_layer", "zinb", "zigdm_like", "beta_binomial"]:
        for mode in ["legacy", "absolute"]:
            pr = dict(base_prm, effect_mode=mode)
            gen(mech, pr, 50, 20000, 123)
    log.append("A1 contract asserts: PASS (4 mechanisms x 2 modes)")

    # A2 判据(i)
    for mech in ["three_layer", "zinb", "zigdm_like", "beta_binomial"]:
        for mode in ["absolute", "legacy"]:
            # 精确层：theta_bar 基线两组差（absolute 应恒 0；legacy 显示挤压）
            pr = dict(base_prm, effect_mode=mode)
            _, t = gen(mech, pr, 50, 20000, 7)
            exact = "n/a"
            if "theta_bar_case" in t:
                exact = float(np.abs(t["theta_bar_case"]
                                     - t["theta_bar_control"]).max())
            diffs, scales, uplift = [], [], []
            for rep in range(10):
                pr = dict(base_prm, effect_mode=mode)
                Y, t = gen(mech, pr, 300, 20000, 900 + rep)
                N = t["depths"].astype(float)
                g = t["group"]
                nd = ~t["da_taxa"]
                x = Y / N[:, None]
                diffs.append(x[g == 1][:, nd].mean(0)
                             - x[g == 0][:, nd].mean(0))
                scales.append(x[:, nd].mean(0))
                uplift.append(Y[g == 1].sum(1).mean()
                              / Y[g == 0].sum(1).mean())
            D = np.mean(diffs, 0)
            sc = np.mean(scales, 0)
            big = sc > np.quantile(sc, 0.5)  # 较丰半数类群
            rows.append(dict(part="A2", mechanism=mech, mode=mode,
                             exact_theta_bar_maxdiff=exact,
                             n_taxa_big=int(big.sum()),
                             med_rel_diff=float(np.median(D[big] / sc[big])),
                             max_abs_rel=float(np.abs(D[big] / sc[big]).max()),
                             frac_rel_gt5=float((np.abs(D[big] / sc[big])
                                                 > 0.05).mean()),
                             frac_neg=float((D < 0).mean()),
                             uplift=float(np.mean(uplift))))

    # A3 判据(ii) 相对挤压保留（含在 A2 行 uplift/相对比例中体现；单独测）
    for mode in ["absolute", "legacy"]:
        ratios = []
        for rep in range(10):
            pr = dict(base_prm, effect_mode=mode)
            Y, t = gen("three_layer", pr, 300, 20000, 1900 + rep)
            N = t["depths"].astype(float)
            g = t["group"]
            nd = ~t["da_taxa"]
            x2 = Y / Y.sum(axis=1, keepdims=True)
            a = x2[g == 1][:, nd].mean(0)
            b = x2[g == 0][:, nd].mean(0)
            keep = b > np.quantile(b, 0.5)
            ratios.append(np.median(a[keep] / b[keep]))
        rows.append(dict(part="A3", mechanism="three_layer", mode=mode,
                         median_rel_ratio=float(np.mean(ratios))))

    # A4 κ 近似误差（effect=1：两模式边缘模型应一致）
    for mech in ["three_layer", "zigdm_like"]:
        va, vl = [], []
        for rep in range(20):
            for mode, acc in (("absolute", va), ("legacy", vl)):
                pr = dict(base_prm, effect_mode=mode, effect_size=1.0)
                Y, t = gen(mech, pr, 200, 20000, 2900 + rep)
                x = Y / t["depths"][:, None].astype(float)
                acc.append(x.var(axis=0, ddof=1))
        va = np.mean(va, 0)
        vl = np.mean(vl, 0)
        keep = vl > 1e-8
        ratio = va[keep] / vl[keep]
        rows.append(dict(part="A4", mechanism=mech,
                         var_ratio_mean=float(ratio.mean()),
                         var_ratio_med=float(np.median(ratio)),
                         var_ratio_q10=float(np.quantile(ratio, 0.1)),
                         var_ratio_q90=float(np.quantile(ratio, 0.9))))
    return rows


def part_b(log):
    rows = []
    base_prm = dict(effect_size=1.0, structural_zero_rate=0.1,
                    informative_zeros=False, dispersion=15.0, da_fraction=0.1,
                    depth_cv=0.3, effect_mode="absolute")
    # B1 全局 null R=20
    for mech in ["three_layer", "zinb"]:
        fdp_r, frac_r, fb = [], [], []
        for rep in range(20):
            Y, t = gen(mech, base_prm, 100, 20000, 3900 + rep)
            r = abs_nb_glm(Y, t["group"], N=t["depths"],
                           W=t["presence"].astype(float))
            fdp_r.append(1.0 if r["reject"].any() else 0.0)
            frac_r.append(r["reject"].mean())
            fb.append(r["n_fallback"])
        rows.append(dict(part="B1", mechanism=mech, R=20,
                         fdr_global_null=float(np.mean(fdp_r)),
                         mean_frac_rejected=float(np.mean(frac_r)),
                         mean_fallback=float(np.mean(fb))))
    # B2 absolute DA 格 oracle 掩码（bridge s8/s9 参数）
    for cid, mech, depth, sz, inf, eff, n, dv in [
            (1008, "beta_binomial", 20000, 0.0, False, 4.0, 50, 3.0),
            (1009, "zinb", 100000, 0.3, True, 4.0, 300, 3.0)]:
        pr = dict(effect_size=eff, structural_zero_rate=sz,
                  informative_zeros=inf, dispersion=dv, da_fraction=0.1,
                  depth_cv=0.3, effect_mode="absolute")
        fdp_r, tpr_r, nr = [], [], []
        for rep in range(20):
            Y, t = gen(mech, pr, n, depth, 4900 + rep)
            r = abs_nb_glm(Y, t["group"], N=t["depths"],
                           W=t["presence"].astype(float))
            da = t["abs_da_truth"]
            rej = r["reject"]
            fp = int((rej & ~da).sum())
            tp = int((rej & da).sum())
            fdp_r.append(fp / max(fp + tp, 1) if (fp + tp) else 0.0)
            tpr_r.append(tp / max(int(da.sum()), 1))
            nr.append(int(rej.sum()))
        rows.append(dict(part="B2", cell_id=cid, mechanism=mech, R=20,
                         fdr_oracle=float(np.mean(fdp_r)),
                         tpr=float(np.mean(tpr_r)),
                         mean_n_rej=float(np.mean(nr))))
    return rows


def part_c(df_cfg):
    rows = []
    for _, row in df_cfg.iterrows():
        prm = design.params_for_cell(row)
        seeds = np.random.SeedSequence(int(row["seed"])).spawn(3)
        for rep in range(3):
            rec = dict(cell_id=int(row["cell_id"]),
                       mechanism=row["mechanism"],
                       grid_group=row["grid_group"], rep=rep)
            for mode in ["absolute", "legacy"]:
                pr = dict(prm, effect_mode=mode)
                Y, t = gen(row["mechanism"], pr, int(row["n"]),
                           int(row["depth"]), seeds[rep])
                z = Y == 0
                g = t["group"]
                rec[f"{mode}_zero_rate"] = float(z.mean())
                rec[f"{mode}_struct_share"] = float(
                    t["structural_zeros"][z].mean()) if z.any() else np.nan
                rec[f"{mode}_uplift"] = float(
                    Y[g == 1].sum(1).mean() / Y[g == 0].sum(1).mean())
                rec[f"{mode}_n_da"] = int(t["da_taxa"].sum())
            # 每格 rep0 跑 abs_nb_glm（absolute 臂, oracle 掩码）
            if rep == 0:
                pr = dict(prm, effect_mode="absolute")
                Y, t = gen(row["mechanism"], pr, int(row["n"]),
                           int(row["depth"]), seeds[rep])
                t0 = time.time()
                r = abs_nb_glm(Y, t["group"], N=t["depths"],
                               W=t["presence"].astype(float))
                da = t["abs_da_truth"]
                fp = int((r["reject"] & ~da).sum())
                tp = int((r["reject"] & da).sum())
                rec.update(glm_fdp=(fp / max(fp + tp, 1)
                                    if (fp + tp) else 0.0),
                           glm_tpr=tp / max(int(da.sum()), 1),
                           glm_n_rej=int(r["reject"].sum()),
                           glm_fallback=int(r["n_fallback"]),
                           glm_t=time.time() - t0)
            rows.append(rec)
    return rows


def main():
    log = []
    t0 = time.time()
    ra = part_a(log)
    df_a = pd.DataFrame(ra)
    df_a.to_csv(os.path.join(OUT, "v323_acceptance_a.csv"), index=False)
    print(df_a.to_string(index=False))
    rb = part_b(log)
    df_b = pd.DataFrame(rb)
    df_b.to_csv(os.path.join(OUT, "v323_acceptance_b.csv"), index=False)
    print(df_b.to_string(index=False))
    df_cfg = design.supplementary_grid()
    rc = part_c(df_cfg)
    df_c = pd.DataFrame(rc)
    df_c.to_csv(os.path.join(OUT, "v323_smoke.csv"), index=False)
    print(df_c[["cell_id", "mechanism", "rep", "absolute_struct_share",
                "legacy_struct_share", "absolute_uplift", "glm_fdp",
                "glm_tpr"]].to_string(index=False))
    print("\n".join(log))
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
