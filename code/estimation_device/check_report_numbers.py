"""check_report_numbers.py — REPORT.md 头条数字 ↔ CSV 逐行复算对账。

评审 M1/m5 自查机制：REPORT.md 中每个头条数字必须能由 results/*.csv
（或 data/*.npz + 原始 metadata）单行复算。本脚本：
  1) 从 CSV/npz 重算各数字（不读 REPORT.md 的值）；
  2) 与报告声明的期望值比对（容差按报告舍入精度）；
  3) 按报告显示格式断言该字符串确实出现在 REPORT.md 中（防残留旧值）。

用法：python3 check_report_numbers.py
输出：逐项 PASS/FAIL 表；任一 FAIL 时退出码为 1。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/mnt/agents/output/realdata")
RES = ROOT / "results"
DATA = ROOT / "data"
META = Path("/mnt/agents/output/datasets/mbqc/mbqc_sample_metadata.csv")
SE_CUT = 1.0
DATASETS = ["ibdmdb", "mbqc", "agp"]

REPORT = (ROOT / "REPORT.md").read_text(encoding="utf-8")

results = []  # (item, expected, recomputed, ok, detail)


def check(item, expected, recomputed, ok, detail=""):
    results.append((item, expected, recomputed, bool(ok), detail))


def in_report(item, s):
    ok = s in REPORT
    results.append((f"REPORT 含 '{s}'", "present", "present" if ok else "MISSING",
                    ok, item))


def pertaxon_zone(df):
    on_b = df["on_boundary_pi"] | df["on_boundary_theta"]
    return np.where((df["se_logit_pi"] < SE_CUT) & ~on_b, "identifiable", "ridge")


# ---------------- 1. 可识别性（§4 表）：比例、ridge corr、区域计数 ---------------
EXP_IDENT = {"ibdmdb": (74.5, 210, 72, -0.928),
             "mbqc":   (46.6, 104, 119, -0.441),
             "agp":    (83.5, 167, 33, -0.188)}
for name in DATASETS:
    exp_pct, exp_id, exp_rg, exp_corr = EXP_IDENT[name]
    summ = pd.read_csv(RES / f"identifiability_{name}_summary.csv").iloc[0]
    per = pd.read_csv(RES / f"identifiability_{name}.csv")
    # (a) summary CSV 自洽：summary 必须等于逐类群文件的重算
    zone = pertaxon_zone(per)
    n_id, n_rg = int((zone == "identifiable").sum()), int((zone == "ridge").sum())
    corr_med = float(per.loc[zone == "ridge", "godambe_corr_alpha_beta"].median())
    check(f"{name} summary==逐类群重算 (n_ident/n_ridge/corr)",
          f"{int(summ['n_identifiable'])}/{int(summ['n_ridge'])}/{summ['ridge_corr_median']:.4f}",
          f"{n_id}/{n_rg}/{corr_med:.4f}",
          n_id == int(summ["n_identifiable"]) and n_rg == int(summ["n_ridge"])
          and abs(corr_med - summ["ridge_corr_median"]) < 1e-9)
    # (b) 与 REPORT §4 表对账
    pct = 100.0 * n_id / len(per)
    check(f"{name} 可识别比例 = {exp_pct}%", exp_pct, round(pct, 1),
          abs(pct - exp_pct) < 0.051, f"{n_id}/{len(per)}")
    check(f"{name} 区域计数 = {exp_id}/{exp_rg}", f"{exp_id}/{exp_rg}",
          f"{n_id}/{n_rg}", n_id == exp_id and n_rg == exp_rg)
    check(f"{name} 脊区 corr 中位 = {exp_corr}", exp_corr, round(corr_med, 3),
          abs(corr_med - exp_corr) < 5e-4)
    in_report(f"{name} §4 表", f"−{abs(exp_corr)}")  # REPORT 用 Unicode 负号
    in_report(f"{name} §4 表", f"**{exp_pct}%**")
# 旧错误值不得残留
check("REPORT 不含旧值 −0.729", "absent",
      "absent" if "-0.729" not in REPORT and "−0.729" not in REPORT else "PRESENT",
      "-0.729" not in REPORT and "−0.729" not in REPORT, "M1(a)")

# ---------------- 2. n_boundary_pi 阶段语义（M1(b)）-----------------------------
EXP_BND = {"ibdmdb": (45, 1, 12, 1), "mbqc": (109, 0, 3, 0), "agp": (24, 9, 1, 0)}
for name in DATASETS:
    exp_pi, exp_th, exp_jpi, exp_jth = EXP_BND[name]
    summ = pd.read_csv(RES / f"fit_{name}_summary.csv").iloc[0]
    pt = pd.read_csv(RES / f"fit_{name}_pertaxon.csv")
    pj = pd.read_csv(RES / f"fit_{name}_pertaxon_joint.csv")
    fin_pi, fin_th = int(pt["on_boundary_pi"].sum()), int(pt["on_boundary_theta"].sum())
    j_pi, j_th = int(pj["on_boundary_pi"].sum()), int(pj["on_boundary_theta"].sum())
    check(f"{name} n_boundary_pi/theta 为最终管线状态",
          f"{exp_pi}/{exp_th}", f"{int(summ['n_boundary_pi'])}/{int(summ['n_boundary_theta'])}",
          int(summ["n_boundary_pi"]) == exp_pi == fin_pi
          and int(summ["n_boundary_theta"]) == exp_th == fin_th,
          "summary==pertaxon 最终值==报告值")
    check(f"{name} n_boundary_*_joint 留底正确",
          f"{exp_jpi}/{exp_jth}",
          f"{int(summ['n_boundary_pi_joint'])}/{int(summ['n_boundary_theta_joint'])}",
          int(summ["n_boundary_pi_joint"]) == exp_jpi == j_pi
          and int(summ["n_boundary_theta_joint"]) == exp_jth == j_th,
          "joint 列==pertaxon_joint 重算")
    check(f"{name} boundary_counts_stage 已标注", "joint|polish|refine",
          str(summ.get("boundary_counts_stage", "MISSING")),
          str(summ.get("boundary_counts_stage", "")) in {"joint", "polish", "refine"})
in_report("§3 表 ibdmdb π 撞边界", "| 45 | 1 |")
in_report("§3 表 mbqc π 撞边界", "| 109 | 0 |")
in_report("§3 表 agp π 撞边界", "| 24 | 9 |")

# ---------------- 3. φ 估计（§3 表 / §6）-----------------------------------------
smb = pd.read_csv(RES / "fit_mbqc_summary.csv").iloc[0]
check("mbqc φ̂ = 1454（内点）", 1454, round(float(smb["phi_hat"])),
      abs(float(smb["phi_hat"]) - 1453.585) < 0.01
      and not bool(smb["phi_on_boundary"]))
check("mbqc SE(log φ̂) = 0.058", 0.058, round(float(smb["se_gamma_logphi"]), 3),
      abs(float(smb["se_gamma_logphi"]) - 0.058) < 5e-4)
in_report("§3 表 mbqc φ̂", "**1,454**")
in_report("§3 表 mbqc SE", "**0.058**")
for name in ["ibdmdb", "agp"]:
    s = pd.read_csv(RES / f"fit_{name}_summary.csv").iloc[0]
    check(f"{name} φ̂ 撞上界 1e5", "1e5/True",
          f"{float(s['phi_hat']):g}/{bool(s['phi_on_boundary'])}",
          float(s["phi_hat"]) == 1e5 and bool(s["phi_on_boundary"]))

# ---------------- 4. 深度平衡 KW（§8.1）------------------------------------------
bt = pd.read_csv(RES / "ibdmdb_depth_balance_tests.csv")
kw = bt[bt["test"] == "Kruskal-Wallis"].iloc[0]
check("KW H = 2.89", 2.89, round(float(kw["statistic"]), 2),
      abs(float(kw["statistic"]) - 2.89) < 5e-3)
check("KW p = 0.236", 0.236, round(float(kw["p_value"]), 3),
      abs(float(kw["p_value"]) - 0.236) < 5e-4)
in_report("§8.1", "H=2.89，p=0.236")

# ---------------- 5. mock 真值验证（§8.2）----------------------------------------
auc = pd.read_csv(RES / "mbqc_mock_auc.csv").set_index("mock")
EXP_MOCK = {"fecal": (0.945, 0.948, 16, 16, 1158, 7),
            "oral":  (0.922, None,  22, 21, 1146, 4)}
for mock, (e_auc, e_sens, e_lab, e_det, e_n, e_fail) in EXP_MOCK.items():
    r = auc.loc[mock]
    check(f"mock {mock} AUC(π̂) = {e_auc}", e_auc, round(float(r["auc_pi"]), 3),
          abs(float(r["auc_pi"]) - e_auc) < 5e-4)
    check(f"mock {mock} n/标注/检出/失败 = {e_n}/{e_lab}/{e_det}/{e_fail}",
          f"{e_n}/{e_lab}/{e_det}/{e_fail}",
          f"{int(r['n_samples'])}/{int(r['n_present_labeled'])}/{int(r['n_present_detected'])}/{int(r['n_taxa_fit_failed'])}",
          int(r["n_samples"]) == e_n and int(r["n_present_labeled"]) == e_lab
          and int(r["n_present_detected"]) == e_det
          and int(r["n_taxa_fit_failed"]) == e_fail)
    check(f"mock {mock} φ 撞 1e6 上界", "1e6/True",
          f"{float(r['phi_hat']):g}/{bool(r['phi_on_grid_top'])}",
          float(r["phi_hat"]) == 1e6 and bool(r["phi_on_grid_top"]))
    if e_sens is not None:
        check(f"mock {mock} AUC 敏感性 = {e_sens}", e_sens,
              round(float(r["auc_pi_sensitivity_excl_ambiguous"]), 3),
              abs(float(r["auc_pi_sensitivity_excl_ambiguous"]) - e_sens) < 5e-4)
    in_report(f"§8.2 {mock} AUC", f"**{e_auc:.3f}**")
# mock 可识别区比例（§8.2 判据：SE(logit π)<1 且未撞 π 边界，检出类群中）
EXP_MOCK_ID = {"fecal": (265, 336, 78.9), "oral": (252, 348, 72.4)}
for mock, (e_id, e_tot, e_pct) in EXP_MOCK_ID.items():
    df = pd.read_csv(RES / f"mbqc_mock_fit_{mock}.csv")
    det = df[df["ever_detected"] == True]
    ok_zone = (det["se_logit_pi"] < SE_CUT) & (~det["on_boundary_pi"])
    n_id, n_tot = int(ok_zone.sum()), len(det)
    check(f"mock {mock} 可识别 {e_id}/{e_tot} = {e_pct}%",
          f"{e_id}/{e_tot}/{e_pct}", f"{n_id}/{n_tot}/{100*n_id/n_tot:.1f}",
          n_id == e_id and n_tot == e_tot and abs(100*n_id/n_tot - e_pct) < 0.051)
    in_report(f"§8.2 {mock} 可识别", f"{e_pct}%（{e_id}/{e_tot}）")

# ---------------- 6. blank 污染谱（§8.3）-----------------------------------------
bc = pd.read_csv(RES / "mbqc_blank_contamination.csv")
rho_f, pv_f = spearmanr(bc["blank_prevalence"], bc["fecal_mock_prevalence"])
rho_o, pv_o = spearmanr(bc["blank_prevalence"], bc["oral_mock_prevalence"])
check("blank×fecal Spearman ρ = 0.782", "0.782/8.8e-149",
      f"{rho_f:.3f}/{pv_f:.2g}",
      abs(rho_f - 0.782) < 5e-4 and abs(pv_f - 8.8e-149) / 8.8e-149 < 0.05)
check("blank×oral Spearman ρ = 0.776", "0.776/1.7e-145",
      f"{rho_o:.3f}/{pv_o:.2g}",
      abs(rho_o - 0.776) < 5e-4 and abs(pv_o - 1.7e-145) / 1.7e-145 < 0.05)
n_det = int((bc["blank_prevalence"] > 0).sum())
n_hi = int((bc["blank_prevalence"] >= 0.10).sum())
n_hi_t1 = int(((bc["blank_prevalence"] >= 0.10) & bc["is_table1_member"]).sum())
check("blank 检出属/≥10%/Table1 = 376/97/27", "376/97/27",
      f"{n_det}/{n_hi}/{n_hi_t1}",
      n_det == 376 and n_hi == 97 and n_hi_t1 == 27)
# blank 中位深度 12,402：从 npz + 原始 metadata 复算
z = np.load(DATA / "mbqc_mockblank_genus.npz", allow_pickle=True)
Yb_all, samples = z["Y"], z["samples"].astype(str)
meta = pd.read_csv(META, usecols=["Unnamed: 0", "specimen_type"])
stype = meta.set_index("Unnamed: 0")["specimen_type"].loc[samples].to_numpy()
mb = stype == "blank"
depths = Yb_all.sum(axis=1)
mb_pos = mb & (depths > 0)
med_blank = int(np.median(depths[mb_pos]))
check("blank 401 个、396 深度>0、中位深度 12,402", "401/396/12402",
      f"{int(mb.sum())}/{int(mb_pos.sum())}/{med_blank}",
      int(mb.sum()) == 401 and int(mb_pos.sum()) == 396 and med_blank == 12402)
in_report("§8.3", "中位深度 12,402")
in_report("§8.3", "ρ=0.782")

# ---------------- 7. §2/§3 其他表内数字 ------------------------------------------
EXP_SUM = {"ibdmdb": (178, 282, 34, 16627.5, 31781, 0.797),
           "mbqc":   (13562, 223, 1, 19672.0, 1121020, 0.784),
           "agp":    (9511, 200, 1250, 10737.0, 300383, 0.774)}
for name, (e_n, e_p, e_min, e_med, e_max, e_zf) in EXP_SUM.items():
    s = pd.read_csv(RES / f"fit_{name}_summary.csv").iloc[0]
    check(f"{name} n/p/深度/零比例",
          f"{e_n}/{e_p}/{e_min}/{e_med}/{e_max}/{e_zf}",
          f"{int(s['n'])}/{int(s['p'])}/{s['N_min']:g}/{s['N_median']:g}/{s['N_max']:g}/{s['zero_fraction']:.3f}",
          int(s["n"]) == e_n and int(s["p"]) == e_p
          and float(s["N_min"]) == e_min and float(s["N_median"]) == e_med
          and float(s["N_max"]) == e_max
          and abs(float(s["zero_fraction"]) - e_zf) < 5e-4)
# AGP φ 边界论证扫描（§3 末段）
ps = pd.read_csv(RES / "phi_scan_agp.csv")
lls = ps.sort_values("phi")["profile_loglik"].to_numpy()
check("agp φ 扫描单调递增至上界", "monotone→-510209.31",
      f"{'monotone' if np.all(np.diff(lls) > 0) else 'NOT'}/{lls[-1]:.2f}",
      np.all(np.diff(lls) > 0) and abs(lls[-1] - (-510209.306)) < 0.01)

# ---------------- 输出 ------------------------------------------------------------
w = max(len(r[0]) for r in results)
n_fail = 0
print(f"{'检查项'.ljust(w)}  期望            重算            结果")
print("-" * (w + 46))
for item, exp, got, ok, detail in results:
    tag = "PASS" if ok else "FAIL"
    n_fail += 0 if ok else 1
    line = f"{item.ljust(w)}  {str(exp).ljust(14)}  {str(got).ljust(14)}  {tag}"
    if detail:
        line += f"  [{detail}]"
    print(line)
print("-" * (w + 46))
print(f"共 {len(results)} 项：{len(results) - n_fail} PASS, {n_fail} FAIL")
sys.exit(1 if n_fail else 0)
