"""四档对比表生成：从 estimated_weighting_*.csv 复算 MEMO 用的全部数字。

产出（/mnt/agents/output/analysis/estimated_weighting/tables/）：
  table_by_cell.csv        每格子×方法 FDR/功效（四档全）
  table_by_informative.csv 按信息性 on/off × 方法（格级均值 ± 格间 SE）
  table_tier_contrast.csv  estimated 对 oracle/placeholder 的差距（配对格级差）
  table_diag_by_cell.csv   每格估计诊断（AUC、post_mean、struct_frac、耗时）
终端打印 MEMO 引用的关键数字。所有数字可从 simulation/results/
estimated_weighting_replicates.csv 复算（本脚本即复算路径）。
"""
import os
import numpy as np
import pandas as pd

SIMRES = "/mnt/agents/output/code/simulation/results"
OUT = "/mnt/agents/output/analysis/estimated_weighting/tables"
os.makedirs(OUT, exist_ok=True)

rep = pd.read_csv(os.path.join(SIMRES, "estimated_weighting_replicates.csv"))
cfg = pd.read_csv("/mnt/agents/output/code/simulation/configs/config_fractional.csv")
diag = pd.read_csv(os.path.join(SIMRES, "estimated_weighting_diagnostics.csv"))

def emp(g, col):
    v = g[col].to_numpy(dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.nan, np.nan, 0
    return v.mean(), (v.std(ddof=1) / np.sqrt(v.size) if v.size > 1 else np.nan), v.size

# ---- 每格×方法 ----
rows = []
for (cid, method), g in rep.groupby(["cell_id", "method"]):
    fdr, fdr_se, n = emp(g, "fdp")
    tpr, tpr_se, _ = emp(g, "tpr")
    rows.append(dict(cell_id=cid, method=method, R_valid=n,
                     emp_fdr=fdr, fdr_mc_se=fdr_se,
                     power=tpr, power_mc_se=tpr_se,
                     mean_rejections=g.loc[g.n_rej >= 0, "n_rej"].mean()))
cell_tbl = cfg.merge(pd.DataFrame(rows), on="cell_id")
cell_tbl.to_csv(os.path.join(OUT, "table_by_cell.csv"), index=False)

# ---- 按信息性 on/off（格级均值的均值；SE = 格间 sd/√格数）----
def grp_summary(df, keys):
    out = []
    for k, g in df.groupby(keys):
        for stat, col in [("emp_fdr", "emp_fdr"), ("power", "power")]:
            pass
        out.append(dict(**dict(zip(keys, k)) if isinstance(keys, list) else {keys: k},
                        n_cells=g["cell_id"].nunique(),
                        emp_fdr=g["emp_fdr"].mean(),
                        fdr_cell_se=g["emp_fdr"].std(ddof=1) / np.sqrt(g["cell_id"].nunique()),
                        power=g["power"].mean(),
                        power_cell_se=g["power"].std(ddof=1) / np.sqrt(g["cell_id"].nunique()),
                        mean_rejections=g["mean_rejections"].mean()))
    return pd.DataFrame(out)

by_inf = grp_summary(cell_tbl, ["informative_zeros", "method"])
by_inf.to_csv(os.path.join(OUT, "table_by_informative.csv"), index=False)

# ---- estimated 对 oracle/placeholder 的格级差距 ----
piv_f = cell_tbl.pivot_table(index="cell_id", columns="method", values="emp_fdr")
piv_p = cell_tbl.pivot_table(index="cell_id", columns="method", values="power")
meta = cfg.set_index("cell_id")[["mechanism", "informative_zeros",
                                 "structural_zero_rate", "dispersion", "n", "depth"]]
contrast = meta.copy()
for wm in ["weighted_welch_t", "exclusion_wilcoxon"]:
    contrast[f"{wm}_fdr_est_minus_oracle"] = piv_f[f"{wm}_estimated"] - piv_f[f"{wm}_oracle"]
    contrast[f"{wm}_fdr_est_minus_placeholder"] = piv_f[f"{wm}_estimated"] - piv_f[f"{wm}_placeholder"]
    contrast[f"{wm}_power_est_minus_oracle"] = piv_p[f"{wm}_estimated"] - piv_p[f"{wm}_oracle"]
    contrast[f"{wm}_power_est_minus_placeholder"] = piv_p[f"{wm}_estimated"] - piv_p[f"{wm}_placeholder"]
contrast.to_csv(os.path.join(OUT, "table_tier_contrast.csv"))

# ---- 诊断 ----
dtbl = diag.groupby("cell_id").agg(
    t_est_mean=("t_est_total", "mean"),
    post_auc=("post_auc", "mean"),
    post_mean=("post_mean", "mean"),
    struct_frac=("struct_frac", "mean"),
    cov_fail_rate=("cov_success", lambda v: 1 - v.mean()),
    cnt_fail_rate=("cnt_success", lambda v: 1 - v.mean()),
    cnt_phi_med=("cnt_phi", "median"),
).reset_index().merge(cfg[["cell_id", "mechanism", "informative_zeros",
                           "structural_zero_rate", "dispersion", "n", "depth"]],
                      on="cell_id")
dtbl.to_csv(os.path.join(OUT, "table_diag_by_cell.csv"), index=False)

# ---- 终端关键数字 ----
pd.set_option("display.width", 260)
pd.set_option("display.float_format", lambda v: f"{v:.4f}")
order = ["naive_welch_t", "tss_wilcoxon",
         "weighted_welch_t_placeholder", "weighted_welch_t_estimated",
         "weighted_welch_t_oracle",
         "exclusion_wilcoxon_placeholder", "exclusion_wilcoxon_estimated",
         "exclusion_wilcoxon_oracle"]
bi = by_inf.set_index("method").loc[[m for m in order if m in by_inf.method.values]].reset_index()
print("=== 按信息性 on/off 汇总（格级均值）===")
print(bi.sort_values(["informative_zeros", "method"],
                     key=lambda s: s.map({m: i for i, m in enumerate(order)})
                     if s.name == "method" else s).to_string(index=False))
print("\n=== 信息性 on 格内 estimated vs oracle/placeholder（格级差均值）===")
on = contrast[contrast.informative_zeros == True]
print(on[[c for c in contrast.columns if c.startswith(("weighted", "exclusion"))]].mean().to_string())
print("\nwrote tables to", OUT)
