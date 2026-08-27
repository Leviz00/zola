"""R 侧四基线（ANCOM-BC2 / LinDA / DESeq2 / corncob）全网格运行驱动。

配套 ``r_baselines_full.R`` 使用，流程（baselines.md 交换/回收格式）：

1. ``export``：按 design.py 种子规则（格子种子 -> SeedSequence.spawn(R) 每重复
   一个子流，与 run_baseline_grid.py 完全一致）把 fractional 48 格 x
   ``n_replicates_screen``(=100) 的数据集写成 Python->R 交换 CSV
   （counts/meta/truth），写到本地盘 ``$HOME/r_grid_exchange``（/mnt 为网络盘，
   慢）；另导出全局零假设确认格 ``null``（three_layer, sz=0.3, 非信息性,
   da_fraction=0，种子与 run_null_confirmation.py 相同：base*10007+7000，
   R=2000 = design.py ``n_replicates`` 确认档）。
2. ``run``：以 (cell, method, rep 区间) 为作业、双 worker 并行调用 Rscript
   （每格每方法一个 R 进程，包加载摊销到 100 个重复；R 侧逐重复落盘，
   断点续跑）。null 格按 500 重复一批便于检查点。
3. ``score``：回收 discoveries，用 metrics.py 逐重复算 FDP/TPR（未检验 =
   未拒绝，缺失类群按 False 计并记录过滤数），汇总格级实证 FDR / 功效
   （FDP 均值 + MC SE），写 ``results/r_baselines_full_{replicates,grid}.csv``
   与 null 格确认结果 ``r_baselines_full_null_R2000.csv``。

用法（simulation/ 目录下）：
    python run_r_baseline_grid.py export [--null-only]
    python run_r_baseline_grid.py bench            # 每方法单重复耗时基准
    python run_r_baseline_grid.py run [--methods a,b,c,d] [--cells 0-47|null]
    python run_r_baseline_grid.py score
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

import design
import generators
import metrics

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
RAW_DIR = os.environ.get(
    "R_RAW_DIR", os.path.join(RESULTS_DIR, "r_baselines_full", "raw"))
LOG_DIR = os.environ.get(
    "R_LOG_DIR", os.path.join(RESULTS_DIR, "r_baselines_full", "logs"))
CONFIG_CSV = os.path.join(HERE, "configs", "config_fractional.csv")
R_SCRIPT = os.path.join(HERE, "r_baselines_full.R")
RSCRIPT = os.path.expanduser("~/micromamba/envs/renv/bin/Rscript")
EXCHANGE_DIR = os.environ.get(
    "R_EXCHANGE_DIR", os.path.expanduser("~/r_grid_exchange"))  # 本地盘优先

P_TAXA = 100  # 与 run_baseline_grid.py 一致（非设计因子）
METHODS = ["ancombc2", "linda", "deseq2", "corncob"]

# 全局零假设确认格（design.py：确认档 R = n_replicates = 2000；参数与
# run_null_confirmation.py 的 null_noninformative 完全对齐）
NULL_PARAMS = dict(effect_size=2.0, dispersion=100.0, base_prevalence=0.9,
                   structural_zero_rate=0.3, informative_zeros=False,
                   da_fraction=0.0)
NULL_SEED = 20260701 * 10007 + 7000
NULL_R = 2000
NULL_BATCH = 500  # 每批重复数（检查点粒度）
NULL_N, NULL_DEPTH = 100, 20000


# ---------------------------------------------------------------------------
# 导出
# ---------------------------------------------------------------------------

def _write_rep(exdir, cell_label, rep, Y, truth):
    """按 baselines.md 交换格式写 cell<l>_rep<r>_{counts,meta,truth}.csv。"""
    stem = os.path.join(exdir, f"cell{cell_label}_rep{rep}")
    sample_ids = [f"s{i:03d}" for i in range(Y.shape[0])]
    taxa = [f"taxon{j:03d}" for j in range(Y.shape[1])]
    pd.DataFrame(Y, index=sample_ids, columns=taxa).to_csv(
        f"{stem}_counts.csv")
    pd.DataFrame({"sample_id": sample_ids, "group": truth["group"]}).to_csv(
        f"{stem}_meta.csv", index=False)
    pd.DataFrame({"taxon": taxa,
                  "da": truth["da_taxa"].astype(int)}).to_csv(
        f"{stem}_truth.csv", index=False)


def export_grid(null_only=False):
    os.makedirs(EXCHANGE_DIR, exist_ok=True)
    if not null_only:
        cfg = pd.read_csv(CONFIG_CSV)
        R = int(cfg["n_replicates_screen"].iloc[0])
        assert (cfg["n_replicates_screen"] == R).all()
        t0 = time.time()
        n_written = 0
        for _, row in cfg.iterrows():
            cid = int(row["cell_id"])
            marker = os.path.join(EXCHANGE_DIR, f"cell{cid}.export_done")
            if os.path.exists(marker):  # 断点续跑：整格已导出则跳过
                continue
            seeds = np.random.SeedSequence(int(row["seed"])).spawn(R)
            params = design.params_for_cell(row)
            for r in range(R):
                Y, truth = generators.generate(
                    row["mechanism"], params, n=int(row["n"]), p=P_TAXA,
                    depths=int(row["depth"]), seed=seeds[r])
                _write_rep(EXCHANGE_DIR, cid, r, Y, truth)
            open(marker, "w").write("ok\n")
            n_written += 1
            print(f"[export] cell {cid} ({row['mechanism']}, n={row['n']}, "
                  f"depth={row['depth']}) x{R} 完成 "
                  f"({time.time() - t0:.0f}s)", flush=True)
        print(f"[export] 筛查网格导出完成：{n_written} 格新导出")
    # null 确认格（R=2000）
    marker = os.path.join(EXCHANGE_DIR, "cellnull.export_done")
    if not os.path.exists(marker):
        t0 = time.time()
        seeds = np.random.SeedSequence(NULL_SEED).spawn(NULL_R)
        for r in range(NULL_R):
            Y, truth = generators.generate(
                "three_layer", NULL_PARAMS, n=NULL_N, p=P_TAXA,
                depths=NULL_DEPTH, seed=seeds[r])
            _write_rep(EXCHANGE_DIR, "null", r, Y, truth)
            if (r + 1) % 200 == 0:
                print(f"[export] null {r + 1}/{NULL_R} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        open(marker, "w").write("ok\n")
        print(f"[export] null 确认格导出完成（R={NULL_R}）")


# ---------------------------------------------------------------------------
# 运行调度
# ---------------------------------------------------------------------------

def _raw_csv(cell_label, method):
    return os.path.join(RAW_DIR, f"cell{cell_label}_{method}.csv")


def _job_done(cell_label, method, rep_lo, rep_hi):
    """判据：raw CSV（成功）与 errors CSV（失败已记录）覆盖全部重复。

    失败的重复已按任务要求记录在 *.errors.csv（不静默跳过、不无限重试），
    汇总时该重复缺测、MEMO 报告各方法错误数。
    """
    have = set()
    for path in (_raw_csv(cell_label, method),
                 _raw_csv(cell_label, method) + ".errors.csv"):
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, usecols=["rep"])
                have |= set(df["rep"].unique())
            except pd.errors.EmptyDataError:
                pass
    return all(r in have for r in range(rep_lo, rep_hi + 1))


def build_jobs(cells, methods):
    """作业 = (cell_label, method, rep_lo, rep_hi)。"""
    jobs = []
    for c in cells:
        if c == "null":
            for m in methods:
                for lo in range(0, NULL_R, NULL_BATCH):
                    hi = min(lo + NULL_BATCH, NULL_R) - 1
                    jobs.append(("null", m, lo, hi))
        else:
            cfg = pd.read_csv(CONFIG_CSV)
            R = int(cfg["n_replicates_screen"].iloc[0])
            for m in methods:
                jobs.append((str(int(c)), m, 0, R - 1))
    return [j for j in jobs if not _job_done(*j)]


def run_jobs(cells, methods, workers=2):
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    jobs = build_jobs(cells, methods)
    print(f"[run] 待跑作业 {len(jobs)} 个（worker={workers}）", flush=True)
    if not jobs:
        return
    procs = {}  # pid -> (job, logfile handle, t0)
    idx = 0
    t_start = time.time()
    n_done = 0
    while idx < len(jobs) or procs:
        while idx < len(jobs) and len(procs) < workers:
            cell_label, m, lo, hi = jobs[idx]
            out_csv = _raw_csv(cell_label, m)
            log_path = os.path.join(
                LOG_DIR, f"cell{cell_label}_{m}_{lo}-{hi}.log")
            lf = open(log_path, "a")
            cmd = [RSCRIPT, R_SCRIPT, EXCHANGE_DIR, str(cell_label), m,
                   str(lo), str(hi), out_csv]
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
            procs[p.pid] = (jobs[idx], lf, time.time(), p)
            print(f"[run] 启动 cell{cell_label} {m} rep{lo}-{hi} (pid {p.pid})",
                  flush=True)
            idx += 1
        time.sleep(5)
        for pid in list(procs):
            job, lf, t0, p = procs[pid]
            rc = p.poll()
            if rc is None:
                continue
            lf.close()
            cell_label, m, lo, hi = job
            ok = _job_done(cell_label, m, lo, hi)
            n_done += 1
            el_total = time.time() - t_start
            eta = el_total / n_done * (len(jobs) - n_done) if n_done else 0
            print(f"[run] {'完成' if ok and rc == 0 else '异常'} "
                  f"cell{cell_label} {m} rep{lo}-{hi} rc={rc} "
                  f"{time.time() - t0:.0f}s | 进度 {n_done}/{len(jobs)} "
                  f"ETA {eta / 3600:.1f}h", flush=True)
            if not ok:  # 未齐：最多重排 2 次（断点续跑会跳过已完成重复）
                n_retry = sum(1 for j in jobs[:idx] if j == job)
                if n_retry <= 2:
                    jobs.append(job)
                    print(f"[run] 重新排队 cell{cell_label} {m} rep{lo}-{hi}",
                          flush=True)
                else:
                    print(f"[run] !! 放弃 cell{cell_label} {m} rep{lo}-{hi} "
                          f"（重排 2 次后仍未齐，详见日志与 errors.csv）",
                          flush=True)
            del procs[pid]


def bench(cells=("1", "2")):
    """每方法单重复耗时基准（默认 cell 1: n=100 与 cell 2: n=300）。"""
    for c in cells:
        for m in METHODS:
            out = os.path.join(RAW_DIR, f"bench_cell{c}_{m}.csv")
            os.makedirs(RAW_DIR, exist_ok=True)
            t0 = time.time()
            subprocess.run([RSCRIPT, R_SCRIPT, EXCHANGE_DIR, c, m, "0", "0",
                            out], check=False,
                           capture_output=True, text=True)
            print(f"[bench] cell{c} {m}: {time.time() - t0:.1f}s "
                  f"（含 R 启动与包加载）", flush=True)


# ---------------------------------------------------------------------------
# 计分
# ---------------------------------------------------------------------------

def _load_truth(cell_label, reps):
    """逐重复真值 {rep: (taxa list, da bool array)}。"""
    out = {}
    for r in reps:
        path = os.path.join(EXCHANGE_DIR,
                            f"cell{cell_label}_rep{r}_truth.csv")
        df = pd.read_csv(path)
        out[r] = (df["taxon"].tolist(), df["da"].to_numpy(dtype=bool))
    return out


def score():
    cfg = pd.read_csv(CONFIG_CSV)
    R_screen = int(cfg["n_replicates_screen"].iloc[0])
    cells = [str(int(c)) for c in cfg["cell_id"]] + ["null"]
    rep_rows = []
    for cell_label in cells:
        is_null = cell_label == "null"
        R = NULL_R if is_null else R_screen
        truth = None
        for m in METHODS:
            path = _raw_csv(cell_label, m)
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            if df.empty:
                continue
            # 清洗：并发写错位残留的坏行（rep 非数值或超出区间），剔除并计数
            rep_num = pd.to_numeric(df["rep"], errors="coerce")
            bad = rep_num.isna() | (rep_num < 0) | (rep_num >= R)
            if bad.any():
                print(f"[score] !! {path}: 剔除 {int(bad.sum())} 行坏行",
                      flush=True)
                df = df.loc[~bad].copy()
                df["rep"] = rep_num[~bad].astype(int)
            # 去重：portal FUSE 抖动期可能重复追加同一 (rep, taxon) 行，
            # 按 (cell, rep, method, taxon) 保留首行（幂等，无重复时无操作）
            n0 = len(df)
            df = df.drop_duplicates(
                subset=["cell", "rep", "method", "taxon"], keep="first")
            if len(df) < n0:
                print(f"[score] !! {path}: 去重 {n0 - len(df)} 行", flush=True)
            # 去重：并发续跑可能产生同 (rep,taxon) 重复行，保留最后写入
            dup = df.duplicated(subset=["rep", "taxon"], keep="last")
            if dup.any():
                print(f"[score] !! {path}: 剔除 {int(dup.sum())} 行重复 "
                      f"(rep,taxon)", flush=True)
                df = df.loc[~dup].copy()
            if truth is None:
                truth = _load_truth(cell_label, range(R))
            for rep, g in df.groupby("rep"):
                taxa, da = truth[int(rep)]
                # 未检验 = 未拒绝（baselines.md 执行注意 2）：方法未输出的
                # 类群按 rejected=False 计；过滤数记入脚注列
                rej_map = dict(zip(g["taxon"], g["rejected"].astype(bool)))
                rejected = np.array([rej_map.get(t, False) for t in taxa])
                fdp_r, n_rej = metrics.fdp(rejected, da)
                tpr_r, _ = metrics.tpr(rejected, da)
                n_filtered = int(len(taxa) - g["taxon"].nunique())
                rep_rows.append(dict(cell_id=cell_label, rep=int(rep),
                                     method=m, fdp=fdp_r, tpr=tpr_r,
                                     n_rej=n_rej, n_filtered=n_filtered))
        print(f"[score] cell {cell_label} 完成", flush=True)
    rep_df = pd.DataFrame(rep_rows)
    rep_path = os.path.join(RESULTS_DIR, "r_baselines_full_replicates.csv")
    rep_df.to_csv(rep_path, index=False)

    summ = []
    for (cid, m), g in rep_df.groupby(["cell_id", "method"]):
        fdr_hat, fdr_se, _ = metrics.empirical_rate(g["fdp"].to_numpy())
        tpr_hat, tpr_se, _ = metrics.empirical_rate(
            g["tpr"].dropna().to_numpy())
        summ.append(dict(cell_id=cid, method=m, R=len(g),
                         emp_fdr=fdr_hat, fdr_mc_se=fdr_se,
                         power=tpr_hat, power_mc_se=tpr_se,
                         mean_rejections=g["n_rej"].mean(),
                         mean_filtered=g["n_filtered"].mean()))
    summ = pd.DataFrame(summ)
    # 筛查格：合并 config 列；null 格单独出确认表
    grid = summ[summ["cell_id"] != "null"].copy()
    grid["cell_id"] = grid["cell_id"].astype(int)
    grid = cfg.merge(grid, on="cell_id")
    grid_path = os.path.join(RESULTS_DIR, "r_baselines_full_grid.csv")
    grid.to_csv(grid_path, index=False)
    null = summ[summ["cell_id"] == "null"].copy()
    null["gate_pass"] = (null["emp_fdr"] - 0.05).abs() <= 0.01 + 1e-12
    null_path = os.path.join(RESULTS_DIR, "r_baselines_full_null_R2000.csv")
    null.to_csv(null_path, index=False)
    print(f"[score] wrote {rep_path}\n[score] wrote {grid_path}\n"
          f"[score] wrote {null_path}")
    return rep_df, grid, null


def _parse_cells(spec):
    if spec == "null":
        return ["null"]
    if "-" in spec:
        a, b = spec.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["export", "run", "score", "bench"])
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--cells", default="0-47")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--null-only", action="store_true")
    args = ap.parse_args()
    methods = [m for m in args.methods.split(",") if m]
    if args.mode == "export":
        export_grid(null_only=args.null_only)
    elif args.mode == "run":
        run_jobs(_parse_cells(args.cells), methods, workers=args.workers)
    elif args.mode == "bench":
        bench()
    elif args.mode == "score":
        score()


if __name__ == "__main__":
    main()
