#!/bin/sh
# 全网格分波运行（每波跑完再开下一波；单方法进程内逐重复落盘，断点续跑）。
# 波次顺序按任务要求：先 ANCOM-BC2/LinDA/corncob 三方法，DESeq2 后台分批殿后，
# 最后是全局零假设格 R=2000 确认档（design.py n_replicates）。
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=python3
$PY run_r_baseline_grid.py run --methods linda    --cells 0-47 --workers 2
$PY run_r_baseline_grid.py run --methods ancombc2 --cells 0-47 --workers 2
$PY run_r_baseline_grid.py run --methods corncob  --cells 0-47 --workers 2
$PY run_r_baseline_grid.py run --methods deseq2   --cells 0-47 --workers 2
$PY run_r_baseline_grid.py run --methods linda,ancombc2,corncob,deseq2 --cells null --workers 2
echo "ALL_WAVES_DONE"
