#!/bin/sh
# waves_local.sh —— 本地盘版 waves（免疫 portal FUSE 写堵）。
# 用环境变量把 RAW/LOG 指到本地盘；顺序与 run_r_grid_waves.sh 一致。
# 断点续跑；由 watchdog.sh 拉起，日志 ~/waves_local.log。
SIM=/mnt/agents/output/code/simulation
cd "$SIM"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export R_RAW_DIR="$HOME/r_raw_local"
export R_LOG_DIR="$HOME/r_raw_local_logs"
PY=python3
$PY run_r_baseline_grid.py run --methods linda    --cells 0-47 --workers 2
$PY run_r_baseline_grid.py run --methods ancombc2 --cells 0-47 --workers 2
$PY run_r_baseline_grid.py run --methods corncob  --cells 0-47 --workers 2
$PY run_r_baseline_grid.py run --methods deseq2   --cells 0-47 --workers 2
$PY run_r_baseline_grid.py run --methods linda,ancombc2,corncob,deseq2 --cells null --workers 2
echo "ALL_WAVES_DONE"
