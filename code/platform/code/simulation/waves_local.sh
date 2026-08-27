#!/bin/sh
# waves_local.sh —— ZOLA R 基线全网格分波运行（raw 落本地 ext4，
# 由 local_guard.sh 定期同步回 portal；规避 FUSE 写死锁期间的数据丢失）。
# 幂等断点续跑：每格每方法按 (rep) 覆盖度判定，逐重复落盘。
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export R_RAW_DIR="/tmp/raw_local"
export R_LOG_DIR="/tmp/raw_local_logs"
export R_EXCHANGE_DIR="/tmp/exchange_local"
mkdir -p "$R_RAW_DIR" "$R_LOG_DIR"
PY=python3
# linda 已 48 格全量完成（portal 已有权威结果，种子到本地），从 ancombc2 续
$PY run_r_baseline_grid.py run --methods ancombc2 --cells 0-47 --workers 2
$PY run_r_baseline_grid.py run --methods corncob  --cells 0-47 --workers 2
$PY run_r_baseline_grid.py run --methods deseq2   --cells 0-47 --workers 2
# null 格 R=2000 前置：等待 null 导出完成（导出与网格波次并行）
while [ ! -f "$R_EXCHANGE_DIR/cellnull.export_done" ]; do sleep 30; done
$PY run_r_baseline_grid.py run --methods linda,ancombc2,corncob,deseq2 --cells null --workers 2
echo "ALL_WAVES_DONE"
