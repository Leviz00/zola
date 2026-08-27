#!/bin/sh
# run_v3_deseq2.sh —— v3 DESeq2 一链到底：环境缺失则装（~80s）→ 8 格 ×
# rep0..19 直写 portal raw/deseq2（逐重复断点续跑，覆盖写无重复行）→
# 未齐 160 主 CSV 则再来一轮。被杀即整体重跑。失败重复落 .errors.csv。
SIM=/mnt/agents/output/code/simulation
V3=/mnt/agents/output/analysis/v3_baselines
export R_WORK_DIR=/tmp/rwork
mkdir -p /tmp/rwork

RSCRIPT="$HOME/micromamba/envs/renv/bin/Rscript"
export V3_RSCRIPT="$RSCRIPT" \
       V3_R_SCRIPT="$V3/scripts/run_v3_baselines.R" \
       V3_EX="$V3/exchange" \
       V3_RAW_LOCAL="$V3/raw" \
       V3_LOG=/tmp/v3_deseq2_waves.log
JOB="$V3/scripts/_v3_job.sh"

round=0
while true; do
  round=$((round + 1))
  echo "$(date '+%F %T') [deseq2] 第 $round 轮"
  if [ ! -x /tmp/rwork/micromamba/envs/renv/bin/Rscript ]; then
    echo "$(date '+%F %T') [deseq2] 装环境 ..."
    sh "$SIM/install_from_local.sh" >> /tmp/install_v3.log 2>&1 || continue
  fi
  ln -sfn /tmp/rwork/micromamba "$HOME/micromamba"

  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  for c in 1000 1002 1004 1005 1006 1007 1008 1009; do
    printf '%s deseq2\n' "$c"
  done | xargs -n 2 -P 2 sh "$JOB"

  n=$(ls "$V3"/raw/deseq2/*.csv 2>/dev/null | grep -vc -e '\.errors\.csv')
  echo "$(date '+%F %T') [deseq2] 主 CSV 计数 $n/160"
  [ "$n" -ge 160 ] && break
  sleep 5
done
echo "$(date '+%F %T') V3_DESEQ2_DONE"
