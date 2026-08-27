#!/bin/sh
# run_v3_waves.sh —— v3 考场外部基线驱动：8 格 × {linda, ancombc2} × rep0..19。
# 2 worker；R 进程逐重复落盘到 /tmp/v3_raw（重置安全），全部完成后经
# portal 写探测 rsync 回 /mnt/agents/output/analysis/v3_baselines/raw/。
# 幂等断点续跑：主 CSV / errors.csv 已存在的重复自动跳过。
#
# 用法：sh run_v3_waves.sh [sync_only]

RSCRIPT="$HOME/micromamba/envs/renv/bin/Rscript"   # 符号链接 → /tmp/rwork
EX=/mnt/agents/output/analysis/v3_baselines/exchange
# 直写 portal：/tmp 会被整体清掉；write.csv 覆盖写 + 断点查存在性，
# 重做同重复只会重写同名文件，不产生重复行。
RAW_LOCAL=/mnt/agents/output/analysis/v3_baselines/raw
RAW_PORTAL=/mnt/agents/output/analysis/v3_baselines/raw
R_SCRIPT=/mnt/agents/output/analysis/v3_baselines/scripts/run_v3_baselines.R
CELLS="1000 1002 1004 1005 1006 1007 1008 1009"
METHODS="linda ancombc2"
LOG=/tmp/v3_waves.log

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# portal 写探测（HANDOFF_R_GUARD：大写会假成功，先探后传）
portal_probe() {
  probe="$RAW_PORTAL/.probe_$$"
  mkdir -p "$RAW_PORTAL" 2>/dev/null
  echo ok > "$probe" 2>/dev/null && [ "$(cat "$probe" 2>/dev/null)" = "ok" ]
  rc=$?
  rm -f "$probe" 2>/dev/null
  return $rc
}

sync_back() {
  for i in 1 2 3 4 5; do
    if portal_probe; then
      rsync -a "$RAW_LOCAL/" "$RAW_PORTAL/"
      # 回读校验：文件数与文件总字节（不含目录项）
      a=$(find "$RAW_LOCAL" -type f | wc -l)
      b=$(find "$RAW_PORTAL" -type f ! -name '.probe_*' | wc -l)
      sa=$(find "$RAW_LOCAL" -type f -printf '%s\n' | awk '{s+=$1} END{print s+0}')
      sb=$(find "$RAW_PORTAL" -type f ! -name '.probe_*' -printf '%s\n' | awk '{s+=$1} END{print s+0}')
      if [ "$a" = "$b" ] && [ "$sa" = "$sb" ]; then
        echo "$(date '+%F %T') [sync] 回传完成 $a 文件 $sa 字节" | tee -a "$LOG"
        return 0
      fi
      echo "$(date '+%F %T') [sync] 校验不齐 local=$a/$sa portal=$b/$sb，重试" >> "$LOG"
    else
      echo "$(date '+%F %T') [sync] portal 写探测失败，重试" >> "$LOG"
    fi
    sleep 20
  done
  echo "$(date '+%F %T') [sync] !! 回传未成功（下次调用 sync_only 续）" >> "$LOG"
  return 1
}

if [ "${1:-}" = "sync_only" ]; then
  sync_back
  exit $?
fi

if [ ! -x "$RSCRIPT" ]; then
  echo "$(date '+%F %T') [waves] !! Rscript 缺失：$RSCRIPT（先建环境/符号链接）" | tee -a "$LOG"
  exit 1
fi

mkdir -p "$RAW_LOCAL"
echo "$(date '+%F %T') [waves] 启动" >> "$LOG"

export V3_RSCRIPT="$RSCRIPT" V3_R_SCRIPT="$R_SCRIPT" V3_EX="$EX" \
       V3_RAW_LOCAL="$RAW_LOCAL" V3_LOG="$LOG"
JOB=/mnt/agents/output/analysis/v3_baselines/scripts/_v3_job.sh

for c in $CELLS; do
  for m in $METHODS; do
    printf '%s %s\n' "$c" "$m"
  done
done | xargs -n 2 -P 2 sh "$JOB"

echo "$(date '+%F %T') [waves] 全部作业结束，回传 portal ..." >> "$LOG"
sync_back
echo "$(date '+%F %T') [waves] V3_WAVES_DONE rc=$?" | tee -a "$LOG"
