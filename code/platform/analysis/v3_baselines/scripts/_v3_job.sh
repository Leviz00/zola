#!/bin/sh
# _v3_job.sh —— 单个 (cell, method) 作业（由 run_v3_waves.sh 经 xargs 调用）
# $1=cell_id $2=method；环境变量由父脚本导出
c="$1"
m="$2"
echo "$(date '+%F %T') [job] cell$c $m 开始" >> "$V3_LOG"
"$V3_RSCRIPT" "$V3_R_SCRIPT" "$V3_EX" "$V3_RAW_LOCAL" "$c" "$m" 0 19 \
  >> "$V3_LOG" 2>&1
rc=$?
echo "$(date '+%F %T') [job] cell$c $m 结束 rc=$rc" >> "$V3_LOG"
exit $rc
