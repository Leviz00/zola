#!/bin/bash
# 断点续跑守护循环：容器周期性重启会杀掉后台进程；本循环在进程被杀后
# 自动重启 run_estimated_weighting.py（checkpoint 跳过已完成重复），
# 直到其正常退出（全网格完成 + 汇总落盘）。
# 停止方法：touch /tmp/estimated_weighting_STOP
cd /mnt/agents/output/code/simulation || exit 1
LOG=/mnt/agents/output/analysis/estimated_weighting/fullgrid_R20c.log
while true; do
  [ -f /tmp/estimated_weighting_STOP ] && { echo "$(date) STOP flag" >> "$LOG"; break; }
  echo "$(date) launch attempt" >> "$LOG"
  python3 run_estimated_weighting.py --R 20 --cores 2 >> "$LOG" 2>&1 && {
    echo "$(date) runner exited 0 (done)" >> "$LOG"; break; }
  echo "$(date) runner died (exit $?), restarting after 10s" >> "$LOG"
  sleep 10
done
