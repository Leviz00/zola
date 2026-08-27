#!/bin/sh
# local_guard.sh —— ZOLA R 侧跨代自治守护（每 60s 一轮）：
#   ① 本地 renv 环境缺失 → 后台 install_from_local.sh（快速离线，
#      含 GIDD 桩包与 microbiome）；
#   ② ~/r_grid_exchange 符号链接（portal 副本仅供参考）；
#   ③ 后台本地导出 exchange（与 ① 并行，确定性种子）；
#   ④ 本地 raw 首次种子（portal → ~/raw_local，仅当本地为空）；
#   ⑤ 环境+grid 导出均就绪 → 确保 waves_local.sh 在跑；
#   ⑥ 每 2 轮（~2min）FUSE 写探测 + raw 回传 portal + status。
# 本进程随容器重启死亡，由 meta-watchdog（agent 轮询）重新拉起。

PORTAL=/mnt/agents/output
SIM="$PORTAL/code/simulation"
RAW_PORTAL="$SIM/results/r_baselines_full/raw"
RAW_LOCAL="/tmp/raw_local"
EX="$PORTAL/r_grid_exchange"
EXL="/tmp/exchange_local"
RSCRIPT="$HOME/micromamba/envs/renv/bin/Rscript"
STATUS="$PORTAL/analysis/r_baselines_full/GUARD_STATUS.txt"
LOG="/tmp/guard.log"

log() { echo "$(date '+%F %T') [guard] $*" >> "$LOG"; }
log "guard 启动 (pid $$)"

n=0
while true; do
  n=$((n + 1))

  # ① 环境（后台安装，不阻塞导出）
  if [ ! -x "$RSCRIPT" ] \
     && ! pgrep -f "install_from_local.s[h]" >/dev/null 2>&1; then
    log "安装 renv 环境 ..."
    (cd "$SIM" && R_WORK_DIR=/tmp/rwork nohup sh install_from_local.sh >> /tmp/install.log 2>&1 &)
  fi

  # ② 符号链接自愈：~/micromamba → /tmp/rwork/micromamba（RSCRIPT 硬编码
  #    路径经链接落到 /tmp 持久副本）；exchange 参考链接
  [ -e "$HOME/micromamba" ] || ln -sfn /tmp/rwork/micromamba "$HOME/micromamba"
  [ -e "$HOME/r_grid_exchange" ] || ln -sfn "$EX" "$HOME/r_grid_exchange"

  # ③ exchange 本地导出（FUSE 读抖动会假 ENOENT 杀死 R 作业；
  #    网格 48 格导完即放行 waves）。串行于 ① 之后：cgroup 内存上限
  #    3GB，install(~1.5GB) 与 export(~0.5GB) 并行叠加浏览器与其他
  #    租户有 OOM 重启之嫌，串行压低内存峰值。
  if [ -x "$RSCRIPT" ] && [ ! -f "$EXL/cell47.export_done" ] \
     && ! pgrep -f "run_r_baseline_grid.py expor[t]" >/dev/null 2>&1; then
    log "本地导出 exchange（grid ~3min）..."
    (cd "$SIM" && R_EXCHANGE_DIR="$EXL" \
       nohup python3 run_r_baseline_grid.py export \
       >> /tmp/export.log 2>&1 &)
  fi

  # ④ raw 种子（仅本地为空时，portal → local 一次性）
  if [ ! -f "$RAW_LOCAL/cell0_linda.csv" ] && [ -f "$RAW_PORTAL/cell0_linda.csv" ]; then
    log "raw 种子：portal → local ..."
    mkdir -p "$RAW_LOCAL"
    rsync -a "$RAW_PORTAL/" "$RAW_LOCAL/" >> "$LOG" 2>&1
    log "raw 种子完成（$(ls "$RAW_LOCAL" | wc -l) 文件）"
  fi

  # ⑤ waves：环境 + grid 导出双就绪
  if [ -x "$RSCRIPT" ] && [ -f "$EXL/cell47.export_done" ]; then
    if ! pgrep -f "waves_local.s[h]" >/dev/null 2>&1 \
       && ! pgrep -f "run_r_baseline_grid.py ru[n]" >/dev/null 2>&1; then
      log "拉起 waves_local.sh ..."
      (cd "$SIM" && nohup sh waves_local.sh >> /tmp/waves.log 2>&1 &)
    fi
  fi

  # ⑥ 每 2 轮（~2min）：FUSE 写探测 + raw 回传 portal + status
  if [ $((n % 2)) -eq 0 ]; then
    if echo "probe $(date +%s)" > "$PORTAL/r_env/.guard_probe" 2>/dev/null; then
      rsync -a "$RAW_LOCAL/" "$RAW_PORTAL/" >/dev/null 2>&1 \
        && log "raw 回传 portal 完成"
      {
        echo "# GUARD_STATUS $(date '+%F %T')"
        echo "raw_local files: $(ls "$RAW_LOCAL" 2>/dev/null | wc -l)"
        for m in ancombc2 corncob deseq2; do
          c=$(ls "$RAW_LOCAL"/cell*_"$m".csv 2>/dev/null | wc -l)
          echo "$m cells present: $c/48"
        done
        tail -3 /tmp/waves.log 2>/dev/null
      } > "$STATUS" 2>/dev/null || true
    else
      log "FUSE 写仍不可用，raw 暂存本地"
    fi
  fi

  sleep 60
done
