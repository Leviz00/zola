#!/bin/sh
# watchdog.sh —— ZOLA R 基线网格：容器周期性重启下的韧性守护。
# 每 60s 检查三件事并修复：① 本地 R 环境（缺则从 portal tarball 恢复）；
# ② ~/r_grid_exchange 符号链接；③ export / waves 进程存活（缺则重启）。
# 本脚本自身也会随容器重启被杀，由 agent 轮询负责重新拉起。
#
# 幂等：所有操作均可重复执行；waves/export 内部逐重复/逐格断点续跑。

PORTAL=/mnt/agents/output
ENV_TAR="$PORTAL/r_env/renv.tar.gz"
SIM="$PORTAL/code/simulation"
EX="$PORTAL/r_grid_exchange"
LOG="$SIM/logs/watchdog.log"
RSCRIPT="$HOME/micromamba/envs/renv/bin/Rscript"

log() { echo "$(date '+%F %T') [wd] $*" >> "$LOG"; }

log "watchdog 启动 (pid $$)"

while true; do
  # 1. R 环境：本地缺失且 portal 有 tarball → 恢复（先 staging 再原子替换）；
  #    tarball 尚无（环境未建成）→ 确保 build_env.sh 在跑（跨周期续下缓存）
  if [ ! -x "$RSCRIPT" ]; then
    if [ -f "$ENV_TAR" ]; then
      log "恢复 renv 环境 ..."
      rm -rf "$HOME/.renv_stage"
      mkdir -p "$HOME/.renv_stage"
      if tar -xzf "$ENV_TAR" -C "$HOME/.renv_stage" 2>>"$LOG"; then
        rm -rf "$HOME/micromamba"
        mv "$HOME/.renv_stage/micromamba" "$HOME/micromamba"
        log "renv 恢复完成"
      else
        log "!! renv 恢复失败（可能被杀），下轮重试"
      fi
      rm -rf "$HOME/.renv_stage"
    elif ! pgrep -f "install_from_local.sh" >/dev/null 2>&1; then
      log "启动 install_from_local.sh（本地 channel 离线安装）..."
      (cd "$SIM" && nohup sh install_from_local.sh \
         > "$HOME/install.log" 2>&1 &)
    fi
  fi

  # 2. exchange 符号链接
  if [ ! -e "$HOME/r_grid_exchange" ]; then
    ln -sfn "$EX" "$HOME/r_grid_exchange" && log "重建 exchange 符号链接"
  fi

  # 3. export / waves 进程
  if [ ! -f "$EX/cellnull.export_done" ]; then
    # 导出未完成：确保 export 在跑（纯 Python，不依赖 R 环境）
    if ! pgrep -f "run_r_baseline_grid.py export" >/dev/null 2>&1; then
      log "启动 export ..."
      (cd "$SIM" && nohup python3 run_r_baseline_grid.py export \
         >> logs/export.log 2>&1 &)
    fi
  elif [ -x "$RSCRIPT" ]; then
    # 导出完成 + R 环境就绪：确保 waves 在跑
    if ! pgrep -f "run_r_grid_waves.sh" >/dev/null 2>&1 \
       && ! pgrep -f "run_r_baseline_grid.py run" >/dev/null 2>&1; then
      log "启动 waves ..."
      (cd "$SIM" && nohup sh run_r_grid_waves.sh \
         >> logs/waves.log 2>&1 &)
    fi
  fi

  sleep 60
done
