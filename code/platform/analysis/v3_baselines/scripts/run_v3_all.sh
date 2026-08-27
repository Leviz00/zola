#!/bin/sh
# run_v3_all.sh —— v3 基线一链到底：环境缺失则装（~80s）→ waves 直写
# portal（逐重复断点续跑）→ 未齐 320 主 CSV 则再来一轮。被杀即整体重跑。
SIM=/mnt/agents/output/code/simulation
V3=/mnt/agents/output/analysis/v3_baselines
export R_WORK_DIR=/tmp/rwork
mkdir -p /tmp/rwork

round=0
while true; do
  round=$((round + 1))
  echo "$(date '+%F %T') [v3all] 第 $round 轮"
  if [ ! -x /tmp/rwork/micromamba/envs/renv/bin/Rscript ]; then
    echo "$(date '+%F %T') [v3all] 装环境 ..."
    sh "$SIM/install_from_local.sh" >> /tmp/install_v3.log 2>&1 || continue
  fi
  ln -sfn /tmp/rwork/micromamba "$HOME/micromamba"
  sh "$V3/scripts/run_v3_waves.sh" >> /tmp/v3_waves.log 2>&1
  n=$(ls "$V3"/raw/linda/*.csv "$V3"/raw/ancombc2/*.csv 2>/dev/null | \
      grep -v -e '\.errors\.csv' -e '\.dropped\.csv' | wc -l)
  echo "$(date '+%F %T') [v3all] 主 CSV 计数 $n/320"
  [ "$n" -ge 320 ] && break
  sleep 5
done
echo "$(date '+%F %T') V3_ALL_DONE"
