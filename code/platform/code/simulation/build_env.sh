#!/bin/sh
# build_env.sh —— 容器 30-45min 周期重启下的 renv 环境韧性构建。
# 核心：micromamba 包缓存（下载的 tarball）每 45s 上行同步到 portal
# （portal FUSE 不支持符号链接，但 tarball 是纯文件），重启后下行恢复，
# 使跨周期下载进度单调累积。环境真正建成后由调用方打包到 portal。
#
# 幂等：反复执行直至打印 BUILD_OK。

PORTAL=/mnt/agents/output/r_env
PCACHE="$PORTAL/pkgs_cache"
MM="$HOME/bin/micromamba"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
SIM=/mnt/agents/output/code/simulation
LOG="$SIM/logs/build_env.log"

log() { echo "$(date '+%F %T') [build] $*" >> "$LOG"; }

mkdir -p "$PCACHE" "$HOME/bin" "$MAMBA_ROOT_PREFIX/pkgs/cache"

# micromamba 二进制：portal 副本恢复（portal 不可执行，拷到本地）
if [ ! -x "$MM" ]; then
  cp "$PORTAL/bin/micromamba" "$MM" && chmod +x "$MM"
  log "micromamba 二进制已从 portal 恢复"
fi

env_ok() {
  [ -x "$MAMBA_ROOT_PREFIX/envs/renv/bin/Rscript" ] || return 1
  "$MM" run -n renv Rscript -e \
    'library(ANCOMBC);library(corncob);library(MicrobiomeStat);library(DESeq2);library(phyloseq);library(microbiome)' \
    >/dev/null 2>&1
}

if env_ok; then log "环境已就绪，跳过"; echo BUILD_OK; exit 0; fi

attempt=0
while true; do
  attempt=$((attempt + 1))
  log "第 $attempt 轮 create（恢复缓存后开工）"
  rsync -a "$PCACHE/" "$MAMBA_ROOT_PREFIX/pkgs/cache/" >> "$LOG" 2>&1

  # 后台周期上行同步缓存（纯 tarball，无符号链接）
  (while true; do
     rsync -a "$MAMBA_ROOT_PREFIX/pkgs/cache/" "$PCACHE/" >/dev/null 2>&1
     sleep 45
   done) &
  SYNCPID=$!

  rm -rf "$MAMBA_ROOT_PREFIX/envs/renv"
  "$MM" create -y -n renv -c conda-forge -c bioconda \
    r-base=4.4.3 \
    bioconductor-ancombc bioconductor-deseq2 bioconductor-phyloseq \
    bioconductor-genomeinfodbdata \
    r-corncob r-microbiomestat >> "$LOG" 2>&1
  RC=$?

  kill $SYNCPID 2>/dev/null
  rsync -a "$MAMBA_ROOT_PREFIX/pkgs/cache/" "$PCACHE/" >/dev/null 2>&1

  if [ $RC -ne 0 ]; then
    log "create rc=$RC（被杀或失败），下轮续下"
    sleep 5
    continue
  fi

  log "create 成功，执行 rebuild_renv.sh 收尾（post-link/microbiome/版本核对）"
  sh "$SIM/rebuild_renv.sh" >> "$LOG" 2>&1 || true

  if env_ok; then
    log "环境校验通过"
    echo BUILD_OK
    exit 0
  fi
  log "env_ok 校验未过，重试"
done
