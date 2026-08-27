#!/bin/sh
# rebuild_renv.sh —— ZOLA 项目 R 基线环境（renv）一键重建脚本。
#
# 背景（baselines.md 头部状态条）：容器 uid=999 无 root，apt 装 R 不可行；
# 改用 micromamba 独立二进制装到 $HOME（/mnt 是网络盘，慢，故不放 /mnt）。
# 已知坑（baselines.md）：① 并发 micromamba 安装会互相破坏包文件 —— 本脚本
# 全程单次串行安装；② 缺 bioconductor-genomeinfodbdata 需显式补装（已列入）。
#
# 用法：sh rebuild_renv.sh        # 幂等：已存在 renv 则跳过创建
# 目标版本（与 baselines.md 2026-07-30 状态条对齐）：
#   R 4.4.3 / ANCOMBC 2.8.0 / corncob 0.4.2 / MicrobiomeStat 1.4 /
#   DESeq2 1.46.0 / phyloseq 1.50.0

set -eu

export MAMBA_ROOT_PREFIX="$HOME/micromamba"
MM="$HOME/bin/micromamba"
ENV=renv

# 1. micromamba 独立二进制（已存在则跳过）
if [ ! -x "$MM" ]; then
  echo "[rebuild] 下载 micromamba 到 \$HOME/bin ..."
  mkdir -p "$HOME/bin"
  (cd "$HOME" && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
     | tar -xj bin/micromamba)
fi
"$MM" --version

# 2. 建环境（单次求解、串行安装；conda-forge + bioconda 二进制）
if [ -d "$MAMBA_ROOT_PREFIX/envs/$ENV" ]; then
  echo "[rebuild] 环境 $ENV 已存在，跳过创建"
else
  echo "[rebuild] 创建环境 $ENV（r-base=4.4.3 + 五个基线包）..."
  "$MM" create -y -n "$ENV" -c conda-forge -c bioconda \
    r-base=4.4.3 \
    bioconductor-ancombc bioconductor-deseq2 bioconductor-phyloseq \
    bioconductor-genomeinfodbdata \
    r-corncob r-microbiomestat
fi

# 3. 已知坑修复（baselines.md 头部状态条 + 2026-07-30 重建实录）
P="$MAMBA_ROOT_PREFIX/envs/$ENV"
# 3a. bioconductor-genomeinfodbdata 的 post-link（下载数据 tarball + R CMD
#     INSTALL）在批量安装中可能未跑成功（包记录存在但 R library 缺包），
#     缺则手动补跑：
if [ ! -d "$P/lib/R/library/GenomeInfoDbData" ]; then
  echo "[rebuild] 补跑 genomeinfodbdata post-link ..."
  (export PREFIX="$P" PATH="$P/bin:$PATH"
   bash "$P/bin/.bioconductor-genomeinfodbdata-post-link.sh")
fi
# 3b. ANCOMBC 2.8.0 的 data_sanity_check 需要 microbiome 包（conda 只有
#     r36/r40 老构建；CRAN 已存档）：编译型依赖走 conda 二进制（容器无
#     gfortran，源码编译 vegan/compositions 会失败），microbiome 本体从
#     Bioconductor 3.20 源码安装（纯 R）。
"$MM" install -y -n "$ENV" -c conda-forge -c bioconda \
  r-vegan r-rtsne r-reshape2 r-compositions r-purrr=1.2.2 r-tidyr=1.3.2
if ! "$MM" run -n "$ENV" Rscript -e 'library(microbiome)' >/dev/null 2>&1; then
  echo "[rebuild] 从 Bioconductor 3.20 源码安装 microbiome ..."
  "$MM" run -n "$ENV" Rscript -e '
repos <- c(BioC = "https://bioconductor.org/packages/3.20/bioc",
           CRAN = "https://cloud.r-project.org")
install.packages("microbiome", repos = repos)'
fi

# 4. 版本核对（对照 baselines.md 记录值）
echo "[rebuild] 版本核对："
"$MM" run -n "$ENV" Rscript -e '
cat("R:", R.version.string, "\n")
for (p in c("ANCOMBC", "corncob", "MicrobiomeStat", "DESeq2", "phyloseq",
            "microbiome", "GenomeInfoDbData"))
  cat(sprintf("%-15s %s\n", p, as.character(packageVersion(p))))
'

# 5. 锁定文件（可复现；microbiome 为源码安装，见 3b 注释）
"$MM" env export -n "$ENV" > "$(dirname "$0")/renv_lock.yml"
echo "[rebuild] 锁定文件已写入 $(dirname "$0")/renv_lock.yml"
