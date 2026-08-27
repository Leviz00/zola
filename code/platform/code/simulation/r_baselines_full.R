# R 侧四基线全网格运行器（ZOLA 项目，baselines.md 的落地实现）。
#
# 功能：对一个格子的若干重复逐个跑 ANCOM-BC2 / LinDA / DESeq2 / corncob
# （全部官方默认参数，名义 FDR 统一锁定 0.05，见 baselines.md 各节「锁定项」），
# 每个重复跑完立刻追加落盘（断点续跑：重跑时已存在的 (rep, method) 组合自动跳过）。
#
# 用法：
#   Rscript r_baselines_full.R <exchange_dir> <cell_label> <methods> <rep_start> <rep_end> <out_csv>
#     exchange_dir : 交换目录，内含 cell<label>_rep<r>_{counts,meta}.csv
#     cell_label   : 格子标签（config 的 cell_id，或 "null" 全局零假设格）
#     methods      : 逗号分隔，子集 of ancombc2,linda,deseq2,corncob
#     rep_start:rep_end : 重复编号区间（含端点，0 基）
#     out_csv      : 回收 CSV（baselines.md 统一回收格式 + qvalue_bh 敏感性列）
#
# 输出列：cell, rep, method, taxon, rejected, pvalue, qvalue, effect_est, qvalue_bh
# 错误处理：逐 (rep, method) tryCatch，失败记入 <out_csv>.errors.csv 并继续；
#           结束时打印错误计数（不静默跳过）。

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop("用法: Rscript r_baselines_full.R <exchange_dir> <cell_label> <methods> <rep_start> <rep_end> <out_csv>")
}
exdir     <- args[1]
cell      <- args[2]
methods   <- strsplit(args[3], ",", fixed = TRUE)[[1]]
rep_start <- as.integer(args[4])
rep_end   <- as.integer(args[5])
out_csv   <- args[6]
err_csv   <- paste0(out_csv, ".errors.csv")

ALPHA <- 0.05  # 名义 FDR 锁定项（baselines.md 执行注意 1）

# 按需加载包（省内存：2 核 4GB 容器，每个 worker 只加载自己方法需要的包）
suppressPackageStartupMessages({
  if (any(methods %in% c("ancombc2", "corncob"))) library(phyloseq)
  if ("ancombc2" %in% methods) library(ANCOMBC)
  if ("corncob"  %in% methods) library(corncob)
  if ("linda"    %in% methods) library(MicrobiomeStat)
  if ("deseq2"   %in% methods) library(DESeq2)
})

# ---- 断点续跑：读取已有输出，跳过已完成的 (rep, method) --------------------
done <- data.frame(rep = integer(0), method = character(0))
if (file.exists(out_csv)) {
  prev <- tryCatch(read.csv(out_csv, stringsAsFactors = FALSE),
                   error = function(e) NULL)
  if (!is.null(prev) && nrow(prev) > 0) {
    done <- unique(prev[, c("rep", "method")])
  }
}
is_done <- function(rep, method) {
  nrow(done) > 0 && any(done$rep == rep & done$method == method)
}

append_rows <- function(df, path) {
  write.table(df, path, sep = ",", row.names = FALSE, col.names = !file.exists(path),
              append = file.exists(path), qmethod = "double")
}
log_error <- function(rep, method, msg) {
  df <- data.frame(cell = cell, rep = rep, method = method,
                   error = gsub("[\r\n]", " ", msg))
  append_rows(df, err_csv)
}

# ---- 单方法封装：输入 counts(样本×类群) 与 meta，输出统一回收格式 ----------
mk_df <- function(method, taxon, rejected, pvalue, qvalue, effect_est,
                  qvalue_bh = NA_real_) {
  data.frame(cell = cell, method = method, taxon = taxon,
             rejected = as.logical(rejected), pvalue = as.numeric(pvalue),
             qvalue = as.numeric(qvalue), effect_est = as.numeric(effect_est),
             qvalue_bh = as.numeric(qvalue_bh))
}

run_ancombc2 <- function(counts, meta) {
  # baselines.md sec.2：group 路径、struc_zero=FALSE、默认 Holm + prv_cut=0.10
  phy <- phyloseq(otu_table(t(counts), taxa_are_rows = TRUE), sample_data(meta))
  fit <- ancombc2(data = phy, fix_formula = "group", group = "group",
                  struc_zero = FALSE, prv_cut = 0.10, lib_cut = 0,
                  alpha = ALPHA, p_adj_method = "holm", verbose = FALSE)
  res <- fit$res
  diff_col <- intersect(c("diff_group1", "diff_abn_group1"), names(res))[1]
  mk_df("ancombc2", res$taxon, res[[diff_col]], res$p_group1, res$q_group1,
        res$lfc_group1, qvalue_bh = p.adjust(res$p_group1, method = "BH"))
}

run_linda <- function(counts, meta) {
  # baselines.md sec.3 的 LinDA；版本适配（详见 MEMO 版本差异节）：
  # conda r-microbiomestat 1.4 与文档记录的 1.2 时代签名/默认不同——
  # (a) feature.dat 要求 类群×样本（转置）；
  # (b) 1.4 工厂默认 prev.filter=0，实测在高过离散格大规模报错
  #     （"NA/NaN/Inf in 'y'"，证据见 *_pf0.errors.csv），故按 baselines.md
  #     记录的 LinDA 默认 prev.cut=0.1 显式设 prev.filter=0.1；
  # (c) 1.4 移除了 lib.cut，且零文库样本会使其比例变换产生 NaN 而报错；
  #     按最小适配原则仅剔除文库量为 0 的样本（与 baselines_py TSS 基线同
  #     规则，baselines.md 执行注意 2 要求记录过滤数），剔除数写入 sidecar
  #     <out_csv>.dropped.csv。其余全部 1.4 默认，alpha=0.05 锁定。
  keep <- rowSums(counts) > 0
  n_dropped <- sum(!keep)
  if (n_dropped > 0) {
    counts <- counts[keep, , drop = FALSE]
    meta <- meta[keep, , drop = FALSE]
  }
  fit <- linda(feature.dat = as.data.frame(t(counts)), meta.dat = meta,
               formula = "~ group", alpha = ALPHA, prev.filter = 0.1,
               verbose = FALSE)
  out <- fit$output$group1
  df <- mk_df("linda", rownames(out), out$reject, out$pvalue, out$padj,
              out$log2FoldChange)
  attr(df, "n_dropped") <- n_dropped
  df
}

run_deseq2 <- function(counts, meta) {
  # baselines.md sec.1：默认 fitType=parametric、independentFiltering=TRUE；
  # 锁定项 alpha=0.05（默认 0.1 必须显式改）
  # 官方 workaround（DESeq2 FAQ/vignette，微生物组惯例）：每个类群都含零时
  # 默认几何均值 size factor 无法计算，改用 poscounts（仅正计数的几何均值）。
  # 这是 FAQ 对该报错给出的官方解法，不改变 DESeq2 的检验流程与其余默认。
  dds <- DESeqDataSetFromMatrix(countData = t(counts), colData = meta,
                                design = ~ group)
  dds <- estimateSizeFactors(dds, type = "poscounts")
  dds <- DESeq(dds, quiet = TRUE)
  res <- results(dds, alpha = ALPHA)  # pAdjustMethod="BH" 为默认
  rejected <- !is.na(res$padj) & res$padj < ALPHA  # 独立过滤 NA = 未检验 = 未拒绝
  mk_df("deseq2", rownames(res), rejected, res$pvalue, res$padj,
        res$log2FoldChange)
}

run_corncob <- function(counts, meta) {
  # baselines.md sec.6：Wald、boot=FALSE、fdr_cutoff=0.05（BH）
  phy <- phyloseq(otu_table(t(counts), taxa_are_rows = TRUE), sample_data(meta))
  cc <- differentialTest(formula = ~ group, phi.formula = ~ group,
                         formula_null = ~ 1, phi.formula_null = ~ 1,
                         data = phy, test = "Wald", boot = FALSE,
                         fdr_cutoff = ALPHA)
  tax <- names(cc$p)
  mk_df("corncob", tax, tax %in% cc$significant_taxa,
        as.numeric(cc$p), as.numeric(cc$p_fdr), NA_real_)
}

RUNNERS <- list(ancombc2 = run_ancombc2, linda = run_linda,
                deseq2 = run_deseq2, corncob = run_corncob)

# ---- 主循环：逐重复、逐方法 ------------------------------------------------
n_err <- 0L
for (rep in rep_start:rep_end) {
  stem <- file.path(exdir, paste0("cell", cell, "_rep", rep))
  counts <- as.matrix(read.csv(paste0(stem, "_counts.csv"),
                               row.names = 1, check.names = FALSE))
  meta <- read.csv(paste0(stem, "_meta.csv"))
  rownames(meta) <- meta$sample_id
  meta$group <- factor(meta$group)  # 水平 0/1，1=case（各方法 results 默认 1 vs 0）
  stopifnot(all(rownames(counts) == rownames(meta)))
  storage.mode(counts) <- "integer"

  for (m in methods) {
    if (is_done(rep, m)) next
    t0 <- proc.time()[["elapsed"]]
    res <- tryCatch(RUNNERS[[m]](counts, meta), error = function(e) e)
    el <- proc.time()[["elapsed"]] - t0
    if (inherits(res, "error")) {
      n_err <- n_err + 1L
      log_error(rep, m, conditionMessage(res))
      cat(sprintf("[cell %s rep %d] %s ERROR (%.1fs): %s\n",
                  cell, rep, m, el, conditionMessage(res)))
    } else {
      res$rep <- rep
      # LinDA 样本剔除 sidecar（baselines.md 执行注意 2：记录过滤数）
      nd <- attr(res, "n_dropped")
      if (!is.null(nd)) {
        append_rows(data.frame(cell = cell, rep = rep, method = m,
                               n_dropped = nd),
                    paste0(out_csv, ".dropped.csv"))
      }
      append_rows(res[, c("cell", "rep", "method", "taxon", "rejected",
                          "pvalue", "qvalue", "effect_est", "qvalue_bh")],
                  out_csv)
      cat(sprintf("[cell %s rep %d] %s ok %.1fs n_rej=%d\n",
                  cell, rep, m, el, sum(res$rejected, na.rm = TRUE)))
    }
  }
}
cat(sprintf("cell %s 完成：错误 %d 个（详见 %s）\n", cell, n_err, err_csv))
