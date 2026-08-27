# v3 考场外部基线 caller：LinDA / ANCOM-BC2（ZOLA 项目）。
#
# 接口契约：analysis/v3_baselines/SPEC.md §1（输入）§2（输出）。
# 参数锁定与 r_baselines_full/MEMO §2/§3 完全一致（不改）：
#   ANCOM-BC2：官方默认（struc_zero=FALSE、prv_cut=0.10、lib_cut=0、
#     p_adj_method="holm"、alpha=0.05）；q_value=q_holm 同源，另导 q_bh。
#   LinDA（MicrobiomeStat 1.4 签名适配，MEMO §3.2）：feature.dat 类群×样本、
#     alpha=0.05、prev.filter=0.1、其余 1.4 默认；仅剔除文库量=0 样本
#     （剔除数写 sidecar .dropped.csv）。
#
# 用法：
#   Rscript run_v3_baselines.R <exchange_dir> <raw_dir> <cell_id> <method> [rep_start] [rep_end]
#     exchange_dir : 含 counts/ meta/ truth/ 子目录（truth 本脚本不读）
#     raw_dir      : 输出根，落 raw_dir/<method>/<cell_id>_rep<r>.csv
#     method       : linda | ancombc2
#     rep_start/end: 默认 0..19（含端点）
#
# 输出列（SPEC §2）：
#   公共：taxon,p_value,q_value,rejected(0/1) + filtered(TRUE/FALSE)
#   ancombc2 追加：q_holm,q_bh
#   被方法过滤/未检验类群：保留行，p_value=NA,q_value=NA,rejected=0,filtered=TRUE
#   重复级失败：raw_dir/<method>/<cell_id>_rep<r>.errors.csv，主 CSV 缺失
# 断点续跑：主 CSV 或 errors.csv 已存在的重复自动跳过。

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("用法: Rscript run_v3_baselines.R <exchange_dir> <raw_dir> <cell_id> <method> [rep_start] [rep_end]")
}
exdir <- args[1]
rawdir <- args[2]
cell  <- args[3]
method <- args[4]
rep_start <- if (length(args) >= 5) as.integer(args[5]) else 0L
rep_end   <- if (length(args) >= 6) as.integer(args[6]) else 19L

ALPHA <- 0.05  # 名义 FDR 锁定项

suppressPackageStartupMessages({
  if (method == "ancombc2") { library(phyloseq); library(ANCOMBC) }
  if (method == "linda")    { library(MicrobiomeStat) }
  if (method == "deseq2")   { library(DESeq2) }
})

out_dir <- file.path(rawdir, method)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# ---- 单方法封装：counts(样本×类群) + meta → 统一格式 -----------------------

run_ancombc2 <- function(counts, meta) {
  # MEMO §2/§3.3：group 路径、官方默认 Holm；另导 BH 敏感性列
  phy <- phyloseq(otu_table(t(counts), taxa_are_rows = TRUE),
                  sample_data(meta))
  fit <- ancombc2(data = phy, fix_formula = "group", group = "group",
                  struc_zero = FALSE, prv_cut = 0.10, lib_cut = 0,
                  alpha = ALPHA, p_adj_method = "holm", verbose = FALSE)
  res <- fit$res
  diff_col <- intersect(c("diff_group1", "diff_abn_group1"), names(res))[1]
  data.frame(taxon = res$taxon, p_value = as.numeric(res$p_group1),
             q_value = as.numeric(res$q_group1),
             rejected = as.integer(res[[diff_col]]),
             filtered = FALSE,
             q_holm = as.numeric(res$q_group1),
             q_bh = as.numeric(p.adjust(res$p_group1, method = "BH")),
             stringsAsFactors = FALSE)
}

run_linda <- function(counts, meta) {
  # MEMO §2/§3.2：仅剔除文库量=0 样本（剔除数 sidecar 记录），
  # prev.filter=0.1 显式，其余 1.4 默认，alpha=0.05、BH（1.4 默认）
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
  df <- data.frame(taxon = rownames(out), p_value = as.numeric(out$pvalue),
                   q_value = as.numeric(out$padj),
                   rejected = as.integer(out$reject), filtered = FALSE,
                   stringsAsFactors = FALSE)
  attr(df, "n_dropped") <- n_dropped
  df
}

run_deseq2 <- function(counts, meta) {
  # MEMO §2/§3.1 + SPEC 附录 A 锁定：poscounts size factor（官方 FAQ 对
  # 全零行几何均值不可算的解法）、design ~ group、results(alpha=0.05)
  # 默认 independentFiltering=TRUE + BH；NA p 值（独立过滤/全零行）
  # → filtered=TRUE；group 因子自然序（0 基准备），results 默认 1 vs 0
  dds <- DESeqDataSetFromMatrix(countData = t(counts), colData = meta,
                                design = ~ group)
  dds <- estimateSizeFactors(dds, type = "poscounts")
  dds <- DESeq(dds, quiet = TRUE)
  res <- results(dds, alpha = ALPHA)
  filt <- is.na(res$padj)
  data.frame(taxon = rownames(res), p_value = as.numeric(res$pvalue),
             q_value = as.numeric(res$padj),
             rejected = as.integer(!filt & res$padj < ALPHA),
             filtered = filt, stringsAsFactors = FALSE)
}

RUNNER <- switch(method, linda = run_linda, ancombc2 = run_ancombc2,
                 deseq2 = run_deseq2,
                 stop("未知 method: ", method))

# ---- 主循环：逐重复 --------------------------------------------------------
n_err <- 0L
for (rep in rep_start:rep_end) {
  stem <- paste0("cell", cell, "_rep", rep)
  out_csv <- file.path(out_dir, paste0(stem, ".csv"))
  err_csv <- file.path(out_dir, paste0(stem, ".errors.csv"))
  if (file.exists(out_csv) || file.exists(err_csv)) next  # 断点续跑

  counts <- as.matrix(read.csv(file.path(exdir, "counts",
                                         paste0(stem, ".csv")),
                               row.names = 1, check.names = FALSE))
  meta <- read.csv(file.path(exdir, "meta", paste0(stem, ".csv")))
  rownames(meta) <- meta$sample
  meta$group <- factor(meta$group)  # 水平 0/1，1=case
  stopifnot(all(rownames(counts) == rownames(meta)))
  storage.mode(counts) <- "integer"
  all_taxa <- colnames(counts)

  t0 <- proc.time()[["elapsed"]]
  res <- tryCatch(RUNNER(counts, meta), error = function(e) e)
  el <- proc.time()[["elapsed"]] - t0

  if (inherits(res, "error")) {
    n_err <- n_err + 1L
    write.csv(data.frame(cell_id = cell, rep = rep, method = method,
                         error = gsub("[\r\n]", " ", conditionMessage(res))),
              err_csv, row.names = FALSE)
    cat(sprintf("[%s rep %d] %s ERROR (%.1fs): %s\n",
                cell, rep, method, el, conditionMessage(res)))
    next
  }

  # LinDA 样本剔除数（rbind 会丢属性，先取出）
  nd <- attr(res, "n_dropped")

  # 未检验/被过滤类群：补 NA 行 + filtered=TRUE（SPEC §2）
  tested <- res$taxon
  missing <- setdiff(all_taxa, tested)
  if (length(missing) > 0) {
    na_row <- data.frame(taxon = missing, p_value = NA_real_,
                         q_value = NA_real_, rejected = 0L,
                         filtered = TRUE, stringsAsFactors = FALSE)
    extra_cols <- setdiff(names(res), names(na_row))
    for (cc in extra_cols) na_row[[cc]] <- NA_real_
    res <- rbind(res, na_row[, names(res)])
  }
  # 排序与真值表一致（taxon_1..taxon_p 数值序）
  ord <- order(as.integer(sub("^taxon_", "", res$taxon)))
  res <- res[ord, ]
  write.csv(res, out_csv, row.names = FALSE, na = "NA")

  if (!is.null(nd)) {
    write.csv(data.frame(cell_id = cell, rep = rep, method = method,
                         n_dropped = nd),
              file.path(out_dir, paste0(stem, ".dropped.csv")),
              row.names = FALSE)
  }
  cat(sprintf("[%s rep %d] %s ok %.1fs n_rej=%d n_filtered=%d\n",
              cell, rep, method, el, sum(res$rejected, na.rm = TRUE),
              length(missing)))
}
cat(sprintf("cell %s %s 完成：错误 %d 个\n", cell, method, n_err))
     filtered = TRUE, stringsAsFactors = FALSE)
    extra_cols <- setdiff(names(res), names(na_row))
    for (cc in extra_cols) na_row[[cc]] <- NA_real_
    res <- rbind(res, na_row[, names(res)])
  }
  # 排序与真值表一致（taxon_1..taxon_p 数值序）
  ord <- order(as.integer(sub("^taxon_", "", res$taxon)))
  res <- res[ord, ]
  write.csv(res, out_csv, row.names = FALSE, na = "NA")

  if (!is.null(nd)) {
    write.csv(data.frame(cell_id = cell, rep = rep, method = method,
                         n_dropped = nd),
              file.path(out_dir, paste0(stem, ".dropped.csv")),
              row.names = FALSE)
  }
  cat(sprintf("[%s rep %d] %s ok %.1fs n_rej=%d n_filtered=%d\n",
              cell, rep, method, el, sum(res$rejected, na.rm = TRUE),
              length(missing)))
}
cat(sprintf("cell %s %s 完成：错误 %d 个\n", cell, method, n_err))
