# R-side baseline smoke run (overall_review.md C2): proves the R environment
# and the exchange format of baselines.md end-to-end on one exported dataset
# per scenario.  Methods: ANCOM-BC2, corncob, LinDA, TSS+Wilcoxon (base R).
# All at the locked nominal FDR 0.05 (baselines.md note 1); ancombc2 keeps
# its default Holm adjustment (raw p exported for the BH sensitivity).
#
# Usage:  Rscript r_baselines_smoke.R <exchange_dir>
# Reads:  cell<name>_rep0_{counts,meta}.csv ; writes
#         cell<name>_rep0_discoveries.csv  (cell, rep, method, taxon,
#         rejected, pvalue, qvalue, effect_est)

args <- commandArgs(trailingOnly = TRUE)
exdir <- if (length(args) >= 1) args[1] else "results/r_exchange"
ALPHA <- 0.05

suppressPackageStartupMessages({
  library(phyloseq)
  library(ANCOMBC)
  library(corncob)
  library(MicrobiomeStat)  # LinDA
})

run_cell <- function(stem) {
  counts <- as.matrix(read.csv(file.path(exdir, paste0(stem, "_counts.csv")),
                               row.names = 1, check.names = FALSE))
  meta <- read.csv(file.path(exdir, paste0(stem, "_meta.csv")))
  rownames(meta) <- meta$sample_id
  meta$group <- factor(meta$group)
  stopifnot(all(rownames(counts) == rownames(meta)))

  out <- list()

  # --- ANCOM-BC2 (baselines.md sec. 2): group path, struc_zero = FALSE ------
  phy <- phyloseq(otu_table(t(counts), taxa_are_rows = TRUE),
                  sample_data(meta))
  fit2 <- ancombc2(data = phy, fix_formula = "group", group = "group",
                   struc_zero = FALSE, prv_cut = 0.10, lib_cut = 0,
                   alpha = ALPHA, p_adj_method = "holm", verbose = FALSE)
  res2 <- fit2$res
  # ANCOMBC >= 2.8 names the decision column diff_group1 (older: diff_abn_group1)
  diff_col <- intersect(c("diff_group1", "diff_abn_group1"), names(res2))[1]
  out[["ancombc2"]] <- data.frame(
    method = "ancombc2", taxon = res2$taxon,
    rejected = res2[[diff_col]],
    pvalue = res2$p_group1, qvalue = res2$q_group1,
    effect_est = res2$lfc_group1)

  # --- corncob (baselines.md sec. 6): Wald, fdr_cutoff = 0.05 ---------------
  cc <- tryCatch(
    differentialTest(formula = ~ group, phi.formula = ~ group,
                     formula_null = ~ 1, phi.formula_null = ~ 1,
                     data = phy, test = "Wald", boot = FALSE,
                     fdr_cutoff = ALPHA),
    error = function(e) e)
  if (!inherits(cc, "error")) {
    sig <- cc$significant_taxa
    tax_cc <- names(cc$p)
    out[["corncob"]] <- data.frame(
      method = "corncob", taxon = tax_cc,
      rejected = tax_cc %in% sig,
      pvalue = as.numeric(cc$p), qvalue = as.numeric(cc$p_fdr),
      effect_est = NA_real_)
  } else {
    message("corncob failed on ", stem, ": ", conditionMessage(cc))
  }

  # --- LinDA (baselines.md sec. 3): defaults, alpha = 0.05 ------------------
  # 版本适配（2026-07-30 环境重建）：conda r-microbiomestat 1.4 的 linda()
  # 签名与 baselines.md 记录的 1.2 时代签名不同（见 MEMO）：
  #   feature.dat（= 旧 otu.tab，但要求 类群 × 样本）/ meta.dat；
  #   默认 prev.filter=0（旧 prev.cut=0.1）、无 lib.cut、
  #   winsor 由 is.winsor+outlier.pct=0.03 控制（旧 winsor.quan=0.97）。
  # 按「官方默认参数」原则用 1.4 全默认 + alpha=0.05 锁定。
  linda_fit <- tryCatch(
    linda(feature.dat = as.data.frame(t(counts)), meta.dat = meta,
          formula = "~ group", alpha = ALPHA, verbose = FALSE),
    error = function(e) e)
  if (!inherits(linda_fit, "error")) {
    outdf <- linda_fit$output$group1
    out[["linda"]] <- data.frame(
      method = "linda", taxon = rownames(outdf),
      rejected = outdf$reject,
      pvalue = outdf$pvalue, qvalue = outdf$padj,
      effect_est = outdf$log2FoldChange)
  } else {
    message("linda failed on ", stem, ": ", conditionMessage(linda_fit))
  }

  # --- TSS + Wilcoxon (baselines.md sec. 7, base R) --------------------------
  rel <- sweep(counts, 1, rowSums(counts), "/")
  pvals <- apply(rel, 2, function(x) {
    if (length(unique(x)) < 2) return(1.0)
    tryCatch(wilcox.test(x[meta$group == 1], x[meta$group == 0],
                         exact = FALSE)$p.value,
             error = function(e) 1.0)
  })
  out[["tss_wilcoxon_r"]] <- data.frame(
    method = "tss_wilcoxon_r", taxon = colnames(counts),
    rejected = p.adjust(pvals, method = "BH") < ALPHA,
    pvalue = pvals, qvalue = p.adjust(pvals, method = "BH"),
    effect_est = NA_real_)

  df <- do.call(rbind, out)
  df$cell <- stem
  df$rep <- 0L
  rownames(df) <- NULL
  write.csv(df, file.path(exdir, paste0(stem, "_discoveries.csv")),
            row.names = FALSE)
  cat(stem, ": methods =", paste(unique(df$method), collapse = ", "), "\n")
}

stems <- sub("_counts\\.csv$", "",
             list.files(exdir, pattern = "_counts\\.csv$"))
for (stem in stems) run_cell(stem)
cat("sessionInfo:\n")
print(sessionInfo())
