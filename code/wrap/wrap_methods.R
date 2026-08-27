# wrap_methods.R -- run one external method on one dataset directory.
# Usage: Rscript wrap_methods.R <method> <dsdir> <outcsv> [nperm]
# dsdir contains Y.csv (n x p counts, header FALSE), meta.csv (group,N).
# Writes outcsv with columns: taxon (1-based), p. NA p allowed.

args <- commandArgs(trailingOnly = TRUE)
method <- args[1]; dsdir <- args[2]; outcsv <- args[3]
nperm <- ifelse(length(args) >= 4, as.integer(args[4]), 10000L)

Y <- as.matrix(read.csv(file.path(dsdir, "Y.csv"), header = FALSE))
meta <- read.csv(file.path(dsdir, "meta.csv"))
storage.mode(Y) <- "integer"
n <- nrow(Y); p <- ncol(Y)
colnames(Y) <- paste0("t", seq_len(p))
rownames(Y) <- paste0("s", seq_len(n))
grp <- as.integer(meta$group)
Nlib <- as.numeric(meta$N)

pv <- rep(NA_real_, p)

if (method == "zinq") {
  suppressMessages(library(ZINQ))
  rel <- sweep(Y, 1, pmax(Nlib, 1), "/")
  for (j in seq_len(p)) {
    pv[j] <- tryCatch({
      dat <- data.frame(y = rel[, j], X = grp)
      res <- ZINQ_tests(formula.logistic = y ~ X,
                        formula.quantile = y ~ X,
                        C = "X", y_CorD = "C", data = dat)
      as.numeric(ZINQ_combination(res, method = "Cauchy"))
    }, error = function(e) NA_real_)
  }
} else if (method == "locom") {
  suppressMessages(library(LOCOM))
  res <- tryCatch(
    locom(otu.table = Y, Y = grp, C = NULL, fdr.nominal = 0.05,
          seed = 20260819, n.cores = 2, n.perm.max = nperm),
    error = function(e) { message("LOCOM error: ", conditionMessage(e)); NULL })
  if (!is.null(res)) {
    kept <- colnames(res$p.otu)
    pv[match(kept, colnames(Y))] <- as.numeric(res$p.otu)
  }
} else if (method == "ldm") {
  suppressMessages(library(LDM))
  df <- data.frame(grp = grp)
  res <- tryCatch(
    ldm(Y ~ grp, data = df, seed = 20260819, n.cores = 2),
    error = function(e) { message("LDM error: ", conditionMessage(e)); NULL })
  if (!is.null(res)) {
    pm <- res$p.otu.omni
    if (is.null(pm)) pm <- res$p.otu.freq
    if (is.matrix(pm)) {
      kept <- colnames(pm); vals <- as.numeric(pm[1, ])
    } else {
      kept <- names(pm); vals <- as.numeric(pm)
    }
    if (is.null(kept) && length(vals) == p) kept <- colnames(Y)
    pv[match(kept, colnames(Y))] <- vals
  }
} else if (method == "linda") {
  ok <- suppressMessages(requireNamespace("MicrobiomeStat", quietly = TRUE))
  meta.dat <- data.frame(grp = factor(grp))
  rownames(meta.dat) <- rownames(Y)
  if (ok) {
    res <- tryCatch(
      MicrobiomeStat::linda(feature.dat = t(Y), meta.dat = meta.dat,
                            formula = "~grp", feature.dat.type = "count"),
      error = function(e) { message("linda error: ", conditionMessage(e)); NULL })
    if (!is.null(res)) {
      out <- res$output[[1]]
      pv[match(rownames(out), colnames(Y))] <- out$pvalue
    }
  } else {
    suppressMessages(library(LinDA))
    res <- tryCatch(
      linda(otu.tab = t(Y), meta = meta.dat, formula = "~grp"),
      error = function(e) { message("linda error: ", conditionMessage(e)); NULL })
    if (!is.null(res)) {
      out <- res$output[[1]]
      pv[match(rownames(out), colnames(Y))] <- out$pvalue
    }
  }
} else stop("unknown method")

write.csv(data.frame(taxon = seq_len(p), p = pv), outcsv, row.names = FALSE)
