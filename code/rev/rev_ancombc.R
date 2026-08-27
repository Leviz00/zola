# rev_ancombc.R -- SPEC-REV-COMPUTE R2: ANCOM-BC2 on one dataset.
# Usage: Rscript rev_ancombc.R <dsdir> <outcsv> [rand]
#   dsdir: Y.csv (n x p counts, no header), meta.csv (group,N[,subject])
#   rand = "subject" -> rand_formula = "(1|subj)" (IBDMDB clusters)
# Config: shipped defaults except struc_zero=TRUE + group (its structural-
# zero detector, the predecessor's convention); p_adj Holm (its default).
# Output: taxon (1-based col index), p, q, diff, passed, szero.

args <- commandArgs(trailingOnly = TRUE)
dsdir <- args[1]; outcsv <- args[2]
rand <- length(args) >= 3 && args[3] == "subject"

suppressMessages(library(ANCOMBC))

Y <- as.matrix(read.csv(file.path(dsdir, "Y.csv"), header = FALSE))
meta <- read.csv(file.path(dsdir, "meta.csv"))
storage.mode(Y) <- "integer"
n <- nrow(Y); p <- ncol(Y)
ft <- t(Y)                                    # taxa x samples
rownames(ft) <- paste0("t", seq_len(p))
colnames(ft) <- paste0("s", seq_len(n))
md <- data.frame(grp = factor(ifelse(meta$group == 1, "b", "a"),
                              levels = c("a", "b")),
                 dummy = 1.0)                 # >=2 cols: their [rows,] drop bug
if (rand) md$subj <- as.factor(meta$subject)
rownames(md) <- colnames(ft)

res <- tryCatch(
  ancombc2(data = ft, taxa_are_rows = TRUE, meta_data = md,
           fix_formula = "grp",
           rand_formula = if (rand) "(1|subj)" else NULL,
           group = "grp", struc_zero = TRUE,
           verbose = FALSE, n_cl = 1),
  error = function(e) { message("ancombc2 error: ", conditionMessage(e)); NULL })

pv <- rep(NA_real_, p); qv <- rep(NA_real_, p)
dv <- rep(FALSE, p); ps <- rep(FALSE, p); sz <- rep(FALSE, p)
if (!is.null(res)) {
  r <- res$res
  idx <- match(r$taxon, rownames(ft))
  pv[idx] <- r$p_grpb
  qv[idx] <- r$q_grpb
  dv[idx] <- r$diff_grpb
  ps[idx] <- r$passed_ss_grpb
  if (!is.null(res$zero_ind)) {
    zi <- res$zero_ind
    zidx <- match(zi$taxon, rownames(ft))
    sz[zidx] <- (zi[[2]] != zi[[3]])          # structural zero in one group
  }
}
write.csv(data.frame(taxon = seq_len(p), p = pv, q = qv, diff = dv,
                     passed = ps, szero = sz), outcsv, row.names = FALSE)
