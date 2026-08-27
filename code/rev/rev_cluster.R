# rev_cluster.R -- SPEC-REV-COMPUTE R1: cluster-capable modes on IBDMDB.
# Usage: Rscript rev_cluster.R <method> <dsdir> <outcsv>
# method: linda_mix | ldm_cluster
# dsdir: Y.csv (n x p, no header), meta.csv (group,N,subject)

args <- commandArgs(trailingOnly = TRUE)
method <- args[1]; dsdir <- args[2]; outcsv <- args[3]

Y <- as.matrix(read.csv(file.path(dsdir, "Y.csv"), header = FALSE))
meta <- read.csv(file.path(dsdir, "meta.csv"))
storage.mode(Y) <- "integer"
n <- nrow(Y); p <- ncol(Y)
colnames(Y) <- paste0("t", seq_len(p))
rownames(Y) <- paste0("s", seq_len(n))
grp <- as.integer(meta$group)
subj <- as.factor(meta$subject)

pv <- rep(NA_real_, p)

if (method == "linda_mix") {
  suppressMessages(library(LinDA))
  md <- data.frame(grp = factor(grp), subj = subj)
  rownames(md) <- rownames(Y)
  res <- tryCatch(
    linda(otu.tab = t(Y), meta = md, formula = "~grp+(1|subj)",
          n.cores = 2),
    error = function(e) { message("linda_mix error: ",
                                  conditionMessage(e)); NULL })
  if (!is.null(res)) {
    out <- res$output[[1]]
    pv[match(rownames(out), colnames(Y))] <- out$pvalue
  }
} else if (method == "ldm_cluster") {
  suppressMessages(library(LDM))
  df <- data.frame(grp = grp, subj = subj)
  # grp is constant within subject: permute between clusters, none within
  res <- tryCatch(
    ldm(Y ~ grp, data = df, seed = 20260819, n.cores = 2,
        cluster.id = subj, perm.within.type = "none",
        perm.between.type = "free"),
    error = function(e) { message("ldm_cluster error: ",
                                  conditionMessage(e)); NULL })
  if (!is.null(res)) {
    pm <- res$p.otu.omni
    if (is.null(pm)) pm <- res$p.otu.freq
    if (is.matrix(pm)) { kept <- colnames(pm); vals <- as.numeric(pm[1, ]) }
    else { kept <- names(pm); vals <- as.numeric(pm) }
    if (is.null(kept) && length(vals) == p) kept <- colnames(Y)
    pv[match(kept, colnames(Y))] <- vals
  }
}

write.csv(data.frame(taxon = seq_len(p), p = pv), outcsv, row.names = FALSE)
