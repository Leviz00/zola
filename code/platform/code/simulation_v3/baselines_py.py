"""Pure-Python DA baselines (no R dependency).

First end-to-end baseline of the simulation study: **TSS + Wilcoxon + BH**
(baseline 7 of ``baselines.md``, anchored as the "most naive practice" lower
bound in paper section 6.4).  The R-side baselines (DESeq2, ANCOM-BC2, LinDA,
LOCOM, LDM, corncob) remain specified in ``baselines.md``; this module is the
pure-Python track so that the fractional 48-cell screen can run before an R
environment is available.

Method definition (locked, mirrors baselines.md section 7)
----------------------------------------------------------
1. TSS normalisation: ``rel_ij = Y_ij / sum_k Y_ik`` (no pseudo-count; the
   Wilcoxon rank-sum test is invariant to per-group monotone transforms of
   the values, but TSS removes the library-size scaling, which is the point
   of the baseline).
2. Per taxon: two-sided Mann-Whitney / Wilcoxon rank-sum test of
   case vs. control relative abundances (asymptotic, no exact p-values).
   Taxa constant within the union of both groups (all-zero taxa) give
   p = 1 (they carry no DA signal; "untested = not rejected" convention of
   baselines.md note 2).
3. Benjamini-Hochberg at the nominal level ALPHA = 0.05 (locked for every
   baseline, cf. baselines.md note 1).

Also provided: the naive Welch t on log10 relative abundances used by
``smoke_test.py`` (kept here so the weighted prototype in ``weighting.py``
can compare against exactly the same naive detector family).

Statistic definitions (for the preregistration, paper section 6.4)
------------------------------------------------------------------
* per-replicate FDP = #(false rejections)/#rejections, 0 if no rejections
  (``metrics.fdp``); empirical FDR of a cell = mean of per-replicate FDPs
  (per-replicate FDP mean, *not* pooled rejections), with Monte Carlo SE
  sd(FDP)/sqrt(R) (``metrics.empirical_rate``).
* per-replicate TPR = #(true DA rejected)/#(true DA); cell power = mean TPR.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "ALPHA",
    "bh_reject",
    "tss_relative_abundance",
    "tss_wilcoxon",
    "naive_welch_t",
    "DetectorResult",
]

ALPHA = 0.05  # nominal FDR level, locked across all baselines (baselines.md)


class DetectorResult(dict):
    """Dict-like result of a detector: ``reject`` (p,) bool, ``pvals`` (p,),
    plus method-specific extras (effect size estimates etc.)."""


def bh_reject(pvals, alpha=ALPHA):
    """Benjamini-Hochberg rejection mask at level ``alpha``."""
    pvals = np.asarray(pvals, dtype=float)
    p = pvals.size
    order = np.argsort(pvals, kind="mergesort")
    ranked = pvals[order]
    thresh = alpha * np.arange(1, p + 1) / p
    below = ranked <= thresh
    reject = np.zeros(p, dtype=bool)
    if below.any():
        k = int(np.max(np.where(below)))
        reject[order[: k + 1]] = True
    return reject


def tss_relative_abundance(Y):
    """Total-sum scaling: rows of ``Y`` normalised to sum 1."""
    Y = np.asarray(Y, dtype=float)
    totals = Y.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("a sample has zero total counts")
    return Y / totals


def filter_empty_samples(Y, group, W=None):
    """Drop samples with library size 0 (no information for any test).

    This is the pure-Python analogue of the library-size filtering every
    R baseline applies (LinDA ``lib.cut``, LOCOM/LDM prevalence filters,
    cf. baselines.md note 2): a sample with zero total counts carries no
    DA signal.  Returns ``(Y, group, W, n_dropped)``; the number dropped is
    recorded by the callers so result tables can footnote the filtering
    rate per cell.
    """
    Y = np.asarray(Y)
    group = np.asarray(group)
    keep = Y.sum(axis=1) > 0
    n_dropped = int((~keep).sum())
    W = None if W is None else np.asarray(W)[keep]
    return Y[keep], group[keep], W, n_dropped


def tss_wilcoxon(Y, group, alpha=ALPHA):
    """TSS + per-taxon Wilcoxon rank-sum + BH baseline.

    Parameters
    ----------
    Y     : int array (2n, p) raw counts
    group : int array (2n,), 0 = control, 1 = case
    alpha : nominal FDR level (default 0.05, locked)

    Returns
    -------
    DetectorResult with keys ``reject`` (p,) bool, ``pvals`` (p,),
    ``log2fc`` (p,) log2 ratio of group mean relative abundances
    (with a pseudo-count of 0.5 / depth scale, for the bias metric only).
    """
    Y = np.asarray(Y)
    group = np.asarray(group)
    Y, group, _, n_dropped = filter_empty_samples(Y, group)
    rel = tss_relative_abundance(Y)
    a = rel[group == 0]
    b = rel[group == 1]
    p = Y.shape[1]
    pvals = np.ones(p)
    for j in range(p):
        x, y = b[:, j], a[:, j]
        # taxa with no variation across all samples carry no signal
        if np.all(x == x[0]) and np.all(y == y[0]) and x[0] == y[0]:
            continue
        try:
            _, pv = stats.mannwhitneyu(x, y, alternative="two-sided")
        except ValueError:
            pv = 1.0
        pvals[j] = pv if np.isfinite(pv) else 1.0
    reject = bh_reject(pvals, alpha)

    totals = Y.sum(axis=1, keepdims=True).astype(float)
    rel_pc = (Y + 0.5) / (totals + 0.5 * Y.shape[1])
    log2fc = (np.log2(rel_pc[group == 1]).mean(axis=0)
              - np.log2(rel_pc[group == 0]).mean(axis=0))
    return DetectorResult(reject=reject, pvals=pvals, log2fc=log2fc,
                          method="tss_wilcoxon", alpha=alpha,
                          n_dropped=n_dropped)


def naive_welch_t(Y, group, alpha=ALPHA):
    """Naive Welch t-test on log10 relative abundances + BH.

    This is the same detector family as the ``smoke_test.py`` reference
    detector (pseudo-count 0.5, Welch t, BH at ``alpha``); it serves as the
    unweighted anchor for the weighted prototype in ``weighting.py``.
    """
    Y = np.asarray(Y)
    group = np.asarray(group)
    Y, group, _, n_dropped = filter_empty_samples(Y, group)
    totals = Y.sum(axis=1, keepdims=True).astype(float)
    rel = (Y + 0.5) / (totals + 0.5 * Y.shape[1])
    log_rel = np.log10(rel)
    a = log_rel[group == 0]
    b = log_rel[group == 1]
    _, pvals = stats.ttest_ind(b, a, axis=0, equal_var=False)
    pvals = np.where(np.isnan(pvals), 1.0, pvals)
    reject = bh_reject(pvals, alpha)
    log2fc = (np.log2(rel[group == 1]).mean(axis=0)
              - np.log2(rel[group == 0]).mean(axis=0))
    return DetectorResult(reject=reject, pvals=pvals, log2fc=log2fc,
                          method="naive_welch_t", alpha=alpha,
                          n_dropped=n_dropped)
