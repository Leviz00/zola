"""Posterior-weighted DA prototype (paper section 3, "research content 3").

The application-level contribution of the paper: inject the zero-source
posterior of eq. (posterior) (``sec3_framework.tex``),

    w_ij = Pr(Z_ij = 0 | Y_ij = 0),

into a downstream differential-abundance (DA) workflow, instead of treating
every observed zero as a low-abundance observation.  A zero that is likely
*structural* (w_ij ~ 1) is a presence-layer event and must not masquerade
as an abundance effect in the DA test; a zero that is likely a *sampling*
zero (w_ij ~ 0) is a genuine low-abundance observation and stays.

Weight-injection interface (contract with the estimation pipeline)
------------------------------------------------------------------
The true posterior estimator is developed in parallel by the estimation
agent; every consumer in this module depends only on the following
interface, so a calibrated estimator can be dropped in unchanged:

* a weight provider returns ``W``, a float array of shape ``(n_samples, p)``
  with entries in ``[0, 1]``;
* ``W[i, j]`` is the estimated posterior ``Pr(Z_ij = 0 | Y_ij = 0)`` — it is
  read **only where ``Y[i, j] == 0``**; entries at positive cells are
  ignored (providers should set them to 0);
* providers shipped here (placeholders / oracles, NOT the estimator):
    - ``oracle_weights(Y, truth)``   : perfect classification from the
      simulation truth (upper bound of the weighting benefit);
    - ``placeholder_weights(Y, ...)``: the smoke-test plug-in score
      (depth-vs-abundance inversion), documented placeholder quality.

Weighted methods (prototype stage)
----------------------------------
1. ``weighted_welch_t`` — per-taxon Welch t on log10 relative abundances
   with per-cell reliability weights u_ij = 1 - W_ij at zero cells (u = 1 at
   positive cells); weighted means/variances with effective sample sizes
   n_eff = (sum u)^2 / sum u^2 and a Welch-Satterthwaite df.  The weights
   are data-dependent, so the nominal p-value calibration is *not*
   guaranteed by theory — this is exactly why the prespecified FDR check
   below is run.
2. ``exclusion_wilcoxon`` — hard version of the same idea on the
   TSS+Wilcoxon baseline: cells with Y_ij == 0 and W_ij >= ``threshold``
   are treated as missing; the remaining per-group relative abundances go
   through the standard Mann-Whitney test.  Wilcoxon family, as requested.

Prespecified checks (paper section 6.4, this round = small-scale feasibility)
-----------------------------------------------------------------------------
* NON-INFORMATIVE structural zeros (absence independent of the group
  label): weight injection must NOT break FDR — criterion
  |FDR_hat - 0.05| <= 0.01 with R = n_replicates_screen = 100 replicates
  (design.py), empirical FDR = mean of per-replicate FDPs, MC SE reported
  alongside (with R = 100 the MC SE itself is ~0.01, so borderline
  decisions are reported with their SE, not hidden).
* INFORMATIVE structural zeros (absence aligned with the case group):
  quantitative report of the FDR improvement of the weighted variants
  relative to the naive detector, and the associated power loss
  (TPR_naive - TPR_weighted).  The full section-6.4 acceptance criterion
  (power gain >= 5pp AND FDR controlled) is out of scope for this round.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from baselines_py import (
    ALPHA,
    DetectorResult,
    bh_reject,
    filter_empty_samples,
    tss_relative_abundance,
)

__all__ = [
    "validate_weights",
    "oracle_weights",
    "placeholder_weights",
    "cell_weights",
    "weighted_welch_t",
    "exclusion_wilcoxon",
    "WEIGHT_PROVIDERS",
    "WEIGHTED_METHODS",
]


# ---------------------------------------------------------------------------
# weight interface
# ---------------------------------------------------------------------------

def validate_weights(W, Y, tol=1e-9):
    """Validate/clip a weight matrix against the interface contract.

    Returns a clipped float copy of ``W`` with entries at ``Y > 0`` forced
    to 0.  Raises on shape mismatch or non-finite values at zero cells.
    """
    W = np.asarray(W, dtype=float)
    Y = np.asarray(Y)
    if W.shape != Y.shape:
        raise ValueError(f"W shape {W.shape} != Y shape {Y.shape}")
    zero = Y == 0
    if not np.isfinite(W[zero]).all():
        raise ValueError("non-finite weight at an observed-zero cell")
    W = np.clip(W, 0.0, 1.0)
    W = np.where(zero, W, 0.0)
    return W


def oracle_weights(Y, truth):
    """Oracle weights: w_ij = 1 iff cell (i, j) is a true structural zero.

    Only cells with Y_ij == 0 can carry weight (the invariant
    structural zero => Y = 0 makes this consistent).
    """
    sz = np.asarray(truth["structural_zeros"], dtype=bool)
    return validate_weights(sz.astype(float), Y)


def placeholder_weights(Y, group=None):
    """Placeholder estimated weights (to be replaced by the estimator).

    Plug-in depth inversion (same score as ``smoke_test.py``): with m_j the
    pooled mean relative abundance of taxon j, a *present* taxon stays
    unobserved at depth N_i with probability ~= (1 - m_j)^{N_i}; the
    structural-zero posterior placeholder is the complement,

        w_ij = 1 - (1 - m_j)^{N_i}   (only where Y_ij == 0).

    Known placeholder weaknesses (documented, not fixed here): no
    overdispersion (uses a binomial detection law), no existence-layer
    prior pi_ij, m_j estimated from the same data (plug-in).  The real
    estimator must return the same (n, p) matrix in [0, 1].
    """
    Y = np.asarray(Y, dtype=float)
    totals = Y.sum(axis=1, keepdims=True)
    # rows with library size 0 contribute nothing to m_hat and get w = 0
    # (a zero at depth 0 is uninformative); detectors filter them anyway
    rel = np.divide(Y, totals, out=np.zeros_like(Y), where=totals > 0)
    m_hat = rel.sum(axis=0) / max(int((totals[:, 0] > 0).sum()), 1)
    n_i = totals[:, 0]
    with np.errstate(invalid="ignore"):
        p_sampling_zero = np.exp(
            n_i[:, None] * np.log1p(-np.clip(m_hat, 0.0, 1.0)[None, :])
        )
    W = 1.0 - p_sampling_zero
    return validate_weights(W, Y)


def cell_weights(W, Y):
    """Per-cell reliability weights u_ij = 1 - w_ij at zero cells, 1 else."""
    W = validate_weights(W, Y)
    return 1.0 - W


WEIGHT_PROVIDERS = {
    "oracle": oracle_weights,          # needs truth: call as oracle(Y, truth)
    "placeholder": placeholder_weights,
}


# ---------------------------------------------------------------------------
# weighted methods
# ---------------------------------------------------------------------------

def _log_rel(Y):
    Y = np.asarray(Y, dtype=float)
    totals = Y.sum(axis=1, keepdims=True)
    rel = (Y + 0.5) / (totals + 0.5 * Y.shape[1])
    return np.log10(rel)


def _weighted_mean_var(x, u):
    """Weighted mean and (moment-type, reliability-weight) variance."""
    su = u.sum()
    if su <= 0:
        return np.nan, np.nan, 0.0
    mean = float((u * x).sum() / su)
    var = float((u * (x - mean) ** 2).sum() / su)
    n_eff = su * su / float((u**2).sum())
    # Bessel-type correction for reliability weights
    if n_eff > 1:
        var *= n_eff / (n_eff - 1.0)
    return mean, var, n_eff


def weighted_welch_t(Y, group, W, alpha=ALPHA):
    """Weighted Welch t-test on log10 relative abundances + BH.

    Zero cells are downweighted by u_ij = 1 - w_ij (w_ij = posterior
    probability the zero is structural); positive cells keep weight 1.
    Per taxon: weighted group means/variances, Welch-Satterthwaite df with
    effective sample sizes.  Taxa with n_eff < 2 in either group or zero
    weighted variance in both groups get p = 1 ("untested = not rejected").
    """
    Y = np.asarray(Y)
    group = np.asarray(group)
    W = validate_weights(W, Y)
    Y, group, W, _ = filter_empty_samples(Y, group, W)
    U = 1.0 - W
    log_rel = _log_rel(Y)
    p = Y.shape[1]
    pvals = np.ones(p)
    g0, g1 = group == 0, group == 1
    for j in range(p):
        m0, v0, ne0 = _weighted_mean_var(log_rel[g0, j], U[g0, j])
        m1, v1, ne1 = _weighted_mean_var(log_rel[g1, j], U[g1, j])
        if min(ne0, ne1) < 2.0:
            continue
        se2 = v0 / ne0 + v1 / ne1
        if not np.isfinite(se2) or se2 <= 0:
            continue
        t = (m1 - m0) / np.sqrt(se2)
        df = se2**2 / (
            (v0 / ne0) ** 2 / (ne0 - 1.0) + (v1 / ne1) ** 2 / (ne1 - 1.0)
        )
        if not np.isfinite(df) or df <= 0:
            continue
        pvals[j] = 2.0 * stats.t.sf(abs(t), df)
    reject = bh_reject(pvals, alpha)
    return DetectorResult(reject=reject, pvals=pvals,
                          method="weighted_welch_t", alpha=alpha)


def exclusion_wilcoxon(Y, group, W, threshold=0.5, alpha=ALPHA):
    """Hard-exclusion weighted Wilcoxon: likely-structural zeros -> missing.

    Cells with Y_ij == 0 and w_ij >= ``threshold`` are excluded; the
    Mann-Whitney test runs on the remaining TSS relative abundances per
    group.  Taxa with fewer than 3 observations in either group get p = 1.
    """
    Y = np.asarray(Y)
    group = np.asarray(group)
    W = validate_weights(W, Y)
    Y, group, W, _ = filter_empty_samples(Y, group, W)
    rel = tss_relative_abundance(Y)
    keep = ~((Y == 0) & (W >= threshold))
    p = Y.shape[1]
    pvals = np.ones(p)
    g0, g1 = group == 0, group == 1
    for j in range(p):
        x = rel[g1 & keep[:, j], j]
        y = rel[g0 & keep[:, j], j]
        if x.size < 3 or y.size < 3:
            continue
        if np.all(x == x[0]) and np.all(y == y[0]) and x[0] == y[0]:
            continue
        try:
            _, pv = stats.mannwhitneyu(x, y, alternative="two-sided")
        except ValueError:
            pv = 1.0
        pvals[j] = pv if np.isfinite(pv) else 1.0
    reject = bh_reject(pvals, alpha)
    return DetectorResult(reject=reject, pvals=pvals,
                          method="exclusion_wilcoxon", alpha=alpha,
                          threshold=threshold)


WEIGHTED_METHODS = {
    "weighted_welch_t": weighted_welch_t,
    "exclusion_wilcoxon": exclusion_wilcoxon,
}
