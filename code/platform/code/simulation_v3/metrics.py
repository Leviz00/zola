"""Evaluation metrics for the simulation study (paper section 6.2/6.4).

Metrics
-------
* empirical FDR (nominal 0.05) and power (TPR) of a DA procedure;
* effect-size estimation bias (on the log2 fold-change scale);
* zero-source classification: AUC, Brier score and reliability-diagram
  data for the posterior probability P(structural zero | Y = 0).

All functions are deliberately dependency-light (numpy only) and are
covered by hand-computed unit tests in ``tests/test_metrics.py``.

Conventions
-----------
* ``fdp`` of a replicate with no rejections is 0 (the standard
  positive-FDR-style convention used in benchmark studies); the number of
  rejections is returned so callers can switch conventions.
* AUC is the Mann-Whitney/rank statistic with mid-ranks for ties; it is
  defined only when both classes are present (``ValueError`` otherwise).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "fdp",
    "tpr",
    "empirical_rate",
    "effect_size_bias",
    "auc",
    "brier",
    "reliability_data",
    "zero_source_metrics",
]


# ---------------------------------------------------------------------------
# DA-error metrics
# ---------------------------------------------------------------------------

def fdp(discovered, truth):
    """False discovery proportion of one replicate.

    Parameters
    ----------
    discovered : bool array (p,), rejected null hypotheses
    truth      : bool array (p,), truly DA taxa

    Returns
    -------
    (fdp, n_rejections) : fdp = #(false rejections) / #rejections, 0 if none.
    """
    discovered = np.asarray(discovered, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if discovered.shape != truth.shape:
        raise ValueError("discovered and truth shapes disagree")
    n_rej = int(discovered.sum())
    if n_rej == 0:
        return 0.0, 0
    n_false = int((discovered & ~truth).sum())
    return n_false / n_rej, n_rej


def tpr(discovered, truth):
    """True positive rate (power) of one replicate.

    Returns (tpr, n_true); tpr is NaN when there are no true DA taxa.
    """
    discovered = np.asarray(discovered, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if discovered.shape != truth.shape:
        raise ValueError("discovered and truth shapes disagree")
    n_true = int(truth.sum())
    if n_true == 0:
        return float("nan"), 0
    return float((discovered & truth).sum() / n_true), n_true


def empirical_rate(values):
    """Mean and Monte Carlo standard error of per-replicate rates.

    The empirical FDR over R replicates is the mean of the per-replicate
    FDPs; its Monte Carlo SE is sd(FDP)/sqrt(R) (sample sd, ddof=1).
    NaNs (e.g. TPR of replicates without true positives) are dropped.
    """
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return float("nan"), float("nan"), 0
    if v.size == 1:
        return float(v[0]), float("nan"), 1
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)), int(v.size)


def effect_size_bias(log2fc_hat, log2fc_true, mask=None):
    """Bias of estimated log2 fold changes.

    Returns dict with mean bias (hat - true), mean *relative* bias
    (hat/true - 1, only over entries with |true| > 0), rmse and the number
    of taxa used.  ``mask`` restricts to a subset (typically the DA taxa).
    """
    hat = np.asarray(log2fc_hat, dtype=float)
    true = np.asarray(log2fc_true, dtype=float)
    if hat.shape != true.shape:
        raise ValueError("hat and true shapes disagree")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        hat, true = hat[mask], true[mask]
    if hat.size == 0:
        raise ValueError("no taxa selected")
    err = hat - true
    nz = np.abs(true) > 0
    rel = (hat[nz] / true[nz] - 1.0) if nz.any() else np.array([np.nan])
    return {
        "bias": float(err.mean()),
        "rel_bias": float(np.nanmean(rel)),
        "rmse": float(np.sqrt((err**2).mean())),
        "n": int(hat.size),
    }


# ---------------------------------------------------------------------------
# zero-source classification metrics
# ---------------------------------------------------------------------------

def auc(scores, labels):
    """Mann-Whitney AUC of ``scores`` for the positive class ``labels``.

    AUC = P(score_pos > score_neg) + 0.5 * P(tie), computed exactly via
    mid-ranks.  Raises ValueError if a class is empty.
    """
    s = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=bool)
    if s.shape != y.shape:
        raise ValueError("scores and labels shapes disagree")
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC undefined: both classes must be present")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    ranks[order] = np.arange(1, s.size + 1)
    # mid-ranks for ties
    sorted_s = s[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    rank_sum_pos = ranks[y].sum()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def brier(probs, labels):
    """Brier score mean((p - y)^2) of predicted probabilities."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.shape != y.shape:
        raise ValueError("probs and labels shapes disagree")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    return float(np.mean((p - y) ** 2))


def reliability_data(probs, labels, n_bins=10):
    """Reliability-diagram (calibration curve) data.

    Equal-width bins over [0, 1].  Returns a dict of arrays: bin left/right
    edges, mean predicted probability, empirical event frequency, and bin
    counts (empty bins give NaN frequencies).  Calibration-in-the-large and
    the expected calibration error (ECE, count-weighted mean |gap|) are
    included for convenience.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.shape != y.shape:
        raise ValueError("probs and labels shapes disagree")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    mean_pred = np.full(n_bins, np.nan)
    frac_pos = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = idx == b
        count[b] = int(m.sum())
        if m.any():
            mean_pred[b] = float(p[m].mean())
            frac_pos[b] = float(y[m].mean())
    valid = count > 0
    ece = float(
        np.nansum(count[valid] * np.abs(mean_pred[valid] - frac_pos[valid]))
        / max(count.sum(), 1)
    )
    return {
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "mean_pred": mean_pred,
        "frac_pos": frac_pos,
        "count": count,
        "ece": ece,
        "calibration_in_the_large": float(y.mean() - p.mean()),
    }


def zero_source_metrics(prob_structural, structural_zeros, y, n_bins=10):
    """Zero-source classification summary restricted to observed zeros.

    Parameters
    ----------
    prob_structural   : array, estimated P(structural zero | Y = 0) per cell
    structural_zeros  : bool array, true absence indicators
    y                 : count array; only cells with y == 0 enter the metrics

    Returns dict with n_zeros, n_structural, n_sampling, auc (None if a class
    is absent), brier and the reliability-diagram data.
    """
    p = np.asarray(prob_structural, dtype=float)
    sz = np.asarray(structural_zeros, dtype=bool)
    y = np.asarray(y)
    if not (p.shape == sz.shape == y.shape):
        raise ValueError("shapes disagree")
    zero = y == 0
    labels = sz[zero]
    scores = p[zero]
    out = {
        "n_zeros": int(zero.sum()),
        "n_structural": int(labels.sum()),
        "n_sampling": int((~labels).sum()),
        "brier": brier(scores, labels) if scores.size else float("nan"),
        "reliability": reliability_data(scores, labels, n_bins)
        if scores.size
        else None,
    }
    try:
        out["auc"] = auc(scores, labels)
    except ValueError:
        out["auc"] = None
    return out
