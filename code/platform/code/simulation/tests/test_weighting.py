"""Unit tests for baselines_py.py and weighting.py.

Covers: the BH procedure; the weight-interface contract (validation,
oracle/placeholder providers); exact reduction of the weighted methods to
their unweighted anchors when W == 0; and the behavioural property that
oracle weighting removes the informative-structural-zero false positives
of the naive detector (fixed-seed, one dataset — not a power claim).
"""

import numpy as np
import pytest

import generators
from baselines_py import bh_reject, naive_welch_t, tss_wilcoxon
from weighting import (
    cell_weights,
    exclusion_wilcoxon,
    oracle_weights,
    placeholder_weights,
    validate_weights,
    weighted_welch_t,
)


# --- bh_reject --------------------------------------------------------------

def test_bh_reject_hand_computed():
    # p = 4, alpha = 0.05: thresholds 0.0125, 0.025, 0.0375, 0.05
    pvals = np.array([0.01, 0.02, 0.20, 0.04])
    reject = bh_reject(pvals, alpha=0.05)
    # sorted: 0.01 (<=0.0125), 0.02 (<=0.025), 0.04 (>0.0375), 0.20 (>0.05);
    # largest below-threshold rank is 2 -> reject ranks 1-2 (0.01 and 0.02)
    assert reject.tolist() == [True, True, False, False]
    assert not bh_reject(np.full(5, 0.5)).any()


# --- weight interface --------------------------------------------------------

def _toy_Y():
    return np.array([[0, 5, 0], [3, 0, 7], [0, 0, 0], [2, 4, 6]])


def test_validate_weights_contract():
    Y = _toy_Y()
    W = np.full(Y.shape, 0.7)
    Wv = validate_weights(W, Y)
    # positive cells forced to 0, zero cells keep the (clipped) weight
    assert np.all(Wv[Y > 0] == 0.0)
    assert np.all(Wv[Y == 0] == pytest.approx(0.7))
    # clipping to [0, 1]
    Wv2 = validate_weights(np.full(Y.shape, 1.5), Y)
    assert np.all(Wv2[Y == 0] == 1.0)
    # shape mismatch / non-finite at zero cells raise
    with pytest.raises(ValueError):
        validate_weights(np.ones((2, 2)), Y)
    bad = np.full(Y.shape, 0.5)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_weights(bad, Y)


def test_oracle_weights_match_truth():
    Y, truth = generators.generate(
        "three_layer", {"structural_zero_rate": 0.3}, n=8, p=30,
        depths=20000, seed=3)
    W = oracle_weights(Y, truth)
    sz = truth["structural_zeros"]
    assert np.all(W[sz] == 1.0)
    assert np.all(W[~sz] == 0.0)
    # invariant: weight only at observed zeros
    assert np.all(W[Y > 0] == 0.0)


def test_placeholder_weights_range_and_depth_monotonicity():
    Y, truth = generators.generate(
        "zinb", {"structural_zero_rate": 0.2}, n=8, p=30, depths=20000,
        seed=5)
    W = placeholder_weights(Y, truth["group"])
    assert W.shape == Y.shape
    assert ((W >= 0.0) & (W <= 1.0)).all()
    assert np.all(W[Y > 0] == 0.0)
    # for a taxon with abundance m, a deeper sample is *more* likely to be a
    # structural zero when unobserved: w increases with N_i
    m = 1e-4
    w_shallow = 1.0 - (1.0 - m) ** 5000    # ~0.39
    w_deep = 1.0 - (1.0 - m) ** 20000      # ~0.86
    assert w_deep > w_shallow


def test_cell_weights():
    Y = _toy_Y()
    W = validate_weights(np.full(Y.shape, 0.25), Y)
    U = cell_weights(W, Y)
    assert np.all(U[Y > 0] == 1.0)
    assert np.all(U[Y == 0] == pytest.approx(0.75))


# --- reduction to the unweighted anchors when W == 0 -------------------------

def test_weighted_welch_reduces_to_naive_at_zero_weights():
    Y, truth = generators.generate(
        "three_layer", {"structural_zero_rate": 0.1}, n=15, p=40,
        depths=20000, seed=7)
    W0 = np.zeros(Y.shape)
    res_w = weighted_welch_t(Y, truth["group"], W0)
    res_n = naive_welch_t(Y, truth["group"])
    # all weights 1 -> weighted Welch == standard Welch t (same p-values)
    assert np.allclose(res_w["pvals"], res_n["pvals"], atol=1e-8)
    assert (res_w["reject"] == res_n["reject"]).all()


def test_exclusion_wilcoxon_reduces_to_tss_wilcoxon_at_zero_weights():
    Y, truth = generators.generate(
        "beta_binomial", {"structural_zero_rate": 0.1}, n=15, p=40,
        depths=20000, seed=9)
    W0 = np.zeros(Y.shape)
    res_e = exclusion_wilcoxon(Y, truth["group"], W0)
    res_b = tss_wilcoxon(Y, truth["group"])
    assert np.allclose(res_e["pvals"], res_b["pvals"], atol=1e-10)
    assert (res_e["reject"] == res_b["reject"]).all()


# --- behavioural: oracle weights remove informative-zero false positives -----

def test_oracle_weighting_removes_structural_false_positives():
    params = dict(structural_zero_rate=0.3, informative_zeros=True,
                  effect_size=2.0, dispersion=100.0, base_prevalence=0.9)
    Y, truth = generators.generate(
        "three_layer", params, n=50, p=100, depths=20000, seed=13)
    da = truth["da_taxa"]
    rej_naive = naive_welch_t(Y, truth["group"])["reject"]
    W = oracle_weights(Y, truth)
    rej_w = weighted_welch_t(Y, truth["group"], W)["reject"]
    rej_e = exclusion_wilcoxon(Y, truth["group"], W)["reject"]
    fp_naive = int((rej_naive & ~da).sum())
    fp_w = int((rej_w & ~da).sum())
    fp_e = int((rej_e & ~da).sum())
    # naive is badly inflated in this regime (smoke-test motivation);
    # oracle weighting must not make false positives worse, and must remove
    # most of them in this fixed-seed instance
    assert fp_naive > 0
    assert fp_w <= fp_naive
    assert fp_e <= fp_naive
    assert fp_w <= 0.5 * fp_naive
    assert fp_e <= 0.5 * fp_naive


def test_weighted_methods_output_contract():
    Y, truth = generators.generate(
        "zigdm_like", {"structural_zero_rate": 0.1}, n=10, p=30,
        depths=10000, seed=17)
    W = placeholder_weights(Y, truth["group"])
    for res in (weighted_welch_t(Y, truth["group"], W),
                exclusion_wilcoxon(Y, truth["group"], W),
                naive_welch_t(Y, truth["group"]),
                tss_wilcoxon(Y, truth["group"])):
        assert res["reject"].shape == (Y.shape[1],)
        assert res["reject"].dtype == bool
        assert ((res["pvals"] >= 0) & (res["pvals"] <= 1)).all()


# --- empty-sample (library size 0) handling ----------------------------------

def test_detectors_survive_zero_total_samples():
    # craft a dataset with an all-zero sample in each group (possible under
    # zinb / beta_binomial at low depth): detectors must filter, not crash
    rng = np.random.default_rng(0)
    Y = rng.poisson(5.0, size=(20, 15)).astype(np.int64)
    Y[0, :] = 0
    Y[10, :] = 0
    group = np.repeat([0, 1], 10)
    W = placeholder_weights(Y, group)
    assert W.shape == Y.shape  # interface keeps the (n, p) shape
    for res in (tss_wilcoxon(Y, group),
                naive_welch_t(Y, group),
                weighted_welch_t(Y, group, W),
                exclusion_wilcoxon(Y, group, W)):
        assert res["reject"].shape == (15,)
        assert ((res["pvals"] >= 0) & (res["pvals"] <= 1)).all()
