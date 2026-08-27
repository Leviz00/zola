"""Unit tests for metrics.py -- all expected values hand-computed."""

import numpy as np
import pytest

import metrics


# --- fdp / tpr -------------------------------------------------------------

def test_fdp_hand_computed():
    # rejected {0,1}, truth {0,2}: one false of two rejections -> 0.5
    val, n_rej = metrics.fdp([True, True, False, False],
                             [True, False, True, False])
    assert n_rej == 2
    assert val == pytest.approx(0.5)


def test_fdp_no_rejections_is_zero():
    val, n_rej = metrics.fdp([False, False], [True, True])
    assert (val, n_rej) == (0.0, 0)


def test_fdp_all_correct():
    val, _ = metrics.fdp([True, False], [True, False])
    assert val == 0.0


def test_tpr_hand_computed():
    # truth {0,2}, discovered {0,1}: one of two found -> 0.5
    val, n_true = metrics.tpr([True, True, False, False],
                              [True, False, True, False])
    assert n_true == 2
    assert val == pytest.approx(0.5)


def test_tpr_no_true_taxa_is_nan():
    val, n_true = metrics.tpr([True], [False])
    assert n_true == 0 and np.isnan(val)


# --- empirical_rate --------------------------------------------------------

def test_empirical_rate_mean_and_se():
    # values 0.04, 0.05, 0.06: mean 0.05; sd = sqrt(1e-4) = 0.01 -> se = 0.01/sqrt(3)
    mean, se, n = metrics.empirical_rate([0.04, 0.05, 0.06])
    assert mean == pytest.approx(0.05)
    assert se == pytest.approx(0.01 / np.sqrt(3))
    assert n == 3


def test_empirical_rate_drops_nan():
    mean, se, n = metrics.empirical_rate([0.1, np.nan, 0.3])
    assert mean == pytest.approx(0.2) and n == 2


# --- effect_size_bias ------------------------------------------------------

def test_effect_size_bias_hand_computed():
    # hat [1.1, 2.1] vs true [1, 2]: errors +0.1, +0.1 -> bias 0.1
    out = metrics.effect_size_bias([1.1, 2.1], [1.0, 2.0])
    assert out["bias"] == pytest.approx(0.1)
    # relative: 1.1/1-1 = 0.1 ; 2.1/2-1 = 0.05 -> mean 0.075
    assert out["rel_bias"] == pytest.approx(0.075)
    assert out["rmse"] == pytest.approx(0.1)
    assert out["n"] == 2


def test_effect_size_bias_mask():
    out = metrics.effect_size_bias([9.0, 1.5], [0.0, 1.0], mask=[False, True])
    assert out["bias"] == pytest.approx(0.5)
    assert out["n"] == 1


# --- auc -------------------------------------------------------------------

def test_auc_hand_computed():
    # scores [0.1, 0.4, 0.35, 0.8], labels [0, 0, 1, 1]
    # ranks: 0.1->1, 0.35->2, 0.4->3, 0.8->4 ; positives at ranks 2 and 4
    # U = (2+4) - 2*3/2 = 3 ; AUC = 3/(2*2) = 0.75
    assert metrics.auc([0.1, 0.4, 0.35, 0.8],
                       [False, False, True, True]) == pytest.approx(0.75)


def test_auc_perfect_and_zero():
    assert metrics.auc([0.1, 0.2, 0.8, 0.9],
                       [False, False, True, True]) == pytest.approx(1.0)
    assert metrics.auc([0.8, 0.9, 0.1, 0.2],
                       [False, False, True, True]) == pytest.approx(0.0)


def test_auc_ties_midrank():
    # scores [0.5, 0.5], labels [0, 1]: mid-ranks 1.5,1.5 -> U = 1.5-1 = 0.5
    assert metrics.auc([0.5, 0.5], [False, True]) == pytest.approx(0.5)


def test_auc_single_class_raises():
    with pytest.raises(ValueError):
        metrics.auc([0.1, 0.2], [True, True])


# --- brier -----------------------------------------------------------------

def test_brier_hand_computed():
    # p = [0.1, 0.9], y = [0, 1]: (0.01 + 0.01)/2 = 0.01
    assert metrics.brier([0.1, 0.9], [0, 1]) == pytest.approx(0.01)


def test_brier_rejects_out_of_range():
    with pytest.raises(ValueError):
        metrics.brier([1.5], [1])


# --- reliability_data ------------------------------------------------------

def test_reliability_data_hand_computed():
    # 2 bins over [0,1]: p = [0.1, 0.3, 0.7, 0.9], y = [0, 1, 1, 1]
    # bin [0,0.5): mean_pred 0.2, frac 0.5, count 2
    # bin [0.5,1]: mean_pred 0.8, frac 1.0, count 2
    rd = metrics.reliability_data([0.1, 0.3, 0.7, 0.9], [0, 1, 1, 1], n_bins=2)
    assert rd["count"].tolist() == [2, 2]
    assert rd["mean_pred"].tolist() == pytest.approx([0.2, 0.8])
    assert rd["frac_pos"].tolist() == pytest.approx([0.5, 1.0])
    # ECE = (2*|0.2-0.5| + 2*|0.8-1.0|)/4 = (0.6+0.4)/4 = 0.25
    assert rd["ece"] == pytest.approx(0.25)
    # calibration in the large: mean y 0.75 - mean p 0.5 = 0.25
    assert rd["calibration_in_the_large"] == pytest.approx(0.25)


def test_reliability_empty_bins_nan():
    # binning convention: [left, right) with the right edge belonging to the
    # next bin, so 0.1 falls in the second bin [0.1, 0.2)
    rd = metrics.reliability_data([0.1], [1], n_bins=10)
    assert np.isnan(rd["frac_pos"][0])
    assert rd["count"][1] == 1


# --- zero_source_metrics ---------------------------------------------------

def test_zero_source_metrics_hand_computed():
    # Y = [[0, 5, 0], [3, 0, 0]]; structural = [[T, F, F], [F, T, F]]
    # zero cells: (0,0) sz, (0,2) sampling, (1,1) sz, (1,2) sampling
    Y = np.array([[0, 5, 0], [3, 0, 0]])
    sz = np.array([[True, False, False], [False, True, False]])
    prob = np.array([[0.9, 0.0, 0.2], [0.0, 0.8, 0.1]])
    out = metrics.zero_source_metrics(prob, sz, Y)
    assert out["n_zeros"] == 4
    assert out["n_structural"] == 2
    assert out["n_sampling"] == 2
    # scores for [sz, samp, sz, samp] = [0.9, 0.2, 0.8, 0.1]; labels [1,0,1,0]
    # ranks: 0.1->1, 0.2->2, 0.8->3, 0.9->4; positives ranks 3,4 -> U=7-3=4 -> AUC 1
    assert out["auc"] == pytest.approx(1.0)
    # Brier = (0.9-1)^2 + (0.2-0)^2 + (0.8-1)^2 + (0.1-0)^2 all /4 = (0.01+0.04+0.04+0.01)/4
    assert out["brier"] == pytest.approx(0.025)
