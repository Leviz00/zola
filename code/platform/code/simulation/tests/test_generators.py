"""Correctness and edge-case tests for generators.py.

Covers the shared interface contract, the seed tree (exact reproducibility),
boundary behaviour (pi == 1 -> no existence-layer zeros; N -> infinity ->
sampling zeros vanish), truth consistency (structural zeros imply Y == 0;
DA taxa excluded from structural taxa), and mechanism-specific identities
(mean matching for beta-binomial/ZINB; absence indicators for ZIGDM).
"""

import numpy as np
import pytest

import generators
from generators import GENERATORS, generate

BASE = dict(n=12, p=40, depths=20000, seed=11)


# --- interface contract -----------------------------------------------------

@pytest.mark.parametrize("mech", sorted(GENERATORS))
def test_interface_contract(mech):
    Y, truth = generate(mech, {"structural_zero_rate": 0.1}, **BASE)
    n, p = BASE["n"], BASE["p"]
    assert Y.shape == (2 * n, p) and np.issubdtype(Y.dtype, np.integer)
    for key in ("mechanism", "group", "depths", "structural_zeros",
                "presence", "da_taxa", "effect_size", "params"):
        assert key in truth, key
    assert truth["mechanism"] == mech
    assert truth["structural_zeros"].shape == (2 * n, p)
    assert truth["structural_zeros"].dtype == bool
    assert truth["da_taxa"].shape == (p,)
    assert (truth["group"] == np.repeat([0, 1], n)).all()
    # complement relation
    assert (truth["presence"] == ~truth["structural_zeros"]).all()


@pytest.mark.parametrize("mech", sorted(GENERATORS))
def test_structural_zero_implies_zero_count(mech):
    """The core invariant: a truly absent taxon must have zero counts."""
    Y, truth = generate(mech, {"structural_zero_rate": 0.3}, **BASE)
    assert (Y[truth["structural_zeros"]] == 0).all()


@pytest.mark.parametrize("mech", sorted(GENERATORS))
def test_reproducibility_seed_tree(mech):
    Y1, t1 = generate(mech, {}, **BASE)
    Y2, t2 = generate(mech, {}, **BASE)
    Y3, _ = generate(mech, {}, n=BASE["n"], p=BASE["p"],
                     depths=BASE["depths"], seed=BASE["seed"] + 1)
    assert np.array_equal(Y1, Y2)
    assert np.array_equal(t1["structural_zeros"], t2["structural_zeros"])
    assert not np.array_equal(Y1, Y3)


@pytest.mark.parametrize("mech", sorted(GENERATORS))
def test_da_taxa_disjoint_from_designated_structural(mech):
    _, truth = generate(mech, {"structural_zero_rate": 0.3,
                               "da_fraction": 0.2}, **BASE)
    assert not (truth["da_taxa"] & truth["structural_taxa"]).any()


@pytest.mark.parametrize("mech", sorted(GENERATORS))
def test_zero_structural_rate_leaves_no_designated_mask(mech):
    _, truth = generate(mech, {"structural_zero_rate": 0.0}, **BASE)
    assert not truth["structural_taxa"].any()
    assert not truth["designated_structural"].any()


@pytest.mark.parametrize("mech", sorted(GENERATORS))
def test_informative_switch_moves_absence_to_case(mech):
    """Informative zeros: the designated taxa vanish in *all* case samples;
    non-informative: absence is uncorrelated with the group label."""
    params = {"structural_zero_rate": 0.2, "informative_zeros": True}
    _, truth = generate(mech, params, **BASE)
    mask = truth["designated_structural"]
    st = truth["structural_taxa"]
    case_rows = truth["group"] == 1
    assert mask[np.ix_(case_rows, st)].all()   # absent in every case sample
    assert not mask[~case_rows].any()          # present in every control

    params["informative_zeros"] = False
    _, truth = generate(mech, params, **BASE)
    mask = truth["designated_structural"]
    st = truth["structural_taxa"]
    # absence now follows a Bernoulli(0.5) pseudo-batch: at 2n=24 rows,
    # both groups must contain absent and present rows with prob ~ 1-2^-11
    sub = mask[np.ix_(case_rows, st)]
    assert sub.any() and not sub.all()
    sub_c = mask[np.ix_(~case_rows, st)]
    assert sub_c.any() and not sub_c.all()


@pytest.mark.parametrize("mech", sorted(GENERATORS))
def test_deep_sequencing_kills_sampling_zeros(mech):
    """N -> infinity: observed zeros among *present* taxa must vanish."""
    Y, truth = generate(
        mech,
        {"structural_zero_rate": 0.0, "base_prevalence": 1.0,
         "presence_noise_sd": 0.0, "dispersion": 1000.0, "depth_cv": 0.0},
        n=4, p=50, depths=5_000_000, seed=5,
    )
    present = truth["presence"]
    assert present.all()  # prevalence 1 & no noise -> nothing absent
    assert (Y == 0).sum() == 0


@pytest.mark.parametrize("mech", sorted(GENERATORS))
def test_effect_size_applied_to_case_means(mech):
    """Case-group mean relative abundance of DA taxa is inflated ~2-fold."""
    params = {"structural_zero_rate": 0.0, "base_prevalence": 1.0,
              "presence_noise_sd": 0.0, "effect_size": 2.0,
              "da_fraction": 0.3, "dispersion": 1000.0}
    Y, truth = generate(mech, params, n=60, p=80, depths=50000, seed=3)
    da = truth["da_taxa"]
    rel = Y / Y.sum(axis=1, keepdims=True)
    case = truth["group"] == 1
    ratio = rel[case][:, da].mean() / rel[~case][:, da].mean()
    # per-taxon ratios vary (renormalisation), but the pooled mean of the
    # DA block must be clearly elevated
    assert 1.3 < ratio < 3.0


# --- three_layer specifics --------------------------------------------------

def test_three_layer_pi_one_gives_no_structural_zeros():
    """Boundary: prevalence == 1 and no presence noise -> pi == 1 -> all
    taxa present, zero structural zeros."""
    _, truth = generate(
        "three_layer",
        {"structural_zero_rate": 0.0, "base_prevalence": 1.0,
         "presence_noise_sd": 0.0},
        **BASE,
    )
    assert truth["structural_zeros"].sum() == 0
    assert (truth["pi"] == 1.0).all()


def test_three_layer_sampling_zeros_decrease_with_depth():
    """Monotone depth effect: deeper sequencing yields fewer sampling zeros
    among present taxa (same seed tree, only N changes)."""
    params = {"structural_zero_rate": 0.0}
    kw = dict(n=8, p=60, seed=21)
    Y_shallow, tr = generate("three_layer", params, depths=2000, **kw)
    Y_deep, _ = generate("three_layer", params, depths=200000, **kw)
    z_shallow = ((Y_shallow == 0) & tr["presence"]).mean()
    z_deep = ((Y_deep == 0) & tr["presence"]).mean()
    assert z_deep < z_shallow


def test_three_layer_theta_dirichlet_mean():
    """theta_i has mean theta_bar_{g(i)} (low dispersion -> tight)."""
    _, truth = generate(
        "three_layer",
        {"structural_zero_rate": 0.0, "base_prevalence": 1.0,
         "presence_noise_sd": 0.0, "dispersion": 5000.0},
        n=40, p=60, depths=10000, seed=9,
    )
    theta_c = truth["theta"][truth["group"] == 0]
    assert np.abs(theta_c.mean(axis=0)
                  - truth["theta_bar_control"]).max() < 0.02


# --- zinb specifics ---------------------------------------------------------

def test_zinb_inflation_taxon_all_zero_in_case():
    """An informative structural taxon must be all-zero in the case group."""
    Y, truth = generate(
        "zinb", {"structural_zero_rate": 0.2, "informative_zeros": True},
        **BASE,
    )
    st = truth["structural_taxa"]
    case = truth["group"] == 1
    assert (Y[np.ix_(case, st)] == 0).all()
    # and positive somewhere in controls (depth 20k >>)
    assert (Y[np.ix_(~case, st)] > 0).any()


def test_zinb_mean_matches_mu():
    """NB mean: E[Y] ~= N_i * m_j among present taxa (low dispersion)."""
    Y, truth = generate(
        "zinb",
        {"structural_zero_rate": 0.0, "base_prevalence": 1.0,
         "dispersion": 5000.0, "effect_size": 1.5},
        n=60, p=50, depths=20000, seed=13,
    )
    case = truth["group"] == 1
    # all taxa present (prevalence 1): compare sample mean to mu
    mu_c = truth["mu"][case].mean(axis=0)
    ratio = Y[case].mean(axis=0) / np.maximum(mu_c, 1e-12)
    big = mu_c > 50  # only where the mean is not degenerate
    assert np.abs(ratio[big] - 1.0).max() < 0.25


# --- zigdm_like specifics ---------------------------------------------------

def test_zigdm_proportions_normalised():
    """Simplification 1: proportions renormalised to sum to 1 per sample."""
    _, truth = generate("zigdm_like", {}, **BASE)
    assert np.allclose(truth["proportions"].sum(axis=1), 1.0)


def test_zigdm_absent_stick_is_zero_count():
    """Absent stick (Delta=1) <=> zero proportion <=> zero count."""
    Y, truth = generate("zigdm_like", {"structural_zero_rate": 0.3}, **BASE)
    sz = truth["structural_zeros"]
    assert (truth["proportions"][sz] == 0).all()
    assert (Y[sz] == 0).all()


def test_zigdm_cross_taxon_independent_absence():
    """With prevalence fixed and no mask, absence must be iid Bernoulli:
    the absence indicators of two arbitrary taxa are ~ independent."""
    _, truth = generate(
        "zigdm_like",
        {"structural_zero_rate": 0.0, "base_prevalence": 0.5},
        n=400, p=10, depths=2000, seed=17,
    )
    sz = truth["structural_zeros"][:, :2].astype(float)
    # joint P both absent ~= 0.25 under independence; tolerance 0.06
    joint = (sz.prod(axis=1)).mean()
    assert abs(joint - 0.25) < 0.06


# --- beta_binomial specifics ------------------------------------------------

def test_beta_binomial_mean_and_overdispersion():
    """BB mean ~= N*mu; larger phi -> variance closer to binomial."""
    kw = dict(n=80, p=40, depths=5000, seed=23)
    common = {"structural_zero_rate": 0.0, "base_prevalence": 1.0,
              "presence_noise_sd": 0.0}
    Y1, t1 = generate("beta_binomial", {**common, "dispersion": 1e6}, **kw)
    Y2, _ = generate("beta_binomial", {**common, "dispersion": 3.0}, **kw)
    mu = t1["mu"]
    rel_var = (Y1 / Y1.sum(axis=1, keepdims=True)).var(axis=0, ddof=1)
    rel_var2 = (Y2 / Y2.sum(axis=1, keepdims=True)).var(axis=0, ddof=1)
    # near-binomial phi=1e6: sampling variance only; phi=3: much larger
    assert rel_var2.mean() > 5 * rel_var.mean()
    # mean check on controls: Y/N ~= mu
    case = t1["group"] == 1
    est = (Y1 / Y1.sum(axis=1, keepdims=True))[~case].mean(axis=0)
    big = mu[0] > 0.01
    assert np.abs(est[big] / mu[0][big] - 1).max() < 0.35


def test_beta_binomial_presence_layer_structural_zeros():
    Y, truth = generate(
        "beta_binomial",
        {"structural_zero_rate": 0.2, "informative_zeros": True},
        **BASE,
    )
    st = truth["structural_taxa"]
    case = truth["group"] == 1
    assert (Y[np.ix_(case, st)] == 0).all()
    assert truth["structural_zeros"][np.ix_(case, st)].all()


# --- dispatcher -------------------------------------------------------------

def test_unknown_mechanism_raises():
    with pytest.raises(KeyError):
        generate("nope", {}, **BASE)
