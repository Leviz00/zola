"""Tests for design.py: replicate derivation, grid sizes, balance, seeds."""

import numpy as np
import pytest

import design


def test_required_replicates_conservative():
    # ceil(0.05 * 0.95 / 0.005^2) = ceil(1900) = 1900; design exports 2000
    assert design.required_replicates() == 1900
    assert design.mc_se_fdr(2000) <= 0.005
    assert design.mc_se_fdr(1899) > 0.005


def test_required_replicates_refinement():
    # E[m] = 20 -> ceil(1900/20) = 95, floored at r_min = 100
    assert design.required_replicates(mean_rejections=20) == 100
    # E[m] = 100 -> ceil(19) -> floored to 100
    assert design.required_replicates(mean_rejections=100) == 100
    # never exceeds the conservative bound
    assert design.required_replicates(mean_rejections=3) <= 1900


def test_mc_se_power_pools_da_taxa():
    # sqrt(0.25 / (20 * 2000)) = 0.0025
    assert design.mc_se_power(2000, tpr=0.5, n_da=20) == pytest.approx(0.0025)


def test_full_grid_size_and_columns():
    df = design.full_grid()
    assert len(df) == 4 * 3 * 3 * 2 * 3 * 3 * 3
    for col in ("cell_id", "mechanism", "depth", "structural_zero_rate",
                "informative_zeros", "effect_size", "n", "dispersion",
                "dispersion_value", "n_replicates", "n_replicates_screen",
                "seed", "design"):
        assert col in df.columns
    assert (df["n_replicates"] == 2000).all()
    # every combination unique
    assert not df.duplicated(subset=list(design.FACTORS)).any()


def test_fractional_grid_exact_one_way_balance():
    df = design.fractional_grid(L=48)
    assert len(df) == 48
    for f, levels in design.FACTORS.items():
        counts = df[f].value_counts()
        assert counts.to_dict() == {lvl: 48 // len(levels) for lvl in levels}
    assert not df.duplicated(subset=list(design.FACTORS)).any()


def test_fractional_grid_two_way_balance():
    df = design.fractional_grid(
        L=48, two_way=[("mechanism", "structural_zero_rate"),
                       ("structural_zero_rate", "informative_zeros")])
    c1 = df.groupby(["mechanism", "structural_zero_rate"]).size()
    assert (c1 == 48 / (4 * 3)).all()
    c2 = df.groupby(["structural_zero_rate", "informative_zeros"]).size()
    assert (c2 == 48 / (3 * 2)).all()


def test_fractional_grid_reproducible():
    a = design.fractional_grid(L=48)
    b = design.fractional_grid(L=48)
    assert a.equals(b)


def test_params_for_cell_translation():
    df = design.full_grid()
    row = df.iloc[0]
    params = design.params_for_cell(row)
    assert params["effect_size"] == float(row["effect_size"])
    assert params["dispersion"] == design.DISPERSION_MAP[row["dispersion"]]
    assert params["informative_zeros"] == bool(row["informative_zeros"])


def test_cell_seeds_unique():
    df = design.full_grid()
    assert df["seed"].is_unique
