"""Factorial design for the simulation study (paper section 6.1, table 4).

Factors (levels)
----------------
mechanism            : {three_layer, zinb, zigdm_like, beta_binomial}   (4)
depth                : {5000, 20000, 100000} reads                      (3)
structural_zero_rate : {0.0, 0.1, 0.3}                                  (3)
informative_zeros    : {False, True}                                    (2)
effect_size          : {1.5, 2.0, 4.0} fold change                      (3)
n                    : {50, 100, 300} samples per group                 (3)
dispersion           : {high, medium, low} overdispersion               (3)
                       -> concentrations {3, 15, 100}; larger = less
                       overdispersed (Dirichlet phi / NB size / BB phi /
                       GD concentration, per mechanism)

Full grid: 4*3*3*2*3*3*3 = 1944 cells.  ``fractional_grid`` builds a
main-effect-balanced screening subset (default 48 runs, divisible by every
number of levels so exact one-way balance is attainable) by a greedy
minimum-imbalance construction; key two-way interactions can be added to
the balance penalty.  This mirrors the plan of section 4: main-effect
screening first, full factorial on the surviving key interactions.

Number of replicates per cell -- Monte Carlo SE of the empirical FDR
--------------------------------------------------------------------
The empirical FDR of a cell is the mean of the per-replicate false
discovery proportions,  FDR_hat = R^{-1} sum_r FDP_r,  with Monte Carlo
SE = sd(FDP)/sqrt(R).  A conservative (distribution-free) bound treats
FDP_r as a [0,1]-variable with variance at most the Bernoulli value at
the nominal level,

    Var(FDP_r) <= FDR (1 - FDR) = 0.05 * 0.95 = 0.0475,

so  SE <= sqrt(0.0475 / R) <= 0.005  iff  R >= 0.0475 / 0.005^2 = 1900.

We therefore adopt **R = 2000** as the default per-cell replicate count
(rounding 1900 up to a round number with 5% slack).  This is conservative
for two reasons:

1. FDP_r is itself an average over m_r rejections, so
   Var(FDP_r) ~= FDR(1-FDR) / E[m_r]  when rejections are weakly
   dependent; cells expecting E[m_r] >= m rejections can afford
   R >= 1900 / m  (floored at ``r_min`` to protect the power SE and
   non-normal cells).  ``required_replicates(mean_rejections=...)``
   implements this refinement.
2. Power estimates pool the m_da truly-DA taxa of each replicate:
   SE(TPR_hat) ~= sqrt(TPR(1-TPR) / (m_da * R)); with m_da = 20 and
   TPR = 0.5 (worst case), R = 2000 gives SE <= 0.0025 < 0.005
   (``mc_se_power``).

Compute budget (fix N4): composite-likelihood fits cost minutes per data
set, so a fractional screen of ~48 cells x R replicates plus a full
factorial over the surviving key interactions stays within the
3x10^2--10^3 core-hour envelope stated in section 6.1 only if per-cell R
is reduced via refinement 1 (e.g. m >= 20 -> R = 100) or the screen uses
a pilot R and R = 2000 is reserved for the final confirmation runs of the
key cells.  The exported configs carry both ``n_replicates`` (default
2000) and ``n_replicates_screen`` (refined value) columns so the runner
can choose.

Seeds: each cell receives ``seed = base_seed * 10007 + cell_id``; the
runner draws per-replicate streams via
``numpy.random.SeedSequence(seed).spawn(R)`` (explicit seed tree, one
child per replicate), so every cell x replicate is exactly reproducible.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

__all__ = [
    "FACTORS",
    "DISPERSION_MAP",
    "full_grid",
    "fractional_grid",
    "required_replicates",
    "mc_se_fdr",
    "mc_se_power",
    "export_config_csv",
    "params_for_cell",
]

FACTORS = {
    "mechanism": ["three_layer", "zinb", "zigdm_like", "beta_binomial"],
    "depth": [5000, 20000, 100000],
    "structural_zero_rate": [0.0, 0.1, 0.3],
    "informative_zeros": [False, True],
    "effect_size": [1.5, 2.0, 4.0],
    "n": [50, 100, 300],
    "dispersion": ["high", "medium", "low"],  # overdispersion level
}

# overdispersion label -> concentration shared by all four mechanisms
DISPERSION_MAP = {"high": 3.0, "medium": 15.0, "low": 100.0}

NOMINAL_FDR = 0.05
TARGET_SE = 0.005
BASE_SEED = 20260701


# ---------------------------------------------------------------------------
# replicate-count derivations
# ---------------------------------------------------------------------------

def required_replicates(fdr=NOMINAL_FDR, se=TARGET_SE, mean_rejections=None,
                        r_min=100, r_max=None):
    """Replicates needed for Monte Carlo SE of the empirical FDR <= ``se``.

    Conservative bound (default): R = ceil(fdr (1-fdr) / se^2) = 1900 for
    fdr = 0.05, se = 0.005 (we export R = 2000, see module docstring).

    If ``mean_rejections`` = E[m_r] is given, uses the refined variance
    Var(FDP) ~= fdr(1-fdr)/E[m_r]  ->  R = ceil(fdr(1-fdr)/(E[m_r] se^2)),
    floored at ``r_min`` and capped at ``r_max`` (default: the conservative
    bound).  The refinement must be justified per cell by a pilot estimate
    of E[m_r]; using it without that justification is not defensible.
    """
    bound = fdr * (1.0 - fdr) / se**2
    if mean_rejections is None:
        return int(np.ceil(bound))
    if mean_rejections < 1:
        raise ValueError("mean_rejections must be >= 1")
    r = int(np.ceil(bound / mean_rejections))
    r = max(r, int(r_min))
    return min(r, int(np.ceil(bound)) if r_max is None else int(r_max))


def mc_se_fdr(R, fdr=NOMINAL_FDR, mean_rejections=None):
    """Monte Carlo SE of the empirical FDR at R replicates (upper bound)."""
    var = fdr * (1.0 - fdr)
    if mean_rejections is not None:
        var /= mean_rejections
    return float(np.sqrt(var / R))


def mc_se_power(R, tpr=0.5, n_da=1):
    """Monte Carlo SE of the power estimate pooling ``n_da`` DA taxa/replicate."""
    return float(np.sqrt(tpr * (1.0 - tpr) / (n_da * R)))


# ---------------------------------------------------------------------------
# grids
# ---------------------------------------------------------------------------

def _grid_frame(cells, design, base_seed):
    df = pd.DataFrame(list(cells), columns=list(FACTORS))
    df.insert(0, "cell_id", np.arange(len(df)))
    df["design"] = design
    df["dispersion_value"] = df["dispersion"].map(DISPERSION_MAP)
    # conservative bound is ceil(0.05*0.95/0.005^2) = 1900; export R = 2000
    # (round number, ~5% slack), see module docstring for the derivation.
    df["n_replicates"] = 2000
    # screening value under the E[m_r] >= 20 refinement (pilot-justified),
    # floored at r_min = 100 to protect the power SE.
    df["n_replicates_screen"] = required_replicates(mean_rejections=20)
    df["seed"] = base_seed * 10007 + df["cell_id"]
    return df


def full_grid(base_seed=BASE_SEED):
    """The full factorial grid (1944 cells)."""
    cells = itertools.product(*FACTORS.values())
    return _grid_frame(cells, "full", base_seed)


def fractional_grid(L=48, base_seed=BASE_SEED, rng_seed=0, two_way=()):
    """Main-effect-balanced fractional factorial subset of ``L`` cells.

    Greedy minimum-imbalance construction: starting from an empty design,
    repeatedly add the full-grid cell that minimises

        penalty = sum_f sum_levels count_f,level^2
                  + 10 * sum_{(f,g) in two_way} sum_{pairs} count_{fg}^2,

    i.e. prefers levels (and, optionally, level *pairs*) used least so far.
    With L divisible by every number of levels this attains exact one-way
    balance (an OA-strength-2-style main-effect screen).  L = 48 is
    divisible by 4, 3 and 2, hence exact balance is attainable for every
    factor; ties are broken with ``rng_seed`` for reproducibility.

    ``two_way``: iterable of factor-name pairs whose two-way marginals are
    also balanced (key interactions to protect during screening).
    """
    if L % max(len(v) for v in FACTORS.values()) != 0:
        raise ValueError("L should be divisible by every factor's #levels "
                         "(4, 3, 2) for exact balance")
    names = list(FACTORS)
    levels = [FACTORS[f] for f in names]
    all_cells = list(itertools.product(*levels))
    rng = np.random.default_rng(rng_seed)

    one_way = [{lvl: 0 for lvl in lv} for lv in levels]
    two_way = [tuple(pair) for pair in two_way]
    pair_index = [(names.index(f), names.index(g)) for f, g in two_way]
    two_way_counts = [dict() for _ in pair_index]

    chosen = []
    chosen_set = set()
    for _ in range(L):
        best, best_pen = None, np.inf
        for cell in all_cells:
            if cell in chosen_set:
                continue
            pen = sum((one_way[k][cell[k]] + 1) ** 2 for k in range(len(names)))
            for t, (a, b) in enumerate(pair_index):
                pen += 10.0 * (two_way_counts[t].get((cell[a], cell[b]), 0) + 1) ** 2
            pen += rng.random() * 1e-9  # reproducible tie-breaking
            if pen < best_pen:
                best, best_pen = cell, pen
        chosen.append(best)
        chosen_set.add(best)
        for k in range(len(names)):
            one_way[k][best[k]] += 1
        for t, (a, b) in enumerate(pair_index):
            key = (best[a], best[b])
            two_way_counts[t][key] = two_way_counts[t].get(key, 0) + 1

    df = _grid_frame(chosen, "fractional", base_seed)
    return df


def params_for_cell(row):
    """Translate a config row into the ``params`` dict of the generators.

    v3: optional columns (effect_mode, base_prevalence, presence_noise_sd,
    da_fraction, depth_cv) are passed through when present; effect_mode
    defaults to "legacy" for backward compatibility with the old configs.
    """
    params = {
        "effect_size": float(row["effect_size"]),
        "structural_zero_rate": float(row["structural_zero_rate"]),
        "informative_zeros": bool(row["informative_zeros"]),
        "dispersion": float(row["dispersion_value"]),
    }
    for key in ("effect_mode", "base_prevalence", "presence_noise_sd",
                "da_fraction", "depth_cv"):
        if key in row.index and not (isinstance(row[key], float)
                                     and np.isnan(row[key])):
            v = row[key]
            if key == "effect_mode":
                params[key] = str(v)
            else:
                params[key] = float(v)
    return params


# ---------------------------------------------------------------------------
# v3.3 supplementary grid (~10 cells; effect_mode=absolute, dual-track truth)
# ---------------------------------------------------------------------------

def supplementary_grid(base_seed=BASE_SEED):
    """Supplementary cells for the v3.2/v3.3 exam-repair round.

    Three groups (see method_fix/v3/V323_MEMO.md for the rationale and the
    measured structural-zero shares):

    * ``phi3000`` (4): identifiable-phi cells, dispersion_value=3000, depths
      {5000, 20000, 100000} straddling 3*phi = 9000, zinb/three_layer x
      informative T/F.
    * ``realfrac`` (4): realised "structural zeros / all zeros" share tuned
      into 36-66% via base_prevalence/depth. Measured on the actual config
      seeds (3 reps each, method_fix/v3/v323_smoke.csv): three_layer 0.387
      (depth 50000, bp 0.77), zinb 0.47-0.52 (depth 500, bp 0.9),
      beta_binomial 0.38-0.40 (depth 100000, bp 0.85), zigdm_like 0.40-0.41
      (depth 100000, bp 0.75).
    * ``bridge`` (2): replicas of fractional cells 6 and 13 parameters with
      effect_mode="absolute" for old-vs-new comparison.

    All rows carry effect_mode="absolute"; the legacy twin is produced at
    runtime under the same seed (shared latent layers).
    """
    rows = [
        # mechanism, depth, sz, informative, effect, n, dispersion_value,
        # base_prevalence, grid_group
        ("three_layer", 5000, 0.1, False, 2.0, 100, 3000.0, None, "phi3000"),
        ("three_layer", 100000, 0.1, True, 2.0, 100, 3000.0, None, "phi3000"),
        ("zinb", 20000, 0.1, False, 2.0, 100, 3000.0, None, "phi3000"),
        ("zinb", 100000, 0.1, True, 2.0, 100, 3000.0, None, "phi3000"),
        ("three_layer", 50000, 0.15, False, 2.0, 100, 15.0, 0.77, "realfrac"),
        ("zinb", 500, 0.1, False, 2.0, 100, 15.0, 0.9, "realfrac"),
        ("beta_binomial", 100000, 0.1, False, 2.0, 100, 15.0, 0.85,
         "realfrac"),
        ("zigdm_like", 100000, 0.1, False, 2.0, 100, 15.0, 0.75, "realfrac"),
        # bridge: replicas of fractional cells 6 and 13 (absolute mode)
        ("beta_binomial", 20000, 0.0, False, 4.0, 50, 3.0, None, "bridge"),
        ("zinb", 100000, 0.3, True, 4.0, 300, 3.0, None, "bridge"),
    ]
    recs = []
    for k, (mech, depth, sz, inf, eff, n, dv, bp, grp) in enumerate(rows):
        recs.append(dict(
            cell_id=1000 + k, mechanism=mech, depth=depth,
            structural_zero_rate=sz, informative_zeros=inf, effect_size=eff,
            n=n, dispersion=("p3000" if dv == 3000.0 else
                             ("high" if dv == 3.0 else "medium")),
            design="supplementary", dispersion_value=dv,
            n_replicates=2000, n_replicates_screen=100,
            seed=base_seed * 10007 + 1000 + k,
            effect_mode="absolute",
            base_prevalence=(np.nan if bp is None else bp),
            grid_group=grp,
            note={"phi3000": "identifiable phi, 3phi=9000",
                  "realfrac": "measured struct-share in [0.36, 0.66]",
                  "bridge": "replica of fractional cell "
                            + ("6" if k == 8 else "13")}[grp],
        ))
    return pd.DataFrame(recs)


def export_supplementary_csv(path, base_seed=BASE_SEED):
    df = supplementary_grid(base_seed=base_seed)
    df.to_csv(path, index=False)
    return df


def export_config_csv(path, mode="full", L=48, two_way=(), base_seed=BASE_SEED):
    """Write the design table to CSV; returns the DataFrame."""
    if mode == "full":
        df = full_grid(base_seed=base_seed)
    elif mode == "fractional":
        df = fractional_grid(L=L, base_seed=base_seed, two_way=two_way)
    else:
        raise ValueError("mode must be 'full' or 'fractional'")
    df.to_csv(path, index=False)
    return df
