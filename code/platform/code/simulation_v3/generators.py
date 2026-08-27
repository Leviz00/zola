"""Data generators for the simulation study of paper section 6.

Four mechanisms, one unified interface
--------------------------------------
Every generator implements

    generate(params, n, p, depths, seed) -> (Y, truth)

with

* ``params``  : dict of simulation knobs (see ``DEFAULT_PARAMS``); unknown keys
  are ignored so that one config row can drive all four mechanisms.
* ``n``       : number of samples **per group** (two-group design, total 2n).
* ``p``       : number of taxa.
* ``depths``  : either a scalar nominal sequencing depth (per-sample depths are
  drawn log-normal around it with coefficient of variation
  ``params['depth_cv']``) or an array-like of length ``2n`` giving the exact
  per-sample depths N_i.
* ``seed``    : int or ``numpy.random.SeedSequence``.  Reproducibility uses an
  explicit seed tree: ``SeedSequence(seed).spawn(k)`` hands one independent
  child stream to each random component (composition, presence, per-sample
  proportions, counts, ...), so every cell x replicate is exactly
  reproducible and adding a component never reshuffles earlier streams.

Returns
-------
``Y``     : int array, shape (2n, p).  Rows 0..n-1 are group 0 (control),
            rows n..2n-1 are group 1 (case).
``truth`` : dict with the keys every mechanism must provide:

    ``mechanism``        str
    ``group``            int array (2n,), 0 = control, 1 = case
    ``depths``           int array (2n,), realised N_i
    ``structural_zeros`` bool array (2n, p), True where taxon is truly absent
                         (existence-layer zero, *before* sampling)
    ``presence``         bool array (2n, p), complement of ``structural_zeros``
    ``da_taxa``          bool array (p,), truly differentially abundant taxa
    ``effect_size``      float, fold change used for the DA taxa
    ``params``           dict, the resolved parameter dict
    plus mechanism-specific extras (pi, theta_bar, phi, ...).

Invariant (asserted in the test-suite): ``Y[structural_zeros] == 0`` -- a truly
absent taxon must yield zero counts, while ``Y == 0 & presence`` are sampling
zeros whose proportion must vanish as N_i -> infinity.

Design factors (paper section 6, table 4) mapped onto ``params``
----------------------------------------------------------------
``structural_zero_rate`` : fraction of taxa entirely missing in one condition
``informative_zeros``    : True  -> the missing condition is the case group
                           (absence aligned with the phenotype);
                           False -> the missing condition is an independent
                           Bernoulli(1/2) pseudo-batch, uncorrelated with the
                           phenotype (non-informative structural zeros).
``effect_size``          : fold change {1.5, 2, 4}
``dispersion``           : concentration parameter (Dirichlet phi / NB size /
                           beta-binomial phi / GD concentration); larger =
                           less overdispersion.  May also be an array of
                           length p (per-taxon dispersion level).
``da_fraction``          : fraction of non-structural taxa that are truly DA.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DEFAULT_PARAMS",
    "GENERATORS",
    "generate",
    "three_layer",
    "zinb",
    "zigdm_like",
    "beta_binomial",
]

# ---------------------------------------------------------------------------
# common parameter defaults and helpers
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "effect_size": 2.0,          # fold change of DA taxa in the case group
    "structural_zero_rate": 0.1,  # fraction of taxa absent in one condition
    "informative_zeros": True,    # absence aligned with phenotype?
    "dispersion": 15.0,          # concentration (larger = less overdispersed)
    "da_fraction": 0.1,          # fraction of non-structural taxa that are DA
    "depth_cv": 0.3,             # CV of per-sample depths around nominal depth
    # existence-layer knobs (three_layer; reused where meaningful)
    "base_prevalence": None,     # None -> Beta(2, 8); scalar in (0, 1] fixes it
    "presence_noise_sd": 0.5,    # sd of the factor-coupling loadings u_j
    "factor_rank": 3,            # rank r of the presence-layer factor term
    "base_sigma": 1.0,           # sd of log base composition (log-normal)
    "seed_mask": 0.5,            # P(non-informative absence condition) per sample
    # v3.2a: "legacy" (DA fold-change + simplex renormalise -> passive squeeze
    # of non-DA taxa) vs "absolute" (DA injected after the measurement layer:
    # non-DA taxa keep identical E[Y|N] across groups; case library size
    # floats up; relative-abundance squeeze remains as a compositional
    # consequence).  See method_fix/v3/V323_MEMO.md.
    "effect_mode": "legacy",
}

_BIG_LOGIT = 8.0  # logit shift that effectively zeroes a presence probability


def _draw_counts_absolute(mu, kappa, rng):
    """v3.2a absolute-mode count layer: per-taxon independent beta-binomial.

    ``mu``    : (2n, p) target count means E[Y_ij] (already including the
                DA fold-change on case & DA taxa).
    ``kappa`` : (2n, p) or broadcastable beta-binomial concentration
                (larger = less overdispersed; binomial limit as kappa -> inf).
    Draws p_ij ~ Beta(pi kappa, (1-pi) kappa), Y_ij ~ Binomial(N'_i, p_ij)
    with pi_ij = mu_ij / N'_i, N'_i = sum_j mu_ij (case depth floats up).
    Returns (Y, Np).
    """
    mu = np.asarray(mu, dtype=float)
    Np = mu.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pi = np.where(Np[:, None] > 0, mu / Np[:, None], 0.0)
    a = np.clip(pi * kappa, 1e-9, None)
    b = np.clip((1.0 - pi) * kappa, 1e-9, None)
    pij = rng.beta(a, b)
    pij = np.where(mu > 0, pij, 0.0)
    n_draw = np.maximum(np.rint(Np), 0).astype(np.int64)[:, None]
    Y = rng.binomial(n_draw, pij).astype(np.int64)
    return Y, Np


def _kappa_dm(phi, pi):
    """DM-edge -> beta-binomial concentration conversion, kappa = phi/(1-pi).

    For a Dirichlet-multinomial marginal the per-taxon beta-binomial
    overdispersion parameterisation of the estimation model uses
    phi~ = phi / (1 - theta~_ij); approximation error vs the exact DM
    marginal is recorded in method_fix/v3 (V323_MEMO).
    """
    return np.asarray(phi, dtype=float) / np.clip(1.0 - pi, 1e-6, None)


def _resolve_params(params):
    resolved = dict(DEFAULT_PARAMS)
    params = dict(params or {})
    unknown = sorted(set(params) - set(DEFAULT_PARAMS))
    resolved.update(params)
    if unknown:
        # extra keys are tolerated (one config row drives all mechanisms)
        # but recorded so that typos stay visible.
        resolved["extra_params"] = unknown
    return resolved


def _spawn(seed, n_children):
    """Explicit seed tree: one child SeedSequence per random component.

    ``seed`` may be an int or an existing ``SeedSequence`` (e.g. one child
    of a per-replicate spawn), so cell -> replicate -> component nesting
    stays a proper tree.
    """
    if isinstance(seed, np.random.SeedSequence):
        return seed.spawn(n_children)
    return np.random.SeedSequence(seed).spawn(n_children)


def _groups(n):
    return np.repeat([0, 1], n)


def _resolve_depths(depths, n_total, rng, cv):
    depths = np.asarray(depths, dtype=float)
    if depths.ndim == 0:
        if cv <= 0:
            d = np.full(n_total, float(depths))
        else:
            sigma2 = np.log1p(cv**2)
            mu = np.log(float(depths)) - sigma2 / 2.0
            d = np.exp(rng.normal(mu, np.sqrt(sigma2), size=n_total))
        return np.maximum(np.rint(d), 1).astype(np.int64)
    if depths.shape != (n_total,):
        raise ValueError(
            f"depths must be scalar or length {n_total}, got shape {depths.shape}"
        )
    return np.maximum(np.rint(depths), 1).astype(np.int64)


def _base_composition(p, rng, sigma=1.0):
    """Log-normal-type base composition, normalised to the simplex."""
    m = np.exp(rng.normal(0.0, sigma, size=p))
    return m / m.sum()


def _apply_effect(m_control, da_mask, effect_size):
    """Fold-change the DA taxa in the case group, renormalise."""
    m_case = m_control.copy()
    m_case[da_mask] *= effect_size
    return m_case / m_case.sum()


def _choose_structural_taxa(p, rate, rng):
    n_sz = int(round(rate * p))
    if n_sz == 0:
        return np.zeros(p, dtype=bool)
    idx = rng.choice(p, size=n_sz, replace=False)
    mask = np.zeros(p, dtype=bool)
    mask[idx] = True
    return mask


def _choose_da_taxa(p, da_fraction, forbidden, rng):
    """DA taxa are drawn from the non-structural taxa."""
    allowed = np.where(~forbidden)[0]
    n_da = int(round(da_fraction * p))
    n_da = min(n_da, allowed.size)
    mask = np.zeros(p, dtype=bool)
    if n_da > 0:
        mask[rng.choice(allowed, size=n_da, replace=False)] = True
    return mask


def _structural_mask(struct_taxa, group, informative, rng, seed_mask=0.5):
    """Deterministic absence mask of shape (2n, p).

    informative  -> the structural taxa are absent in *all* case samples;
    non-informative -> they are absent in the samples of a pseudo-batch
    B_i ~ Bernoulli(seed_mask) drawn independently of the group label.
    """
    n_total = group.size
    mask = np.zeros((n_total, len(struct_taxa)), dtype=bool)
    if not struct_taxa.any():
        return mask
    if informative:
        rows = group == 1
    else:
        rows = rng.random(n_total) < seed_mask
    mask[np.ix_(rows, struct_taxa)] = True
    return mask


def _prevalences(p, params, rng):
    base = params["base_prevalence"]
    if base is None:
        return rng.beta(2.0, 8.0, size=p)
    base = float(base)
    if not (0.0 < base <= 1.0):
        raise ValueError("base_prevalence must lie in (0, 1]")
    return np.full(p, base)


def _common_truth(mechanism, group, depths, sz_mask, da_mask, params):
    return {
        "mechanism": mechanism,
        "group": group,
        "depths": depths,
        "structural_zeros": sz_mask,
        "presence": ~sz_mask,
        "da_taxa": da_mask,
        # v3.2a dual-track truth: in "absolute" mode this is the set of taxa
        # whose E[Y|N] truly differs across groups (clean absolute truth);
        # in "legacy" mode it is the set of *directly manipulated* taxa
        # (non-marked taxa are passively squeezed, see TEST_FAILURE_MEMO).
        "abs_da_truth": da_mask.copy(),
        "effect_mode": str(params.get("effect_mode", "legacy")),
        "effect_size": float(params["effect_size"]),
        "params": {k: v for k, v in params.items()},
    }


# ---------------------------------------------------------------------------
# (a) three_layer: the paper's three-layer model (baseline generator)
# ---------------------------------------------------------------------------

def three_layer(params, n, p, depths, seed):
    """Baseline generator: the three-layer model of section 3.

    Existence layer   : logit pi_ij = a_j + u_j' lambda_i,
                        lambda_i ~ N(0, I_r); deterministic structural mask
                        overrides pi for the structural-zero cells.
    Composition layer : theta_i ~ Dirichlet(phi * theta_bar_{g(i)}).
    Measurement layer : theta_tilde_ij = theta_ij Z_ij / sum_k theta_ik Z_ik,
                        Y_i ~ Multinomial(N_i, theta_tilde_i)
                        (detection efficiency rho is fixed to 1 here, matching
                        identifiability condition A1 of section 4).

    Setting ``base_prevalence=1.0`` and ``presence_noise_sd=0`` gives
    pi == 1 exactly, hence no existence-layer zeros at all (edge-case test).
    """
    prm = _resolve_params(params)
    absolute = prm["effect_mode"] == "absolute"
    group = _groups(n)
    ss = _spawn(seed, 8 if absolute else 7)
    rng_depth, rng_comp, rng_prev, rng_mask, rng_fac, rng_theta, rng_y = (
        np.random.default_rng(s) for s in ss[:7]
    )
    # v3.2a: absolute mode appends one child stream; shared layers (depths,
    # composition, prevalence, masks, factors, theta) stay identical to the
    # legacy run at the same seed.
    rng_abs = np.random.default_rng(ss[7]) if absolute else None

    N = _resolve_depths(depths, 2 * n, rng_depth, prm["depth_cv"])
    m0 = _base_composition(p, rng_comp, prm["base_sigma"])
    prev = _prevalences(p, prm, rng_prev)

    struct_taxa = _choose_structural_taxa(p, prm["structural_zero_rate"], rng_mask)
    sz_mask = _structural_mask(
        struct_taxa, group, prm["informative_zeros"], rng_mask, prm["seed_mask"]
    )
    da_mask = _choose_da_taxa(p, prm["da_fraction"], struct_taxa, rng_mask)
    # v3.2a absolute: no fold-change at the composition layer -> no passive
    # squeeze; the DA effect is injected at the count layer below.
    m_case = (m0.copy() if absolute
              else _apply_effect(m0, da_mask, prm["effect_size"]))
    theta_bar = np.where(group[:, None] == 1, m_case[None, :], m0[None, :])

    # existence layer: logit pi_ij = a_j + u_j' lambda_i
    r = int(prm["factor_rank"])
    sd = float(prm["presence_noise_sd"])
    with np.errstate(divide="ignore"):
        a = np.log(prev / (1.0 - prev))  # a_j = logit(prev_j); prev=1 -> +inf
    if r > 0 and sd > 0:
        U = rng_fac.normal(0.0, sd, size=(p, r))
        lam = rng_fac.normal(0.0, 1.0, size=(2 * n, r))
        logits = a[None, :] + lam @ U.T
    else:
        logits = np.broadcast_to(a[None, :], (2 * n, p)).copy()
    with np.errstate(over="ignore", invalid="ignore"):
        pi = np.where(logits >= 0, 1.0 / (1.0 + np.exp(-logits)),
                      np.exp(logits) / (1.0 + np.exp(logits)))
    Z = (rng_fac.random((2 * n, p)) < pi) & ~sz_mask
    # deterministic structural absence overrides the Bernoulli draw
    pi = np.where(sz_mask, 0.0, pi)

    # composition layer: theta_i ~ Dirichlet(phi * theta_bar_g)
    phi = np.asarray(prm["dispersion"], dtype=float)
    theta = np.stack(
        [rng_theta.dirichlet(phi * theta_bar[i]) for i in range(2 * n)]
    )

    # measurement layer: mask-renormalise, then multinomial
    c = theta * Z
    S = c.sum(axis=1, keepdims=True)
    if (S <= 0).any():
        raise RuntimeError("a sample has no present taxon; relax parameters")
    theta_tilde = c / S
    if absolute:
        # v3.2a: DA fold-change enters *after* the measurement layer.
        # mu_ij = N_i * theta_tilde_ij * (effect if case & DA); non-DA taxa
        # keep E[Y_ij | N_i, theta_i] = N_i theta_tilde_ij in BOTH groups.
        mu = theta_tilde * N[:, None].astype(float)
        case_da = (group == 1)[:, None] & da_mask[None, :]
        mu = np.where(case_da, mu * prm["effect_size"], mu)
        kappa = _kappa_dm(phi, theta_tilde)
        Y, Np = _draw_counts_absolute(mu, kappa, rng_abs)
    else:
        Y = np.stack(
            [rng_y.multinomial(int(N[i]), theta_tilde[i])
             for i in range(2 * n)]
        ).astype(np.int64)

    # every existence-layer zero (Z == 0) is a structural zero, whether it
    # came from the designated group-level mask or from low prevalence pi.
    truth = _common_truth("three_layer", group, N, ~Z, da_mask, prm)
    truth.update(
        pi=pi, theta_bar_control=m0, theta_bar_case=m_case, theta=theta,
        prevalence=prev, structural_taxa=struct_taxa,
        designated_structural=sz_mask,
    )
    if absolute:
        truth.update(mu_mean=mu, kappa=kappa, count_total_target=Np,
                     theta_tilde=theta_tilde)
    return Y, truth


# ---------------------------------------------------------------------------
# (b) zinb: per-taxon zero-inflated negative binomial (adversary 1)
# ---------------------------------------------------------------------------

def zinb(params, n, p, depths, seed):
    """Adversary 1: per-taxon ZINB, no compositional coupling.

    Y_ij = 0 with probability omega_ij (latent indicator G_ij = 0 ->
    *structural* zero), else Y_ij ~ NB(mean = N_i * mtilde_j^{g(i)},
    size = dispersion).  Means are normalised so that E[sum_j Y_ij] = N_i
    among present taxa, keeping depths comparable with the other mechanisms.
    """
    prm = _resolve_params(params)
    absolute = prm["effect_mode"] == "absolute"
    group = _groups(n)
    ss = _spawn(seed, 7 if absolute else 6)
    rng_depth, rng_comp, rng_prev, rng_mask, rng_lat, rng_y = (
        np.random.default_rng(s) for s in ss[:6]
    )
    rng_abs = np.random.default_rng(ss[6]) if absolute else None

    N = _resolve_depths(depths, 2 * n, rng_depth, prm["depth_cv"])
    m0 = _base_composition(p, rng_comp, prm["base_sigma"])
    prev = _prevalences(p, prm, rng_prev)

    struct_taxa = _choose_structural_taxa(p, prm["structural_zero_rate"], rng_mask)
    sz_mask = _structural_mask(
        struct_taxa, group, prm["informative_zeros"], rng_mask, prm["seed_mask"]
    )
    da_mask = _choose_da_taxa(p, prm["da_fraction"], struct_taxa, rng_mask)
    # v3.2a absolute: no composition-layer fold-change (no squeeze).
    m_case = (m0.copy() if absolute
              else _apply_effect(m0, da_mask, prm["effect_size"]))
    m = np.where(group[:, None] == 1, m_case[None, :], m0[None, :])

    # latent presence indicator: omega = 1 on the structural mask,
    # omega = 1 - prev elsewhere (chance-level absence outside the mask).
    omega = np.where(sz_mask, 1.0, (1.0 - prev)[None, :])
    present = rng_lat.random((2 * n, p)) >= omega  # G_ij = 1 -> present
    present &= ~sz_mask

    size = np.asarray(prm["dispersion"], dtype=float)
    if absolute:
        # v3.2a: ZINB's native count family is NB, so the absolute layer
        # stays NB (no beta-binomial conversion -> zero kappa approximation
        # error).  mu_ij = N_i m0_j on present taxa, fold-change on case&DA.
        mu_abs = N[:, None].astype(float) * m0[None, :]
        case_da = (group == 1)[:, None] & da_mask[None, :]
        mu_abs = np.where(case_da, mu_abs * prm["effect_size"], mu_abs)
        mu_abs = np.where(present, mu_abs, 0.0)
        prob = size / (size + mu_abs)
        Y = np.where(
            present,
            rng_abs.negative_binomial(
                np.broadcast_to(size, mu_abs.shape), prob),
            0,
        ).astype(np.int64)
    else:
        mu = N[:, None] * m
        prob = size / (size + mu)  # NB(size, prob): mean = size*(1-prob)/prob
        Y = np.where(
            present,
            rng_y.negative_binomial(np.broadcast_to(size, mu.shape), prob),
            0,
        ).astype(np.int64)

    truth = _common_truth("zinb", group, N, ~present, da_mask, prm)
    # NOTE: for ZINB the structural zeros are the *realised* latent absence
    # indicators (mask plus chance-level inflation zeros), i.e. G_ij == 0.
    truth.update(
        omega=omega, mu=(mu_abs if absolute else mu), size=size,
        structural_taxa=struct_taxa,
        designated_structural=sz_mask,
    )
    if absolute:
        truth.update(mu_mean=mu_abs,
                     count_total_target=np.where(present, mu_abs, 0.0)
                     .sum(axis=1))
    return Y, truth


# ---------------------------------------------------------------------------
# (c) zigdm_like: simplified stick-breaking zero-inflated generalised
#     Dirichlet-multinomial (adversary 2)
# ---------------------------------------------------------------------------

def zigdm_like(params, n, p, depths, seed):
    """Adversary 2: simplified ZIGDM (Tang & Chen 2019, Biostatistics 20:698).

    Faithful parts of the construction (their section 2.2):
      * stick-breaking: V_ij ~ Beta(a_j, b_j) iid across j,
        P_ij = V_ij * prod_{k<j} (1 - V_ik);
      * zero-inflated sticks: absence indicators Delta_ij ~ Bernoulli(omega_j)
        **independent across taxa** (their Delta_j = I(Z_j = 0)); an absent
        stick forces P_ij = 0;
      * counts: Y_i | P_i ~ Multinomial(N_i, P_i) (the GDM marginal mixture).

    Documented simplifications relative to the published ZIGDM:
      1. Taxa are stick-broken in the fixed order 1..p; the original keeps a
         (p+1)-th residual category.  Here the residual mass is redistributed
         by **renormalising P_i over the present taxa** so rows sum to 1.
      2. Beta parameters are derived from a target mean composition m and a
         single concentration c (a_j = v_j c, b_j = (1 - v_j) c with
         v_j = m_j / (1 - sum_{k<j} m_k)); the full model allows free
         (a_j, b_j) and links them to covariates through a regression.
      3. Group differences enter through m (mean) only; the published model
         also regresses dispersion on covariates.
    """
    prm = _resolve_params(params)
    absolute = prm["effect_mode"] == "absolute"
    group = _groups(n)
    ss = _spawn(seed, 7 if absolute else 6)
    rng_depth, rng_comp, rng_prev, rng_mask, rng_stick, rng_y = (
        np.random.default_rng(s) for s in ss[:6]
    )
    rng_abs = np.random.default_rng(ss[6]) if absolute else None

    N = _resolve_depths(depths, 2 * n, rng_depth, prm["depth_cv"])
    m0 = _base_composition(p, rng_comp, prm["base_sigma"])
    prev = _prevalences(p, prm, rng_prev)

    struct_taxa = _choose_structural_taxa(p, prm["structural_zero_rate"], rng_mask)
    sz_mask = _structural_mask(
        struct_taxa, group, prm["informative_zeros"], rng_mask, prm["seed_mask"]
    )
    da_mask = _choose_da_taxa(p, prm["da_fraction"], struct_taxa, rng_mask)
    # v3.2a absolute: no fold-change in the stick means (no squeeze).
    m_case = (m0.copy() if absolute
              else _apply_effect(m0, da_mask, prm["effect_size"]))
    m = np.where(group[:, None] == 1, m_case[None, :], m0[None, :])

    # stick-breaking means from the target composition (per group)
    c = np.asarray(prm["dispersion"], dtype=float)  # concentration
    Y = np.zeros((2 * n, p), dtype=np.int64)
    P_all = np.zeros((2 * n, p))
    for i in range(2 * n):
        mi = m[i]
        remaining = 1.0 - np.concatenate([[0.0], np.cumsum(mi)[:-1]])
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.clip(mi / np.maximum(remaining, 1e-12), 1e-6, 1.0 - 1e-6)
        a_j = v * c
        b_j = (1.0 - v) * c
        V = rng_stick.beta(a_j, b_j)
        # cross-taxon independent absence indicators (ZIGDM Delta_j)
        omega = np.where(sz_mask[i], 1.0, 1.0 - prev)
        present = rng_stick.random(p) >= omega
        present &= ~sz_mask[i]
        sticks = V * present
        P = sticks * np.concatenate([[1.0], np.cumprod(1.0 - sticks)[:-1]])
        P[~present] = 0.0
        tot = P.sum()
        if tot <= 0:  # all sticks absent (measure-zero in practice)
            present[np.argmax(mi)] = True
            P = present.astype(float)
            tot = P.sum()
        P /= tot  # simplification 1: renormalise over present taxa
        P_all[i] = P
        Y[i] = rng_y.multinomial(int(N[i]), P)

    if absolute:
        # v3.2a: DA fold-change after the stick/multinomial layer; GDM-edge
        # conversion kappa = c/(1-P_ij) (same family as the DM conversion).
        mu = P_all * N[:, None].astype(float)
        case_da = (group == 1)[:, None] & da_mask[None, :]
        mu = np.where(case_da, mu * prm["effect_size"], mu)
        kappa = _kappa_dm(c, P_all)
        Y, Np = _draw_counts_absolute(mu, kappa, rng_abs)

    sz_realised = P_all == 0.0  # absent stick <=> zero proportion (eq. 2.1)
    truth = _common_truth("zigdm_like", group, N, sz_realised, da_mask, prm)
    truth.update(
        proportions=P_all, structural_taxa=struct_taxa,
        designated_structural=sz_mask,
    )
    if absolute:
        truth.update(mu_mean=mu, kappa=kappa, count_total_target=Np)
    return Y, truth


# ---------------------------------------------------------------------------
# (d) beta_binomial: corncob-type per-taxon beta-binomial with a logistic
#     presence layer (adversary 3)
# ---------------------------------------------------------------------------

def beta_binomial(params, n, p, depths, seed):
    """Adversary 3: corncob-type beta-binomial + logistic presence layer.

    Presence layer : Z_ij ~ Bernoulli(pi_ij) with pi from the same
                     logistic construction as ``three_layer`` (informative /
                     non-informative structural zeros alike).
    Count layer    : Y_ij | Z_ij = 1 ~ BetaBinomial(N_i, mu_ij, phi),
                     mu_ij = m_j^{g(i)}; sampled as p_ij ~ Beta(mu phi,
                     (1-mu) phi), Y_ij ~ Binomial(N_i, p_ij).  Taxa are
                     mutually independent given the means -- no compositional
                     coupling, as in corncob.
    """
    prm = _resolve_params(params)
    absolute = prm["effect_mode"] == "absolute"
    group = _groups(n)
    ss = _spawn(seed, 8 if absolute else 7)
    rng_depth, rng_comp, rng_prev, rng_mask, rng_fac, rng_beta, rng_y = (
        np.random.default_rng(s) for s in ss[:7]
    )
    rng_abs = np.random.default_rng(ss[7]) if absolute else None

    N = _resolve_depths(depths, 2 * n, rng_depth, prm["depth_cv"])
    m0 = _base_composition(p, rng_comp, prm["base_sigma"])
    prev = _prevalences(p, prm, rng_prev)

    struct_taxa = _choose_structural_taxa(p, prm["structural_zero_rate"], rng_mask)
    sz_mask = _structural_mask(
        struct_taxa, group, prm["informative_zeros"], rng_mask, prm["seed_mask"]
    )
    da_mask = _choose_da_taxa(p, prm["da_fraction"], struct_taxa, rng_mask)
    # v3.2a absolute: no fold-change in the mean layer (no squeeze); the
    # beta-binomial count layer is native, so kappa = phi is exact.
    m_case = (m0.copy() if absolute
              else _apply_effect(m0, da_mask, prm["effect_size"]))
    mu = np.where(group[:, None] == 1, m_case[None, :], m0[None, :])

    r = int(prm["factor_rank"])
    sd = float(prm["presence_noise_sd"])
    with np.errstate(divide="ignore"):
        a = np.log(prev / (1.0 - prev))
    if r > 0 and sd > 0:
        U = rng_fac.normal(0.0, sd, size=(p, r))
        lam = rng_fac.normal(0.0, 1.0, size=(2 * n, r))
        logits = a[None, :] + lam @ U.T
    else:
        logits = np.broadcast_to(a[None, :], (2 * n, p)).copy()
    with np.errstate(over="ignore", invalid="ignore"):
        pi = np.where(logits >= 0, 1.0 / (1.0 + np.exp(-logits)),
                      np.exp(logits) / (1.0 + np.exp(logits)))
    Z = (rng_fac.random((2 * n, p)) < pi) & ~sz_mask
    pi = np.where(sz_mask, 0.0, pi)

    phi = np.asarray(prm["dispersion"], dtype=float)
    if absolute:
        # v3.2a: mu_ij = N_i m0_j on present taxa, fold-change on case&DA;
        # native beta-binomial layer with exact kappa = phi.
        mu_abs = N[:, None].astype(float) * m0[None, :]
        case_da = (group == 1)[:, None] & da_mask[None, :]
        mu_abs = np.where(case_da, mu_abs * prm["effect_size"], mu_abs)
        mu_abs = np.where(Z, mu_abs, 0.0)
        Y, Np = _draw_counts_absolute(
            mu_abs, np.broadcast_to(phi, mu_abs.shape), rng_abs)
    else:
        mu_c = np.clip(mu, 1e-9, 1.0 - 1e-9)
        pij = rng_beta.beta(mu_c * phi, (1.0 - mu_c) * phi)
        Y = np.where(
            Z, rng_y.binomial(N[:, None], pij), 0
        ).astype(np.int64)

    truth = _common_truth("beta_binomial", group, N, ~Z, da_mask, prm)
    truth.update(
        pi=pi, mu=(mu_abs / np.maximum(N[:, None], 1) if absolute else mu),
        phi=phi, structural_taxa=struct_taxa,
        prevalence=prev, designated_structural=sz_mask,
    )
    if absolute:
        truth.update(mu_mean=mu_abs, count_total_target=Np)
    return Y, truth


# ---------------------------------------------------------------------------
# registry / dispatcher
# ---------------------------------------------------------------------------

GENERATORS = {
    "three_layer": three_layer,
    "zinb": zinb,
    "zigdm_like": zigdm_like,
    "beta_binomial": beta_binomial,
}


def generate(mechanism, params, n, p, depths, seed):
    """Dispatch to ``GENERATORS[mechanism]``; validates the shared contract."""
    if mechanism not in GENERATORS:
        raise KeyError(
            f"unknown mechanism {mechanism!r}; choices: {sorted(GENERATORS)}"
        )
    Y, truth = GENERATORS[mechanism](params, n, p, depths, seed)
    if Y.shape != truth["structural_zeros"].shape:
        raise AssertionError("Y and structural_zeros shapes disagree")
    if (Y[truth["structural_zeros"]] != 0).any():
        raise AssertionError("invariant violated: structural zero with Y > 0")
    return Y, truth
