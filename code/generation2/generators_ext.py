"""generators_ext.py — v4 testbed extension (NEW module; read-only discipline:
code/simulation_v3 untouched, helpers imported).

Adds the mechanism ``three_layer_real``: the paper's three-layer generator with
a REALISTIC backbone calibrated to the empirical yardstick
(ibdmdb genus-282 computed in-session + AGP documented values; see
realism_scorecard_old.csv / yardstick_ibdmdb.csv):

  * U-shaped per-taxon prevalences (rare majority + mid + high-prevalence
    core) instead of Beta(2,8)/constant;
  * heavy-tailed base composition (log-normal sigma default 2.2 -> >=4 orders
    of magnitude span);
  * depth_cv default 1.0 (real 0.6-1.1);
  * BIDIRECTIONAL intensity DA (half up x eff, half down /eff) at the count
    layer (absolute mode) -> case library uplift ~= 1.0;
  * native PRESENCE-DA channel: selected taxa get case-group presence odds
    multiplied by ``presence_effect_or`` (partial absence, more realistic
    than 1009's total block absence). Truth key ``pres_da_truth``.

Seed tree: _spawn(seed, 6) = (depth, composition, prevalence, selection,
factors+Z, counts). New mechanism => no legacy stream-compatibility burden.
"""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, "/home/claude/ch_smoke/code/simulation_v3")
import generators as G  # read-only helpers

DEFAULTS = dict(
    effect_size=2.0, da_fraction=0.10, da_updown=True,
    presence_da_fraction=0.0, presence_effect_or=0.25,
    dispersion=15.0, depth_cv=0.7, base_sigma=2.0,
    # TWO-SCALE structure (Remark 2 of theory_draft_v1, empirically forced by
    # the yardstick): between-sample composition heterogeneity lives in a
    # per-taxon multiplicative Gamma layer (theta_ij = m0_j * G_ij,
    # G_ij ~ Gamma(a_j, 1/a_j), a_j ~ logN(log(a_med), a_logsd)) -- abundant
    # taxa swing across samples (moment-phi ~ tens) -- while the measurement
    # layer sits at its near-multinomial limit (meas_kappa large), so
    # detection stays depth-and-presence driven as in real detection fits.
    gamma_shape_med=1.5, gamma_shape_logsd=0.5, meas_kappa=1e4,
    # abundance-coupled presence: logit pi_j = pres_a0 + pres_b * s_j + eps,
    # s_j = standardized log m0_j (abundant taxa are prevalent), eps ~ N(0, sd)
    # calibrated 2026-08-18 against the ibdmdb-282 yardstick (pilot memo):
    # zero .85 | dcv .79 | prev_med .05 | prev_max .98 | span 4.0 | uplift ~1.0
    pres_a0=-2.2, pres_b=2.2, pres_eps_sd=0.7,
    factor_rank=3, presence_noise_sd=0.5,
)


def _presence_coupled(m0, prm, rng):
    s = np.log(m0)
    s = (s - s.mean()) / max(s.std(), 1e-9)
    logit_pi = (float(prm["pres_a0"]) + float(prm["pres_b"]) * s
                + rng.normal(0.0, float(prm["pres_eps_sd"]), size=m0.size))
    return np.clip(1.0 / (1.0 + np.exp(-logit_pi)), 0.02, 0.995)


def three_layer_real(params, n, p, depths, seed):
    prm = dict(DEFAULTS)
    prm.update({k: v for k, v in (params or {}).items() if v is not None})
    group = G._groups(n)
    ss = G._spawn(seed, 7)
    rng_depth, rng_comp, rng_prev, rng_sel, rng_fac, rng_kap, rng_y = (
        np.random.default_rng(s) for s in ss)

    N = G._resolve_depths(depths, 2 * n, rng_depth, float(prm["depth_cv"]))
    m0 = G._base_composition(p, rng_comp, float(prm["base_sigma"]))
    prev = _presence_coupled(m0, prm, rng_prev)

    # expected realized prevalence (detection-limited): eligibility only
    med_dep = float(np.median(N))
    prev_exp = prev * (1.0 - np.exp(-0.7 * med_dep * m0))

    # --- DA selection (label-blind truth design) ---
    eligible_int = np.where((prev_exp >= 0.15) & (prev_exp <= 0.97)
                            & (m0 <= 0.05))[0]
    n_int = int(round(float(prm["da_fraction"]) * p))
    int_idx = rng_sel.choice(eligible_int, size=min(n_int, eligible_int.size),
                             replace=False)
    int_mask = np.zeros(p, dtype=bool)
    int_mask[int_idx] = True
    sign = np.ones(p)
    if bool(prm["da_updown"]) and len(int_idx) > 1:
        # library-balanced sign assignment: enumerate up/down patterns and
        # pick the one minimising the net case-library shift
        # (up adds m*(eff-1); down removes m*(1-1/eff)); keeps uplift ~1.0
        eff0 = float(prm["effect_size"])
        m_sel = m0[int_idx]
        k = len(int_idx)
        if k <= 14:
            best, best_net = None, np.inf
            for msk in range(1, 2 ** k - 1):
                bits = (msk >> np.arange(k)) & 1        # 1 = down
                if abs(int(bits.sum()) * 2 - k) > 1:    # keep counts balanced
                    continue
                net = (m_sel[bits == 0] * (eff0 - 1.0)).sum() \
                    - (m_sel[bits == 1] * (1.0 - 1.0 / eff0)).sum()
                if abs(net) < best_net:
                    best, best_net = bits.copy(), abs(net)
            sign[int_idx[best == 1]] = -1.0
        else:
            order = int_idx[np.argsort(m0[int_idx])]
            sign[order[::2]] = -1.0

    pres_mask = np.zeros(p, dtype=bool)
    n_pres = int(round(float(prm["presence_da_fraction"]) * p))
    if n_pres > 0:
        elig = np.where((prev_exp >= 0.30) & (prev_exp <= 0.90)
                        & ~int_mask)[0]
        pres_idx = rng_sel.choice(elig, size=min(n_pres, elig.size),
                                  replace=False)
        pres_mask[pres_idx] = True

    # --- presence layer: logit pi_ij = logit(prev_j) + u'lambda + case shift
    r = int(prm["factor_rank"]); sd = float(prm["presence_noise_sd"])
    a = np.log(prev / (1.0 - prev))
    U = rng_fac.normal(0.0, sd, size=(p, r))
    lam = rng_fac.normal(0.0, 1.0, size=(2 * n, r))
    logits = a[None, :] + lam @ U.T
    shift = np.log(float(prm["presence_effect_or"]))
    logits = logits + np.where((group[:, None] == 1) & pres_mask[None, :],
                               shift, 0.0)
    pi = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    Z = rng_fac.random((2 * n, p)) < pi

    # --- composition layer (no group difference; absolute mode) ---
    # per-taxon multiplicative Gamma heterogeneity (mean-1), then simplex
    a_j = np.exp(rng_kap.normal(np.log(float(prm["gamma_shape_med"])),
                                float(prm["gamma_shape_logsd"]), size=p))
    Gmult = rng_fac.gamma(a_j[None, :], 1.0 / a_j[None, :], size=(2 * n, p))
    theta = m0[None, :] * Gmult
    theta = theta / theta.sum(axis=1, keepdims=True)

    # --- measurement layer: mask-renormalise + bidirectional count-layer DA
    c = theta * Z
    S = c.sum(axis=1, keepdims=True)
    dead = (S[:, 0] <= 0)
    if dead.any():                       # keep the most abundant taxon alive
        j0 = int(np.argmax(m0))
        Z[dead, j0] = True
        c = theta * Z
        S = c.sum(axis=1, keepdims=True)
    theta_t = c / S
    mu = theta_t * N[:, None].astype(float)
    eff = float(prm["effect_size"])
    fold = np.where(sign[None, :] > 0, eff, 1.0 / eff)
    case_da = (group[:, None] == 1) & int_mask[None, :]
    mu = np.where(case_da, mu * fold, mu)
    # measurement layer at its near-multinomial limit
    kappa = np.full_like(mu, float(prm["meas_kappa"]))
    Y, Np = G._draw_counts_absolute(mu, kappa, rng_y)

    truth = dict(
        mechanism="three_layer_real", group=group, depths=N,
        structural_zeros=~Z, presence=Z,
        da_taxa=int_mask, abs_da_truth=int_mask.copy(),
        pres_da_truth=pres_mask, effect_sign=sign,
        effect_mode="absolute", effect_size=eff,
        pi=pi, prevalence=prev, theta_bar=m0, gamma_shape=a_j,
        count_total_target=Np, params=dict(prm),
    )
    assert (Y[truth["structural_zeros"]] == 0).all()
    return Y, truth


def generate_ext(mechanism, params, n, p, depths, seed):
    """Dispatcher covering both the extension and the original mechanisms."""
    if mechanism == "three_layer_real":
        return three_layer_real(params, n, p, depths, seed)
    return G.generate(mechanism, params, n, p, depths, seed)
