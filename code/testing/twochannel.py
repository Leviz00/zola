"""twochannel.py — Hurdle-ZOLA two-channel test (dev/smoke implementation).

Channels (theory_draft_v1.tex Secs. 3-4):
  detection : score U1_j = sum_i (x_i - xbar) (D_ij - qhat_j(N_i)),
              qhat_j(N) = pi_j (1 - g(N; theta_j, phi)) fitted label-blind,
              shared phi profiled on a grid (per-taxon (pi_j, theta_j) MLE).
  intensity : zero-truncated NB null fit on positives (intercept + log-offset),
              score U2_j = sum_{i in pos} (x_i - xbar_pos) s_ij,
              s_ij = (y_i - mu_i/(1-p0_i)) / (1 + mu_i/r)   [see derivation].
Combination: C_j = z1_j^2 + z2_j^2 with z = permutation-studentized scores;
p-values by joint label permutation (K permutations, add-one rule) => exact
validity per Theorem A regardless of fit quality (nuisances are label-blind).

Dev-level guardrails (documented, mirror abs_glm.py conventions):
  - detection channel inactive (z1=0) if taxon has <3 detections or <3 zeros;
  - intensity channel inactive (z2=0) if <3 positives in either group;
  - fit failures fall back to constant qhat (label-blind => validity intact).
"""
from __future__ import annotations

import sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, expit, logit

sys.path.insert(0, "/home/claude/ch_smoke/code/estimation_v3")
from model import log_g  # closed-form Beta-mixed non-detection probability

PHI_GRID = np.array([3.0, 15.0, 100.0, 1000.0, 3000.0, 1e4, 1e5])


# ---------------------------------------------------------------------------
# detection-curve fitting (label-blind)
# ---------------------------------------------------------------------------

def _bern_negll(par, D, N, phi):
    lp, lt = par
    pi = expit(lp)
    theta = min(np.exp(lt), 0.949999)
    with np.errstate(all="ignore"):
        g = np.exp(log_g(N, theta, phi))
        q = np.clip(pi * (1.0 - g), 1e-12, 1.0 - 1e-12)
        val = -(D * np.log(q) + (1 - D) * np.log(1.0 - q)).sum()
    return val if np.isfinite(val) else 1e12


# Amendment A1 (SPEC-CH): L-BFGS-B iteration/function-eval caps added after a
# line-search hang in the coarse phi scan (one (taxon, phi, start) combination
# looped >90 s). Caps are label-blind numerical guardrails; validity is
# unaffected (Theorem A), typical fits converge in <50 iterations.
_OPT_OPTS = dict(maxiter=100, maxfun=250, ftol=1e-9)


def _fit_taxon(D, N, phi, starts):
    best = None
    for x0 in starts:
        try:
            r = minimize(_bern_negll, x0, args=(D, N, phi),
                         method="L-BFGS-B",
                         bounds=[(-13.0, 13.0), (np.log(1e-9), np.log(0.95))],
                         options=_OPT_OPTS)
            if best is None or r.fun < best.fun:
                best = r
        except Exception:
            continue
    return best


def fit_detection_curves(D, N, phi_grid=PHI_GRID, refine_starts=3):
    """Label-blind per-taxon (pi_j, theta_j) with shared profiled phi.

    D: (n, p) binary detection matrix; N: (n,) depths.
    Returns dict(qhat=(n,p), phi=float, pi=(p,), theta=(p,), ok=(p,) bool).
    """
    n, p = D.shape
    prev = D.mean(axis=0)
    share = np.maximum((D * 1.0).mean(axis=0) * 0 + 1e-7, 1e-7)  # placeholder
    # crude abundance start: detected-count share proxy
    active = (D.sum(axis=0) >= 3) & ((1 - D).sum(axis=0) >= 3)

    def starts_for(j, extra=False):
        p0 = np.clip(prev[j], 0.02, 0.98)
        s0 = [(logit(p0), np.log(1e-4)),
              (logit(np.clip(p0 + 0.2, 0.05, 0.995)), np.log(1e-3))]
        if extra:
            s0 += [(12.0, np.log(1e-5)), (logit(p0), np.log(1e-2))]
        return s0

    # coarse profile over phi (single cheap start per taxon)
    tot = np.full(len(phi_grid), np.inf)
    for k, phi in enumerate(phi_grid):
        s = 0.0
        for j in range(p):
            if not active[j]:
                continue
            r = _fit_taxon(D[:, j], N, phi, starts_for(j)[:2])
            s += (r.fun if r is not None else 0.0)
        tot[k] = s
    phi_hat = float(phi_grid[int(np.argmin(tot))])

    # refined per-taxon fit at phi_hat
    pi = np.full(p, np.nan)
    theta = np.full(p, np.nan)
    qhat = np.tile(prev, (n, 1))          # fallback: constant curve
    ok = np.zeros(p, dtype=bool)
    for j in range(p):
        if not active[j]:
            continue
        r = _fit_taxon(D[:, j], N, phi_hat, starts_for(j, extra=True))
        if r is None or not np.isfinite(r.fun):
            continue
        lp, lt = r.x
        pi[j] = expit(lp)
        theta[j] = np.exp(lt)
        g = np.exp(log_g(N, theta[j], phi_hat))
        qhat[:, j] = np.clip(pi[j] * (1.0 - g), 1e-10, 1 - 1e-10)
        ok[j] = True
    return dict(qhat=qhat, phi=phi_hat, pi=pi, theta=theta, ok=ok,
                active=active)


# ---------------------------------------------------------------------------
# zero-truncated NB null fit (label-blind) and score residuals
# ---------------------------------------------------------------------------

def _tnb_negll(par, y, off):
    b0, logr = par
    r = np.exp(logr)
    mu = np.exp(b0 + off)
    ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
          + r * (np.log(r) - np.log(r + mu)) + y * (np.log(mu) - np.log(r + mu)))
    logp0 = r * (np.log(r) - np.log(r + mu))
    ll -= np.log1p(-np.exp(np.clip(logp0, -700, -1e-12)))
    return -ll.sum()


def tnb_null_residuals(y, off):
    """Fit truncated-NB null (intercept + offset) on positives; return score
    residuals s_i = (y_i - mu_i/(1-p0_i)) / (1 + mu_i/r)."""
    m = np.log(np.maximum(y.mean() / np.exp(off).mean(), 1e-8))
    best = None
    for x0 in [(m, 0.0), (m, 2.0), (m, -1.5)]:
        try:
            r_ = minimize(_tnb_negll, x0, args=(y, off), method="L-BFGS-B",
                          bounds=[(-40, 40), (-6, 12)], options=_OPT_OPTS)
            if best is None or r_.fun < best.fun:
                best = r_
        except Exception:
            continue
    if best is None or not np.isfinite(best.fun):
        return None
    b0, logr = best.x
    r = np.exp(logr)
    mu = np.exp(b0 + off)
    p0 = np.exp(r * (np.log(r) - np.log(r + mu)))
    s = (y - mu / np.clip(1.0 - p0, 1e-10, None)) / (1.0 + mu / r)
    return s


# ---------------------------------------------------------------------------
# two-channel permutation test
# ---------------------------------------------------------------------------

def median_ratio_offset(Y, n_ref=50):
    """Label-blind poscounts median-ratio size factors (anchor: majority of
    the reference set non-DA). Reference = top-prevalence taxa; per sample,
    s_i = median over the reference taxa DETECTED in that sample of
    Y_ij / geomean_j, with geomean_j computed over the samples detecting j
    (poscounts convention: ratios never involve zeros, so s_i does not
    degrade into a detection counter under high zero fractions -- the
    failure mode measured in SPEC-CH Amendment A3 self-review round 2)."""
    Y = np.asarray(Y, dtype=float)
    prev = (Y > 0).mean(axis=0)
    ref = np.argsort(-prev)[:n_ref]
    Yr = Y[:, ref]
    with np.errstate(divide="ignore"):
        L = np.where(Yr > 0, np.log(Yr), np.nan)
    g = np.nanmean(L, axis=0, keepdims=True)
    R = L - g
    s = np.exp(np.nanmedian(R, axis=1))
    s = np.where(np.isfinite(s), s, 1.0)
    return s * Y.sum(axis=1).mean()


def _perm_within_strata(x, strata, rng):
    out = x.copy()
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        out[idx] = rng.permutation(out[idx])
    return out


def two_channel_test(Y, N, group, nu=None, K=999, seed=20260818,
                     phi_grid=PHI_GRID, strata=None, clusters=None):
    """Returns per-taxon results dict.

    nu     : intensity offset (default = N; use median_ratio_offset(Y) for
             the anchored default of Amendment A3).
    strata : optional array of stratum labels; permutations are performed
             within strata (Theorem A conditioning — e.g. realized-library
             deciles when the library~group diagnostic fires, or
             subject/batch clusters in real cohorts)."""
    Y = np.asarray(Y)
    n, p = Y.shape
    N = np.asarray(N, dtype=float)
    nu = N if nu is None else np.asarray(nu, dtype=float)
    x = np.where(np.asarray(group) == 1, 0.5, -0.5)

    D = (Y > 0).astype(float)
    det = fit_detection_curves(D, N, phi_grid=phi_grid)
    R1 = D - det["qhat"]                       # (n, p) detection residuals

    R2 = np.zeros((n, p))
    int_ok = np.zeros(p, dtype=bool)
    for j in range(p):
        pos = Y[:, j] > 0
        if (pos & (x > 0)).sum() < 3 or (pos & (x < 0)).sum() < 3:
            continue
        s = tnb_null_residuals(Y[pos, j].astype(float), np.log(nu[pos]))
        if s is None:
            continue
        R2[pos, j] = s
        int_ok[j] = True
    det_ok = det["active"]
    R1[:, ~det_ok] = 0.0

    # joint permutation (seed may be int or entropy list per SPEC-CH §1)
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    X = np.empty((K + 1, n))
    X[0] = x
    if clusters is not None:
        # cluster-level permutation (e.g. subject-level labels in
        # longitudinal cohorts): each cluster carries one label; labels are
        # permuted across clusters (optionally within strata of clusters)
        clusters = np.asarray(clusters)
        uc, first = np.unique(clusters, return_index=True)
        cx = x[first]                       # one label per cluster
        cs = (np.asarray(strata)[first] if strata is not None
              else np.zeros(len(uc)))
        pos = {c: np.where(clusters == c)[0] for c in uc}
        for k in range(1, K + 1):
            pcx = _perm_within_strata(cx, cs, rng)
            xk = np.empty(n)
            for c, v in zip(uc, pcx):
                xk[pos[c]] = v
            X[k] = xk
    elif strata is None:
        for k in range(1, K + 1):
            X[k] = rng.permutation(x)
    else:
        strata = np.asarray(strata)
        for k in range(1, K + 1):
            X[k] = _perm_within_strata(x, strata, rng)
    Xc = X - X.mean(axis=1, keepdims=True)
    U1 = Xc @ R1                                # (K+1, p)
    U2 = Xc @ R2

    def studentize(U):
        mu = U.mean(axis=0, keepdims=True)
        sd = U.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-12, 1.0, sd)
        return (U - mu) / sd

    Z1, Z2 = studentize(U1), studentize(U2)
    Z1[:, ~det_ok] = 0.0
    Z2[:, ~int_ok] = 0.0
    C = Z1 ** 2 + Z2 ** 2

    def perm_p(M):
        obs = M[0]
        return (1.0 + (M[1:] >= obs[None, :]).sum(axis=0)) / (K + 1.0)

    def within_set_p(M):
        """Per-element right-tail rank p within the (K+1)-set, per column.

        Symmetric in the exchangeable tuple => the vector of within-set
        p-values is itself exchangeable; any symmetric combination of them
        remains a valid permutation statistic."""
        order = np.argsort(-M, axis=0, kind="stable")
        ranks = np.empty_like(order)
        rows = np.arange(M.shape[0])
        for c in range(M.shape[1]):
            ranks[order[:, c], c] = rows
        return (ranks + 1.0) / (M.shape[0])

    # adaptive (ACAT/Cauchy) combination, permutation-calibrated:
    # per-permutation channel p-values by within-set rank -> Cauchy average
    # over ACTIVE channels -> compare observed vs permuted ACAT stats.
    P1 = within_set_p(Z1 ** 2)
    P2 = within_set_p(Z2 ** 2)
    eps = 1.0 / (2.0 * (K + 1.0))
    T1 = np.tan(np.pi * (0.5 - np.clip(P1, eps, 1 - eps)))
    T2 = np.tan(np.pi * (0.5 - np.clip(P2, eps, 1 - eps)))
    act = det_ok.astype(float) + int_ok.astype(float)
    act_safe = np.where(act > 0, act, 1.0)
    ACAT = (T1 * det_ok[None, :] + T2 * int_ok[None, :]) / act_safe[None, :]
    p_comb = perm_p(ACAT)
    p_comb_chisq = perm_p(C)  # secondary: 2-df sum combination
    p_det = perm_p(Z1 ** 2)
    p_int = perm_p(Z2 ** 2)
    attribution = np.where(Z1[0] ** 2 >= Z2[0] ** 2, "det", "int")
    tested = det_ok | int_ok
    p_comb[~tested] = 1.0
    p_comb_chisq[~tested] = 1.0
    p_det[~det_ok] = 1.0
    p_int[~int_ok] = 1.0
    return dict(p_comb=p_comb, p_comb_chisq=p_comb_chisq,
                p_det=p_det, p_int=p_int,
                z_det=Z1[0], z_int=Z2[0], attribution=attribution,
                phi_hat=det["phi"], pi=det["pi"], theta=det["theta"],
                det_ok=det_ok, int_ok=int_ok, tested=tested)


def bh_reject(pvals, alpha=0.05):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    thresh = alpha * (np.arange(1, m + 1)) / m
    passed = p[order] <= thresh
    rej = np.zeros(m, dtype=bool)
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        rej[order[: kmax + 1]] = True
    return rej
