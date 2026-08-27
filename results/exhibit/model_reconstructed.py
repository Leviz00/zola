"""model.py — minimal reconstruction (batch-2 exhibit session, 2026-08-23).

The archived estimation-layer code tree stayed on the original platform
(handover MANIFEST: code/ = reference-only, not copied); twochannel.py
imports exactly one symbol from it. Reconstructed here from the
manuscript's Sec-2 closed form:

    g(N; theta, phi) = E[(1-beta)^N],  beta ~ Beta(phi*theta, phi*(1-theta))
                     = B(phi*theta, phi*(1-theta)+N) / B(phi*theta, phi*(1-theta))

so log_g = betaln(phi*theta, phi*(1-theta)+N) - betaln(phi*theta, phi*(1-theta)).

Verification: end-to-end bridge assertions in run_exhibit_real.py — with
frozen seeds the native-arm p_comb must reproduce the archived
real10k_taxa.csv rows (4-dp) and the archived rejection sets exactly;
a wrong log_g changes qhat, the statistics, and the p-values, and fails
the bridge. Do not use this stub for estimation-layer work beyond log_g.
"""
import numpy as np
from scipy.special import betaln


def log_g(N, theta, phi):
    a = phi * theta
    b = phi * (1.0 - theta)
    N = np.asarray(N, dtype=float)
    return betaln(a, b + N) - betaln(a, b)
