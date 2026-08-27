"""Regression bridges for the ZOLA package (added 2026-08-23; additive only —
no frozen script is modified).

Replays the exact verification bridges from the project's archived
engineering records as executable tests, so any future environment or code
drift is caught against archived ground truth:

  T1  log_g equivalence: original Gamma-form (code/platform) vs the betaln
      reconstruction (code/testing) on a 168-point grid.        [fast, no data]
  T2  battery bridge, cell 2001 rep 0 (three_layer_real path):
      frozen SeedSequence -> generators_ext -> twochannel; every scored
      field must equal the archived fix3_v4_detail.csv row.     [~15 s]
  T3  battery bridge, cell 2005 rep 0 (legacy `generators` path
      via ADV-BB mechanism).                                    [~16 s]
  T4  cohort bridge (native AGP official list, K=9999) — OPTIONAL:
      runs only when ZOLA_DATA_ROOT points at a directory laid out like
      /mnt/user-data/uploads (npz + metadata); skipped otherwise.

Run:  python tests/test_regression_bridges.py           (T1–T3; T4 if data)
      ZOLA_FAST=1 python tests/test_regression_bridges.py   (T1 only)
Also collectable by pytest. Set OPENBLAS/OMP/MKL threads to 1 for exact
reproduction (done below), matching the archived runs (2-CPU container).
"""
import os, sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P_SIM = os.path.join(ROOT, "code", "platform", "code", "simulation_v3")
P_EST = os.path.join(ROOT, "code", "platform", "code", "estimation_v3")
P_TESTING = os.path.join(ROOT, "code", "testing")
P_GEN2 = os.path.join(ROOT, "code", "generation2")
for _p in (P_SIM, P_EST, P_TESTING, P_GEN2):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Archived expectations (sources in comments; do not edit without a new
# bridge run and an archive entry).
EXPECT_2001 = dict(nrej=12, fdp=0.16666666666666666, tpr=1.0, tprP=1.0,
                   typeI=0.03333333333333333,
                   lib_p=0.5939049833229416)      # fix3_v4_detail.csv 2001,0
EXPECT_2005 = dict(nrej=12, fdp=0.25, tpr=0.9, tprA=0.9,
                   typeI=0.08888888888888889, oracle_tpr=0.8,
                   lib_p=0.22365750523137062)     # fix3_v4_detail.csv 2005,0
EXPECT_AGP = dict(n_rej=14)                       # real10k_summary.csv


def test_log_g_equivalence():
    import importlib.util as ilu

    def load(name, path):
        spec = ilu.spec_from_file_location(name, path)
        mod = ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    orig = load("model_orig", os.path.join(P_EST, "model.py"))
    recon = load("model_recon",
                 os.path.join(P_TESTING, "model_RECONSTRUCTED.py"))
    Ns = np.array([1, 10, 100, 1000, 19672, 1e5, 1.12e6])
    worst = 0.0
    for th in (1e-6, 1e-4, 1e-2, 0.113, 0.5, 0.99):
        for ph in (3.0, 100.0, 1453.585, 1e5):
            d = np.max(np.abs(np.asarray(orig.log_g(Ns, th, ph))
                              - np.asarray(recon.log_g(Ns, th, ph))))
            worst = max(worst, float(d))
    assert worst < 1e-6, f"log_g drift: {worst}"
    print(f"T1 log_g equivalence OK (max |delta| {worst:.2e})")


def _one_run(cell, rep):
    """Portable replica of run_fix3_v4.one_run's scoring for one dataset
    (the frozen runner hard-codes container paths inside the worker; this
    replica imports through the package layout instead — statistics and
    seeds identical)."""
    import time
    from scipy.stats import ranksums
    from generators_ext import generate_ext
    from twochannel import two_channel_test, bh_reject, median_ratio_offset
    from run_pilot_v4 import CELLS

    spec = CELLS[cell]
    seed = np.random.SeedSequence([20260819, cell]).spawn(20)[rep]
    Y, tr = generate_ext(spec["mech"], spec["params"], spec["n"], 100,
                         spec["depth"], seed=seed)
    g = tr["group"]; N = tr["depths"].astype(float)
    A = tr["abs_da_truth"].astype(bool)
    Pt = tr.get("pres_da_truth", np.zeros(100, dtype=bool)).astype(bool)
    Ut = A | Pt
    lib = Y.sum(axis=1).astype(float)
    lr = ranksums(lib[g == 1], lib[g == 0]).pvalue
    nu = median_ratio_offset(Y, n_ref=50)
    res = two_channel_test(Y, lib, g, nu=nu, K=999,
                           seed=[20260819, 9, cell, rep])
    rej = bh_reject(res["p_comb"], 0.05)

    def ft(rej, truth):
        nr = int(rej.sum()); fp = int((rej & ~truth).sum())
        tp = int((rej & truth).sum())
        return nr, (fp / nr if nr else 0.0), \
            (tp / truth.sum() if truth.sum() else np.nan)

    nr, fdp, tpr = ft(rej, Ut)
    _, _, tprA = ft(rej, A)
    _, _, tprP = ft(rej, Pt)
    out = dict(nrej=nr, fdp=fdp, tpr=tpr, tprA=tprA, tprP=tprP,
               typeI=float((res["p_comb"][~Ut] < 0.05).mean()), lib_p=lr)
    if cell in (2005, 2006) and A.any():
        hits = 0
        pres = tr["presence"]
        for j in np.where(A)[0]:
            m = pres[:, j]
            if (m & (g == 1)).sum() >= 3 and (m & (g == 0)).sum() >= 3:
                ra = Y[m, j] / N[m]
                pv = ranksums(ra[g[m] == 1], ra[g[m] == 0]).pvalue
                hits += pv < 0.05 / max(A.sum(), 1)
        out["oracle_tpr"] = hits / A.sum()
    return out


def _check(cell, got, exp):
    for k, v in exp.items():
        gv = float(got[k])
        assert abs(gv - float(v)) < 1e-9, \
            f"cell {cell} field {k}: got {gv}, expected {v}"
    print(f"T battery bridge cell {cell} rep 0 EXACT "
          f"(lib_p={got['lib_p']:.16f})")


def test_battery_bridge_2001():
    if os.environ.get("ZOLA_FAST"):
        print("T2 skipped (ZOLA_FAST)"); return
    _check(2001, _one_run(2001, 0), EXPECT_2001)


def test_battery_bridge_2005():
    if os.environ.get("ZOLA_FAST"):
        print("T3 skipped (ZOLA_FAST)"); return
    _check(2005, _one_run(2005, 0), EXPECT_2005)


def test_cohort_bridge_agp_optional():
    root = os.environ.get("ZOLA_DATA_ROOT")
    if not root:
        print("T4 skipped (set ZOLA_DATA_ROOT to run the cohort bridge)")
        return
    import run_rich as rr
    rr.UP = root
    rr.K = 9999
    from twochannel import fit_detection_curves, median_ratio_offset, \
        bh_reject
    name, (Y, N, g, taxa, st, cl, tag) = \
        list(rr.load_cohorts(("agp",)).items())[0]
    keep = np.sort(np.argsort(-(Y > 0).mean(0))[:100])
    Yk = Y[:, keep]
    nu = median_ratio_offset(Yk)
    qhat = fit_detection_curves((Yk > 0).astype(float), N)["qhat"]
    res = rr.two_channel_m(Yk, N, g, qhat, nu, seed=[20260826, tag],
                           strata=st, clusters=cl)
    n_rej = int(bh_reject(res["p_comb"], 0.05).sum())
    assert n_rej == EXPECT_AGP["n_rej"], f"AGP n_rej {n_rej} != 14"
    print("T4 cohort bridge (AGP native, K=9999) EXACT: 14 rejections")


if __name__ == "__main__":
    test_log_g_equivalence()
    test_battery_bridge_2001()
    test_battery_bridge_2005()
    test_cohort_bridge_agp_optional()
    print("ALL BRIDGES GREEN")
