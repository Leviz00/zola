"""run_rev_ancombc.py -- SPEC-REV-COMPUTE R2 driver: ANCOM-BC2 battery.

Datasets identical to the fix3 battery / wrap comparison (same generator,
same SeedSequence([20260819, cell]).spawn(20)[rep]). Scoring:
  primary  = its shipped call (diff: Holm q<0.05) vs union truth
  also     = raw p archived per dataset for wrap A0/A2 arms
  typeI    = raw p<0.05 rate on null taxa
Usage: python3 run_rev_ancombc.py [cells_csv] [R]
"""
import os, sys, time
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import subprocess
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude/ch_smoke")
from run_fix3_v4 import FIX_CELLS
from generators_ext import generate_ext

OUT = "/home/claude/ch_smoke/rev"
DS = "/home/claude/ch_smoke/rev_ds/anc"
os.makedirs(OUT, exist_ok=True)
os.makedirs(DS, exist_ok=True)
RS = "/usr/bin/Rscript"
SCRIPT = "/home/claude/ch_smoke/rev_ancombc.R"

cells = ([int(c) for c in sys.argv[1].split(",")] if len(sys.argv) > 1
         else sorted(FIX_CELLS))
R = int(sys.argv[2]) if len(sys.argv) > 2 else 20


def score(rej, truth):
    nr = int(rej.sum()); fp = int((rej & ~truth).sum())
    tp = int((rej & truth).sum())
    return (fp / nr if nr else 0.0), (tp / truth.sum() if truth.sum()
                                      else np.nan)


rows = []
t00 = time.time()
for cell in cells:
    spec = FIX_CELLS[cell]
    for rep in range(R):
        seed = np.random.SeedSequence([20260819, cell]).spawn(20)[rep]
        Y, tr = generate_ext(spec["mech"], spec["params"], spec["n"], 100,
                             spec["depth"], seed=seed)
        g = tr["group"].astype(int)
        A = tr["abs_da_truth"].astype(bool)
        Pt = tr.get("pres_da_truth", np.zeros(100, dtype=bool)).astype(bool)
        Ut = A | Pt
        dsdir = os.path.join(DS, f"{cell}_{rep}")
        os.makedirs(dsdir, exist_ok=True)
        np.savetxt(os.path.join(dsdir, "Y.csv"), Y, fmt="%d", delimiter=",")
        pd.DataFrame({"group": g, "N": Y.sum(1)}).to_csv(
            os.path.join(dsdir, "meta.csv"), index=False)
        out = os.path.join(dsdir, "p_ancombc.csv")
        if not os.path.exists(out):
            t0 = time.time()
            pr = subprocess.run([RS, SCRIPT, dsdir, out],
                                capture_output=True, text=True,
                                timeout=1800)
            secs = time.time() - t0
        else:
            secs = 0.0
        if not os.path.exists(out):
            print(f"[{cell},{rep}] FAILED: {pr.stderr[-200:]}", flush=True)
            rows.append(dict(cell=cell, rep=rep, failed=1, secs=round(secs, 1)))
            continue
        d = pd.read_csv(out)
        pv, qv = d["p"].to_numpy(), d["q"].to_numpy()
        diff = d["diff"].to_numpy().astype(bool)
        passed = d["passed"].to_numpy().astype(bool)
        szero = d["szero"].to_numpy().astype(bool)
        tested = np.isfinite(pv)
        fdp, tpr = score(diff, Ut)
        fdp_ss, tpr_ss = score(diff & passed, Ut)
        nulls = ~Ut
        typeI = float((pv[nulls & tested] < 0.05).mean()) if (
            nulls & tested).any() else np.nan
        rows.append(dict(cell=cell, rep=rep, failed=0,
                         n_tested=int(tested.sum()),
                         n_filtered=int((~tested).sum()),
                         nrej=int(diff.sum()), fdp=fdp, tpr=tpr,
                         nrej_ss=int((diff & passed).sum()),
                         fdp_ss=fdp_ss, tpr_ss=tpr_ss,
                         szero_n=int(szero.sum()),
                         szero_hitP=int((szero & Pt).sum()),
                         typeI=typeI, secs=round(secs, 1)))
        print(f"[{cell},{rep}] nrej={rows[-1]['nrej']} "
              f"fdp={fdp:.3f} tpr={tpr if tpr==tpr else -1:.3f} "
              f"typeI={typeI:.3f} {secs:.0f}s "
              f"({time.time()-t00:.0f}s total)", flush=True)
    pd.DataFrame(rows).to_csv(f"{OUT}/ancombc_battery_detail.csv",
                              index=False)

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/ancombc_battery_detail.csv", index=False)
ok = df[df.failed == 0]
s = ok.groupby("cell")[["fdp", "tpr", "fdp_ss", "tpr_ss", "typeI",
                        "nrej", "n_filtered", "szero_n"]].mean().round(3)
s.to_csv(f"{OUT}/ancombc_battery_summary.csv")
print(s.to_string())
