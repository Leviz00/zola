"""run_rev_anc_gen1.py -- cross-check line: our ANCOMBC 2.6.1 bridge on
frozen-grid cells vs the predecessor's archived numbers
(1005: FDR 0.060 / TPR 0.840; 1009-A: 0.476 / 0.790, intensity truth).
Runs R reps of gen-1 cells through the same bridge; scores vs A (intensity)
truth with its shipped call (Holm q<0.05).
"""
import os, sys, time
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import subprocess
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/claude/ch_smoke")
sys.path.insert(0, "/home/claude/ch_smoke/code/simulation_v3")
import design, generators  # noqa: E402

CFG = pd.read_csv(
    "/home/claude/ch_smoke/code/simulation_v3/configs/config_supplementary.csv"
).set_index("cell_id")
DS = "/home/claude/ch_smoke/rev_ds/anc_gen1"
os.makedirs(DS, exist_ok=True)
RS = "/usr/bin/Rscript"
SCRIPT = "/home/claude/ch_smoke/rev_ancombc.R"

cells = [int(c) for c in sys.argv[1].split(",")] if len(sys.argv) > 1 \
    else [1005, 1009]
R = int(sys.argv[2]) if len(sys.argv) > 2 else 20

rows = []
for cell in cells:
    row = CFG.loc[cell]
    prm = design.params_for_cell(row)
    n = int(row["n"])
    seeds = np.random.SeedSequence(int(row["seed"])).spawn(20)
    for rep in range(R):
        Y, tr = generators.generate(row["mechanism"], prm, n, 100,
                                    row["depth"], seed=seeds[rep])
        g = tr["group"].astype(int)
        A = tr["abs_da_truth"].astype(bool)
        dsdir = os.path.join(DS, f"{cell}_{rep}")
        os.makedirs(dsdir, exist_ok=True)
        np.savetxt(os.path.join(dsdir, "Y.csv"), Y, fmt="%d", delimiter=",")
        pd.DataFrame({"group": g, "N": Y.sum(1)}).to_csv(
            os.path.join(dsdir, "meta.csv"), index=False)
        out = os.path.join(dsdir, "p_ancombc.csv")
        if not os.path.exists(out):
            t0 = time.time()
            pr = subprocess.run([RS, SCRIPT, dsdir, out],
                                capture_output=True, text=True, timeout=1800)
            secs = time.time() - t0
        else:
            secs = 0.0
        if not os.path.exists(out):
            print(f"[{cell},{rep}] FAILED {pr.stderr[-150:]}", flush=True)
            continue
        d = pd.read_csv(out)
        diff = d["diff"].to_numpy().astype(bool)
        nr = int(diff.sum())
        fp = int((diff & ~A).sum()); tp = int((diff & A).sum())
        rows.append(dict(cell=cell, rep=rep, nrej=nr,
                         fdp=fp / nr if nr else 0.0,
                         tpr=tp / A.sum() if A.sum() else np.nan,
                         szero_n=int(d["szero"].sum()),
                         secs=round(secs, 1)))
        print(rows[-1], flush=True)

df = pd.DataFrame(rows)
df.to_csv("/home/claude/ch_smoke/rev/anc_gen1_check.csv", index=False)
print(df.groupby("cell")[["fdp", "tpr", "nrej", "szero_n"]].mean().round(3))
