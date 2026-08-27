"""e1_check.py — 归一化耦合误差 E1 的直接量化（fix_N1 §3.2 的 (E1)）。

对主校准设计（φ=3000，8 信息类群 + π=1 大体量类群）与对照设计
（π_bulk=0.99），在大样本 Monte Carlo 下比较三层联合模型的经验检出率与
ZIBB 边际公式 π[1−g(N;θ̄,φ)]。结果写 results/e1_check.csv。
"""

import numpy as np
import pandas as pd
from pathlib import Path

import model

OUT = Path(__file__).parent / "results"
PI8 = np.array([0.85, 0.87, 0.89, 0.91, 0.93, 0.94, 0.95, 0.95])
TH8 = np.array([7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5]) * 1e-4
PHI = 3000.0


def e1_fast(pi_bulk, j, Nb, seed, nrep=200_000):
    rng = np.random.default_rng(seed)
    thb = np.concatenate([TH8, [1.0 - TH8.sum()]])
    pia = np.concatenate([PI8, [pi_bulk]])
    p = len(thb)
    Theta = rng.dirichlet(PHI * thb, size=nrep)
    Z = rng.random((nrep, p)) < pia[None, :]
    C = Theta * Z
    pj = C[:, j] / C.sum(1)
    det = (rng.random(nrep) < (1.0 - (1.0 - pj) ** Nb)) & Z[:, j]
    emp = float(det.mean())
    the = float(model.detection_prob(Nb, PI8[j], TH8[j], PHI))
    se = float(np.sqrt(emp * (1 - emp) / nrep))
    return emp, the, se


if __name__ == "__main__":
    rows = []
    for pi_bulk in [1.0, 0.99]:
        for j in [0, 3, 7]:
            for Nb in [1e3, 1e4, 1e5]:
                emp, the, se = e1_fast(pi_bulk, j, Nb, seed=100 + j)
                rows.append({
                    "pi_bulk": pi_bulk, "taxon": j, "N": Nb,
                    "empirical_detection": emp, "zibb_formula": the,
                    "rel_diff": emp / the - 1.0, "mc_se": se,
                    "mc_se_rel": se / the,
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e1_check.csv", index=False)
    print(df.to_string(index=False))
