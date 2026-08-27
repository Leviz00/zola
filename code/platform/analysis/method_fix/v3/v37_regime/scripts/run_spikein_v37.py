"""run_spikein_v37.py — M3b：ibdmdb / mbqc spike-in 阳性对照。

prep 与 M3a 完全一致（top-100、平衡子样、解析 Ŵ、分层置换）。
注入：15 个类群（高/中/低流行率各 5，固定 rng 选择），case 组 Y>0 细胞
×2 或 ×4（档内交替），round；缺席细胞保持零；N 保持原始实测库容。
端点：注入回收率（按 fold/流行率档），非注入类群 FDP；est/plac 两臂。
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

V37 = "/mnt/agents/output/analysis/method_fix/v3/v37_regime"
sys.path.insert(0, V37 + "/scripts")
from run_real_v37 import (analytic_what, prep_with_W, make_perms,  # noqa: E402
                          load_ds, group_map_ibdmdb, group_map_mbqc)
from perm_glm import calibrated_test  # noqa: E402

SEED = 20260305


def choose_spiked(Y, k_tier=5, seed=777):
    prev = (Y > 0).mean(axis=0)
    rng = np.random.default_rng(seed)
    tiers = {"high": (prev > 0.7), "mid": (prev > 0.3) & (prev <= 0.7),
             "low": (prev > 0.1) & (prev <= 0.3)}
    picked = []
    for tier, mask in tiers.items():
        cand = np.where(mask)[0]
        take = rng.choice(cand, min(k_tier, len(cand)), replace=False)
        for t in take:
            picked.append((int(t), tier))
    folds = [2.0, 4.0] * 8
    return [(j, tier, folds[i]) for i, (j, tier) in enumerate(picked)]


def main():
    rows = []
    for name, gmap_fn, maxg in (("ibdmdb", group_map_ibdmdb, 10**9),
                                ("mbqc", None, 350)):
        Y, N, names, taxa = load_ds(name)
        gmap = gmap_fn() if gmap_fn else group_map_mbqc(names)
        W = analytic_what(Y, N, name)
        Y2, g, N2, W2, taxa2, df = prep_with_W(Y, N, names, taxa, W, gmap,
                                              max_per_group=maxg)
        spiked = choose_spiked(Y2)
        sp_idx = np.array([j for j, _, _ in spiked])
        Ys = Y2.copy()
        for j, tier, fold in spiked:
            m = g == 1
            Ys[m, j] = np.round(Ys[m, j] * fold)
        truth = np.zeros(Ys.shape[1], dtype=bool)
        truth[sp_idx] = True
        perms = make_perms(name, g, df)
        keep_est = ~((Ys == 0) & (W2 >= 0.5))
        for arm, keep in (("est", keep_est.astype(float)), ("plac", None)):
            r = calibrated_test(Ys, g, N2, W=keep, perms=perms)
            rej = r["reject"]
            fp = int((rej & ~truth).sum())
            tp = int((rej & truth).sum())
            fdp = fp / (fp + tp) if fp + tp else 0.0
            for j, tier, fold in spiked:
                rows.append(dict(dataset=name, arm=arm, taxon=taxa2[j],
                                 tier=tier, fold=fold,
                                 recovered=bool(rej[j]),
                                 p=float(r["pvals"][j]),
                                 q=float(r["qvals"][j])))
            rows.append(dict(dataset=name, arm=arm, taxon="__summary__",
                             tier="", fold=np.nan, recovered=np.nan,
                             p=np.nan, q=np.nan,
                             fwer=r["fwer"], n_rej=int(rej.sum()),
                             n_fp=fp, n_tp=tp, fdp=fdp,
                             power=tp / len(spiked)))
            print(name, arm, rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"{V37}/spikein.csv", index=False)


if __name__ == "__main__":
    main()
