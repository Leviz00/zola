"""run_one_v38b.py — Stage B DA 链路 + 可部署退化评估（方案①最终形态）。
退化规则（冻结，data-only）：触发 = corr(scores, 1−det_j) < 0.15；
Ŵ_new[零细胞] ← 1−det_j（det_j = 类群 j 经验检出率，无任何 φ̂ 依赖）。
--no-cal：只输出触发/掩码/AUC 统计，跳过校准检验。"""
import argparse, sys, time
import numpy as np, pandas as pd
SIM = "/mnt/agents/output/code/simulation_v3"
V34 = "/mnt/agents/output/analysis/method_fix/v3/v34_full"
V36 = "/mnt/agents/output/analysis/method_fix/v3/v36_calibration"
sys.path.insert(0, V36); sys.path.insert(1, SIM)
import design, generators  # noqa: E402
from perm_glm import calibrated_test  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
CFG = pd.read_csv(f"{SIM}/configs/config_supplementary.csv")
TAU = 0.15

def mw_auc(s, l):
    r = rankdata(s); p = l == 1; n1, n0 = p.sum(), (~p).sum()
    return np.nan if n1 == 0 or n0 == 0 else (r[p].sum()-n1*(n1+1)/2)/(n1*n0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int, required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-cal", action="store_true")
    a = ap.parse_args()
    row = CFG[CFG.cell_id == a.cell].iloc[0]
    seeds = np.random.SeedSequence(int(row["seed"])).spawn(20)
    prm = design.params_for_cell(row); prm["effect_mode"] = "absolute"
    z = np.load(f"{V34}/npz/cell{a.cell}_rep{a.rep}.npz")
    scores, labels = z["scores"], z["labels"]
    Y, truth = generators.generate(row["mechanism"], prm, n=int(row["n"]),
                                   p=100, depths=int(row["depth"]), seed=seeds[a.rep])
    Y = np.asarray(Y, float); zero = Y == 0
    det = (Y > 0).mean(0)
    s_det = 1 - np.broadcast_to(det[None, :], Y.shape)[zero]
    ok = scores.std() > 0 and s_det.std() > 0
    corr = np.corrcoef(scores, s_det)[0, 1] if ok else 0.0
    fired = bool(corr < TAU and s_det.std() > 0)
    W_new = np.zeros(Y.shape); W_new[zero] = np.where(fired, s_det, scores)
    W_old = np.zeros(Y.shape); W_old[zero] = scores
    keep_new = ~((Y == 0) & (W_new >= 0.5)); keep_old = ~((Y == 0) & (W_old >= 0.5))
    out = dict(cell_id=a.cell, rep=a.rep, fired=fired, corr=corr,
               mask_changed=bool((keep_new != keep_old).any()),
               mask_frac_new=float((~keep_new).mean()),
               mask_frac_old=float((~keep_old).mean()),
               auc_old=mw_auc(scores, labels),
               auc_new=mw_auc(np.where(fired, s_det, scores), labels))
    if not a.no_cal:
        N = truth["depths"].astype(float); group = truth["group"].astype(float)
        da = truth["abs_da_truth"]
        rng = np.random.default_rng(20260305)
        perms = [rng.permutation(group) for _ in range(20)]
        t0 = time.time()
        r = calibrated_test(Y, group, N, W=keep_new.astype(float), perms=perms)
        rej = r["reject"]; fp, tp = int((rej & ~da).sum()), int((rej & da).sum())
        out.update(fdp=fp/(fp+tp) if fp+tp else 0.0, tpr=tp/max(int(da.sum()), 1),
                   fwer=r["fwer"], n_rej=fp+tp, t_cal=time.time()-t0)
    pd.DataFrame([out]).to_csv(a.out, index=False)
    print("saved", a.out, flush=True)

if __name__ == "__main__":
    main()
