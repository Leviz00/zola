"""phi_scan.py — profile 对数似然随 φ 的扫描（判断 φ 是否单调冲向边界）。"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/mnt/agents/output/code/estimation")
from composite_likelihood import fit_composite, detection_indicators, composite_loglik
name = sys.argv[1]
phis = [float(x) for x in sys.argv[2:]]
z = np.load(f'/mnt/agents/output/realdata/data/{name}_genus.npz', allow_pickle=True)
Y, dep = z['Y'], z['depths'].astype(float)
D = detection_indicators(Y)
rows = []
for phi in phis:
    out = fit_composite(D, dep, phi_known=phi)
    # 重建 psi 评估 profile 复合对数似然（φ 固定）
    psi = np.concatenate([np.log(out['pi']/(1-out['pi'])), np.log(out['theta'])])
    ll = composite_loglik(psi, D, dep, phi_known=phi)
    rows.append({'phi': phi, 'profile_loglik': ll, 'success': bool(out['success'])})
    print(f'phi={phi:g} profile_ll={ll:.2f} success={out["success"]}', flush=True)
pd.DataFrame(rows).to_csv(f'/mnt/agents/output/realdata/results/phi_scan_{name}.csv', index=False)
