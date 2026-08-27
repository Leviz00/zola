import sys, numpy as np, pandas as pd
sys.path.insert(0, "/mnt/agents/output/code/estimation")
from composite_likelihood import fit_composite, detection_indicators
name = sys.argv[1]
z = np.load(f'/mnt/agents/output/realdata/data/{name}_genus.npz', allow_pickle=True)
Y, dep = z['Y'], z['depths'].astype(float)
s = pd.read_csv(f'/mnt/agents/output/realdata/results/fit_{name}_summary.csv').iloc[0]
df = pd.read_csv(f'/mnt/agents/output/realdata/results/fit_{name}_pertaxon.csv')
prof = fit_composite(detection_indicators(Y), dep, phi_known=float(s['phi_hat']))
la = np.log(prof['pi']/(1-prof['pi'])); lb = np.log(prof['theta'])
out = df.copy()
out['logit_pi_profile'] = la; out['log_theta_profile'] = lb
out['pi_profile'] = prof['pi']; out['theta_profile'] = prof['theta']
d_a = la - df['logit_pi'].to_numpy(); d_b = lb - df['log_theta'].to_numpy()
big = (np.abs(d_a)/np.maximum(df['se_logit_pi'],1e-9) > 2) | (np.abs(d_b)/np.maximum(df['se_log_theta'],1e-9) > 2)
out['moved_gt2SE'] = big
out.to_csv(f'/mnt/agents/output/realdata/results/profile_polish_{name}.csv', index=False)
print('moved >2SE:', int(big.sum()), '/', len(df))
print(out.loc[big, ['taxon','logit_pi','logit_pi_profile','log_theta','log_theta_profile','se_logit_pi','se_log_theta','prevalence','on_boundary_pi','on_boundary_theta']].to_string())
