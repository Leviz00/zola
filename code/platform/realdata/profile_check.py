import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "/mnt/agents/output/code/estimation")
from composite_likelihood import fit_composite, detection_indicators
name = sys.argv[1]
z = np.load(f'/mnt/agents/output/realdata/data/{name}_genus.npz', allow_pickle=True)
Y, dep = z['Y'], z['depths'].astype(float)
s = pd.read_csv(f'/mnt/agents/output/realdata/results/fit_{name}_summary.csv').iloc[0]
df = pd.read_csv(f'/mnt/agents/output/realdata/results/fit_{name}_pertaxon.csv')
t0 = time.time()
prof = fit_composite(detection_indicators(Y), dep, phi_known=float(s['phi_hat']))
el = time.time()-t0
d_alpha = np.log(prof['pi']/(1-prof['pi'])) - df['logit_pi'].to_numpy()
d_beta = np.log(prof['theta']) - df['log_theta'].to_numpy()
se_a, se_b = df['se_logit_pi'].to_numpy(), df['se_log_theta'].to_numpy()
print(f'{name} profile polish {el:.0f}s success={prof["success"]}')
print('max|DlogitPi|=%.4f median=%.2e max|D|/SE=%.3f' % (np.max(np.abs(d_alpha)), np.median(np.abs(d_alpha)), np.max(np.abs(d_alpha)/np.maximum(se_a,1e-9))))
print('max|DlogTheta|=%.4f median=%.2e max|D|/SE=%.3f' % (np.max(np.abs(d_beta)), np.median(np.abs(d_beta)), np.max(np.abs(d_beta)/np.maximum(se_b,1e-9))))
