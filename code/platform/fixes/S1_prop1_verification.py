# S1_prop1_verification.py
# Numerical verification accompanying S1_proof.tex (proof of Proposition 1).
# Role: auxiliary, constant-magnitude verification only (per review C3: numerics
# support constants, not the logical chain; the proofs are self-contained).
# Python 3.12, numpy/scipy/mpmath. Run: python S1_prop1_verification.py
import numpy as np
from scipy.special import gammaln, digamma, polygamma
import mpmath as mp

def log_g(N, theta, phi):
    return (gammaln(phi*(1-theta)+N) - gammaln(phi*(1-theta))
            + gammaln(phi) - gammaln(phi+N))
def gfun(N, theta, phi): return np.exp(log_g(N, theta, phi))
def S_N(N, x): return digamma(x+N) - digamma(x)
def U_N(N, x): return polygamma(1, x) - polygamma(1, x+N)

print("="*70)
print("V1 [Lemma S1.2]: 1/(2x) < log x - psi(x) < 1/x for x>0")
xs = np.logspace(-3, 3, 121); d = np.log(xs) - digamma(xs)
assert np.all(d > 1/(2*xs)) and np.all(d < 1/xs); print("  PASS on 121-point grid")

print("="*70)
print("V2 [Cor S1.3]: |S_N(phi) - log(1+N/phi)| <= 1/phi")
worst = 0.0
for phi in [0.3,0.5,1,2,5,10]:
    for N in [1,2,3,5,10,30,100,1000,10000]:
        worst = max(worst, abs(S_N(N,phi)-np.log(1+N/phi))*phi)
assert worst < 1.0; print(f"  PASS, worst err*phi = {worst:.3f}")

print("="*70)
print("V3 [Lemma S1.14]: R(phi)=S_{N2}/S_{N1} strictly increasing; elasticity regimes")
def Rratio(phi,N1,N2): return S_N(N2,phi)/S_N(N1,phi)
def elast(phi,N1,N2,h=1e-6):
    return (np.log(Rratio(phi*(1+h),N1,N2))-np.log(Rratio(phi*(1-h),N1,N2)))/(2*np.log(1+h))
phis = np.linspace(0.3, 20, 400)
for (N1,N2) in [(1,100),(10,1000),(3,30)]:
    Rs = np.array([Rratio(p,N1,N2) for p in phis]); assert np.all(np.diff(Rs)>0)
print("  strict increase PASS")
print(f"  elasticity phi=1: linear zone (0.2,0.5): {elast(1,0.2,0.5):.3f} "
      f"[bound {(0.5-0.2)/2*(1.2)**2*1.5:.3f}]")
print(f"                  crossing (1,10): {elast(1,1,10):.3f}  log zone (100,1000): "
      f"{elast(1,100,1000):.3f} [bound {(2)/np.log(101):.3f}]")

print("="*70)
print("V4 [Lemma S1.11]: ridge derivative = (a^2/2)(S^2-U)+O(a^3), a=phi*theta")
def ridge_deriv(N, pi, theta, phi, h=1e-7):
    c = pi*theta; f = lambda t: t*(1-gfun(N, c/t, phi))
    return (f(pi*(1+h))-f(pi*(1-h)))/(2*pi*h)
for phi in [0.5,1,5]:
    for N in [10,1000]:
        th = 0.001; a = phi*th
        ex = ridge_deriv(N,0.7,th,phi); ap = (a**2/2)*(S_N(N,phi)**2-U_N(N,phi))
        print(f"  phi={phi} N={N}: exact/approx = {ex/ap:.4f}")
print("  N=1 degeneracy: S_1^2-U_1 =", S_N(1,1.3)**2-U_N(1,1.3), "(ridge exact: D(1)=pi*theta)")

print("="*70)
print("V5 [Thm S1.13]: Jacobian condition number = Theta(a^-2); cf. fix_N1 (P4)")
depths = np.logspace(1,4,7)
def jac(pi,theta,phi,Ns):
    J = np.zeros((len(Ns),3))
    for i,N in enumerate(Ns):
        g = gfun(N,theta,phi); Sb = S_N(N,phi*(1-theta)); Sf = S_N(N,phi)
        J[i] = [1-g, pi*phi*g*Sb, -pi*g*((1-theta)*Sb-Sf)]
    return J
for th in [0.3,0.1,0.03,0.01,0.003,0.001]:
    sv = np.linalg.svd(jac(0.8,th,1.0,depths),compute_uv=False)
    print(f"  theta={th:6g}: cond={sv[0]/sv[-1]:.3e}, cond*a^2={(sv[0]/sv[-1])*th*th:.3f}")

print("="*70)
print("V6 [Rem S1.17]: log-log slope of S_N decreases 1 -> 0; half point ~ 2-4 phi")
for phi in [0.5,1,2,5,10]:
    Ns = np.logspace(-2,4,400)
    rs = np.array([N*polygamma(1,N+phi)/S_N(N,phi) for N in Ns])
    assert np.all(np.diff(rs)<0)
    Nh = Ns[np.argmin(np.abs(rs-0.5))]
    print(f"  phi={phi}: monotone PASS, N_1/2 = {Nh:.2f} = {Nh/phi:.2f} phi")

print("="*70)
print("V7 [Lemma S1.10]: 1-g = aS - (a^2/2)(S^2-U) + O(a^3)")
for phi in [0.5,1,5]:
    N = 1000; th = 1e-4; a = phi*th
    lhs = (1-gfun(N,th,phi)) - a*S_N(N,phi)
    rhs = -(a**2/2)*(S_N(N,phi)**2-U_N(N,phi))
    print(f"  phi={phi} N={N} th=1e-4: ratio = {lhs/rhs:.5f}")

print("="*70)
print("V8 [Conj S1.9]: full-rank Jacobian on integer supports (mpmath, dps=100)")
mp.mp.dps = 100
def mp_det_scaled(Ns, pi, theta, phi):
    t,p,pi_ = mp.mpf(theta), mp.mpf(phi), mp.mpf(pi)
    rows = []
    for N in Ns:
        g = mp.gamma(p*(1-t)+N)*mp.gamma(p)/(mp.gamma(p*(1-t))*mp.gamma(p+N))
        Sb = mp.psi(0,p*(1-t)+N)-mp.psi(0,p*(1-t)); Sf = mp.psi(0,p+N)-mp.psi(0,p)
        rows.append([1/g-1, pi_*p*Sb, -pi_*((1-t)*Sb-Sf)])
    (a,b,c),(d,e,f),(gg,h,i) = rows
    return a*(e*i-f*h)-b*(d*i-f*gg)+c*(d*h-e*gg)
cases = [([1,2,3],0.7,0.5,1.0), ([20884,47130,981827],0.811,0.9,83.55),
         ([120933,574378,646044],0.908,0.9,59.05), ([10,300,5000],0.7,0.02,1.5),
         ([5,50,500000],0.5,1e-3,3.0)]
for Ns,pi,th,ph in cases:
    det = mp_det_scaled(Ns,pi,th,ph)
    print(f"  Ns={Ns}, th={th}, phi={ph}: scaled det = {mp.nstr(det,4)} nonzero={det!=0}")
print("="*70)
print("V9 [Lemma S1.9]: Teicher condition -- integer-lattice counterexample")
for N in [1,2,5,10,100,10000]:
    rel = 2*gfun(N,0.5,2.0) - gfun(N,1/3,3.0) - gfun(N,2/3,3.0)
    assert abs(rel) < 1e-12, (N, rel)
print("  2*g(1/2,2) - g(1/3,3) - g(2/3,3) = 0 confirmed at 6 depths")
print("  distinct-exponent case: moment-matrix rank on random configs")
rng = np.random.default_rng(0)
ok = 0
for trial in range(200):
    K = int(rng.integers(2, 6)); th = rng.uniform(0.01, 0.95, K)
    ph = rng.uniform(0.2, 20.0, K)
    prods = ph*th
    if np.min(np.diff(np.sort(prods))) < 1e-3: continue
    Ns = np.arange(1, 400)
    G = np.vstack([np.ones_like(Ns, float)] +
                  [gfun(Ns, th[k], ph[k]) for k in range(K)]).T
    sv = np.linalg.svd(G, compute_uv=False)
    if sv[-1]/sv[0] > 1e-10: ok += 1
print(f"  {ok}/200 distinct-product random configs numerically full rank")
print("ALL VERIFICATIONS DONE")
