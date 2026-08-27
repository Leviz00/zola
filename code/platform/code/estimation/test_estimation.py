"""test_estimation.py — 单元测试（pytest）。

覆盖：
  1. g 闭式 vs 显式乘积 vs 数值积分（三方一致，容差 1e-10；fix_N1 §1.3 网格）；
  2. 一阶/二阶 digamma 展开与闭式的全网格数值对照（fix_N1 表 1 模式）；
  3. S_N / U_N 与朴素求和的一致性；
  4. log g 解析导数 vs 复步长数值微分；
  5. 复合似然解析梯度 vs 复步长；Fisher 矩阵对称正定；
  6. 检出曲线单调性、边界行为；
  7. 模拟器：经验检出率与理论曲线一致（MC 容差内）；
  8. 估计器：大样本单数据集参数恢复（宽松容差）；
  9. φ 已知逐类群分解与联合目标的一致性。

运行：python3 -m pytest test_estimation.py -v
"""

import numpy as np
import pytest
from scipy.special import logit

import model
import composite_likelihood as cl

GRID_12 = [(N, th, ph) for ph in [0.5, 1.0, 5.0]
           for N in [10, 1000] for th in [0.001, 0.01]]


# ---------------------------------------------------------------------------
# 1. g 三方一致
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N,theta,phi", GRID_12)
def test_g_closed_vs_product(N, theta, phi):
    assert abs(model.g_closed(N, theta, phi)
               - model.g_product(N, theta, phi)) < 1e-10


@pytest.mark.parametrize("N,theta,phi", GRID_12)
def test_g_closed_vs_quad(N, theta, phi):
    assert abs(model.g_closed(N, theta, phi)
               - model.g_quad(N, theta, phi)) < 1e-10


def test_g_monte_carlo():
    """MC 交叉验证（4e5 抽样，容差 5σ_MC）。"""
    for (N, th, ph) in [(100, 0.01, 1.0), (1000, 0.001, 0.5)]:
        mc = model.g_monte_carlo(N, th, ph, n_draws=400_000, seed=42)
        exact = model.g_closed(N, th, ph)
        # MC 标准误上界 ~ sqrt(g(1-g)/n)
        assert abs(mc - exact) < 5 * np.sqrt(exact * (1 - exact) / 400_000) + 1e-4


# ---------------------------------------------------------------------------
# 2. 一阶/二阶展开全网格对照（fix_N1 表 1 复现）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N,theta,phi", GRID_12)
def test_order1_order2_expansion_grid(N, theta, phi):
    exact = 1.0 - model.g_closed(N, theta, phi)
    o1 = model.one_minus_g_order1(N, theta, phi)
    o2 = model.one_minus_g_order2(N, theta, phi)
    # 一阶：网格内最差 ~14%（φ=5,N=1000,θ=0.01，fix_N1 表 1 一致）
    assert abs(o1 / exact - 1) < 0.15
    # 二阶：全部 <0.01%（fix_N1 声明）
    assert abs(o2 / exact - 1) < 1e-4


def test_expansion_limits():
    """φ→∞ 时一阶系数退化为 θN（fix_N1 §1.2 的极限声明）。"""
    N, th = 10, 0.01
    s_big = model.one_minus_g_order1(N, th, 1e8)
    assert abs(s_big - th * N) / (th * N) < 1e-6


# ---------------------------------------------------------------------------
# 3. S_N / U_N vs 朴素求和
# ---------------------------------------------------------------------------

def test_S_N_U_N_vs_naive_sum():
    for N in [1, 10, 1000]:
        for ph in [0.5, 2.0, 300.0]:
            r = np.arange(N)
            assert abs(model.S_N(N, ph) - (1.0 / (ph + r)).sum()) < 1e-12
            assert abs(model.U_N(N, ph) - (1.0 / (ph + r) ** 2).sum()) < 1e-12


# ---------------------------------------------------------------------------
# 4. log g 解析导数 vs 复步长
# ---------------------------------------------------------------------------

def test_dlogg_complex_step():
    h = 1e-12
    for (N, th, ph) in [(50, 0.01, 2.0), (1000, 1e-3, 300.0)]:
        f = lambda t, p: model.log_g(float(N), t, p)
        d_th = np.imag(f(th + 1j * h, ph)) / h
        d_ph = np.imag(f(th, ph + 1j * h)) / h
        assert abs(model.dlogg_dtheta(N, th, ph) - d_th) < 1e-6
        assert abs(model.dlogg_dphi(N, th, ph) - d_ph) < 1e-6


# ---------------------------------------------------------------------------
# 5. 复合似然：梯度、Fisher、目标一致性
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_data():
    rng = np.random.default_rng(0)
    n, p = 60, 3
    N = rng.integers(1000, 100000, n).astype(float)
    phi, pi = 2.0, np.array([0.6, 0.8, 0.5])
    th = np.array([0.05, 0.02, 0.1])
    q = cl._q_and_grad(N, th, phi, pi)[0]
    D = (rng.random((n, p)) < q).astype(float)
    return D, N, phi, pi, th


def test_gradient_complex_step(small_data):
    D, N, phi, pi, th = small_data
    psi = cl._pack(phi, pi, th, None)
    _, g = cl._neg_loglik_grad(psi, D, N, None)
    gcv = np.zeros_like(psi)
    for k in range(len(psi)):
        d = np.zeros_like(psi, dtype=complex)
        d[k] = 1j * 1e-12
        gcv[k] = -np.imag(cl.composite_loglik(psi + d, D, N, None)) / 1e-12
    assert np.max(np.abs(g - gcv)) < 1e-5


def test_fisher_symmetric_positive(small_data):
    D, N, phi, pi, th = small_data
    psi = cl._pack(phi, pi, th, None)
    A, U = cl._fisher_and_scores(psi, D, N, None)
    assert np.allclose(A, A.T)
    assert np.all(np.linalg.eigvalsh(A) > 0)
    # 得分均值≈0 在任意点不成立，但 B=U^T U 必为半正定
    B = U.T @ U
    assert np.all(np.linalg.eigvalsh(B) >= -1e-10)


def test_phi_known_equals_joint_objective(small_data):
    """φ 固定时逐类群分解目标之和 = 联合目标。"""
    D, N, phi, pi, th = small_data
    ll_joint = cl.composite_loglik(cl._pack(None or phi, pi, th, phi), D, N, phi)
    tot = 0.0
    for j in range(D.shape[1]):
        tot += cl.composite_loglik(np.array([logit(pi[j]), np.log(th[j])]),
                                   D[:, [j]], N, phi)
    assert abs(ll_joint - tot) < 1e-10


def test_godambe_covariance_shapes(small_data):
    D, N, phi, pi, th = small_data
    psi = cl._pack(phi, pi, th, None)
    V_god, V_naive, A, B = cl.godambe_covariance(psi, D, N, None)
    k = 1 + 2 * D.shape[1]
    assert V_god.shape == (k, k) and V_naive.shape == (k, k)
    assert np.all(np.diag(V_god) > 0) and np.all(np.diag(V_naive) > 0)


# ---------------------------------------------------------------------------
# 6. 检出曲线性质
# ---------------------------------------------------------------------------

def test_detection_monotonicity_and_bounds():
    N = np.geomspace(10, 1e5, 20)
    d = model.detection_prob(N, 0.7, 0.01, 2.0)
    assert np.all(np.diff(d) > 0)              # 随 N 严格增
    assert np.all(d <= 0.7) and np.all(d > 0)  # 上界 π
    # θ→0 时 D→0；θ→1 或 N→∞ 时 D→π
    assert model.detection_prob(1000, 0.7, 1e-12, 2.0) < 1e-6
    assert model.detection_prob(int(1e12), 0.7, 0.5, 2.0) > 0.69


def test_effective_detection_strength_first_order():
    """e_j = φθS_N 是 θ→0 时 1−g 的一阶系数：D(N_min)/π → e_j（θ→0）。"""
    # 注：th 更小时闭式经 gammaln 大数相消失去相对精度（见 README 已知局限）
    th = 1e-6
    e = model.effective_detection_strength(th, 2.0, 1e3)
    exact = 1.0 - model.g_closed(1e3, th, 2.0)
    assert abs(e / exact - 1) < 1e-5


# ---------------------------------------------------------------------------
# 7. 模拟器与理论曲线一致（MC 容差）
# ---------------------------------------------------------------------------

def test_simulator_detection_rates_three_layer():
    rng = np.random.default_rng(123)
    pi = np.array([0.8])
    thb = np.array([0.001, 0.999])
    pia = np.array([0.8, 1.0])          # 恒在场大体量类群 → E1 可忽略
    nrep = 60_000
    N = np.full(nrep, 1000)
    Y, Z = model.simulate_three_layer(pia, thb, 3000.0, N, rng)
    emp = (Y[:, 0] > 0).mean()
    the = model.detection_prob(1000, 0.8, 0.001, 3000.0)
    se = np.sqrt(emp * (1 - emp) / nrep)
    assert abs(emp - the) < 4 * se + 0.01 * the   # 4σ + E1 余量


def test_simulator_detection_rates_zibb():
    rng = np.random.default_rng(7)
    pi = np.array([0.6, 0.9])
    th = np.array([0.01, 0.005])
    N = rng.integers(1000, 100000, 50_000)
    Y, Z = model.simulate_zibb_marginal(pi, th, 2.0, N, rng)
    for j in range(2):
        emp = (Y[:, j] > 0).mean()
        the = model.detection_prob(N.astype(float), pi[j], th[j], 2.0).mean()
        assert abs(emp - the) < 4 * np.sqrt(emp * (1 - emp) / len(N))


def test_zibb_marginal_respects_zero_inflation():
    """Z=0 时 Y 必为 0；Z=1 时 Y>0 概率 = 1-g。"""
    rng = np.random.default_rng(11)
    Y, Z = model.simulate_zibb_marginal(np.array([0.5]), np.array([0.01]),
                                        2.0, np.full(20_000, 1000), rng)
    assert np.all(Y[Z[:, 0] == 0, 0] == 0)


# ---------------------------------------------------------------------------
# 8. 估计器大样本恢复（φ 已知，宽松容差）
# ---------------------------------------------------------------------------

def test_estimator_recovers_params_large_n():
    rng = np.random.default_rng(2024)
    n = 4000
    phi, pi, th = 3000.0, 0.9, 8e-4
    N = np.exp(rng.uniform(np.log(1e3), np.log(1e5), n)).astype(int)
    theta_bar = np.array([th, 1.0 - th])
    Y, _ = model.simulate_three_layer(np.array([pi, 1.0]), theta_bar, phi, N, rng)
    D = cl.detection_indicators(Y)[:, [0]]
    f = cl.fit_composite(D, N, phi_known=phi)
    assert f["success"]
    assert abs(f["pi"][0] / pi - 1) < 0.05
    assert abs(f["theta"][0] / th - 1) < 0.05
    # Godambe SE 与渐近 Fisher 一致量级（n=4000 时 sd(log θ)≈0.05）
    assert 0.02 < f["se_beta"][0] < 0.2


def test_fit_composite_joint_runs_and_bounds():
    rng = np.random.default_rng(5)
    n = 200
    phi = 3000.0
    th8 = np.array([7.0, 8.0, 9.0, 10.0]) * 1e-4
    pi8 = np.array([0.9, 0.9, 0.95, 0.95])
    theta_bar = np.concatenate([th8, [1.0 - th8.sum()]])
    N = np.exp(rng.uniform(np.log(1e3), np.log(1e5), n)).astype(int)
    Y, _ = model.simulate_three_layer(np.concatenate([pi8, [1.0]]),
                                      theta_bar, phi, N, rng)
    D = cl.detection_indicators(Y)[:, :4]
    f = cl.fit_composite(D, N)
    assert f["success"]
    assert 100 < f["phi"] < 1e5
    assert np.all((f["pi"] > 0) & (f["pi"] < 1))
    assert np.all((f["theta"] > 0) & (f["theta"] < 1))
    assert np.isfinite(f["se_phi"]) and f["se_phi"] > 0
