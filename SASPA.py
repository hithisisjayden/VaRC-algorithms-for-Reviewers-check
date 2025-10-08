#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.stats import norm, beta
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
import time

# -------------------------
# 全局配置（改这里就行）
# -------------------------
cfg = dict(
    N=10,
    rho_S=0.5,
    rho_D=np.sqrt(0.5),
    rho_L=np.sqrt(0.5),
    alpha_beta=2.0,
    beta_beta=5.0,
    p_min=0.01,          # PD: 0.01, 0.02, ..., 0.10
    seed=42,             # 全局随机种子
    N_outer=1000,        # 外层样本（Z）
    N_inner=1000,        # 内层样本（每个 i 的 ε 与 D）
    reps=10,             # 重复次数，用于SE评估
    N_gh=32,             # Gauss-Hermite 节点数（Z）
    N_gl=32,             # Gauss-Legendre 节点数（e in [0,1]）
    corr_clip=(0.8, 1.2) # Edgeworth 修正裁剪范围
)

alpha_values = [0.95, 0.96, 0.97, 0.98, 0.99]
# 可以用更“整齐”的 a，也可用你给的高精度 a
a_VaR_values = [1.1432, 1.3098, 1.5397, 1.8733, 2.4625]

np.seterr(over='ignore', under='ignore', invalid='ignore', divide='ignore')

# -------------------------
# 构造组合与阈值
# -------------------------
N = cfg['N']
rho_S = cfg['rho_S']
rho_D = cfg['rho_D'] * np.ones(N, dtype=np.float64)
rho_L = cfg['rho_L'] * np.ones(N, dtype=np.float64)
alpha_beta = float(cfg['alpha_beta'])
beta_beta  = float(cfg['beta_beta'])

p = np.array([cfg['p_min'] * (i+1) for i in range(N)], dtype=np.float64)  # 0.01 ~ 0.10
x_d = norm.ppf(1 - p)  # 违约阈值
u_vec = np.ones(N, dtype=np.float64)  # 暴露可改：np.array([...], dtype=np.float64)

# -------------------------
# 求积节点缓存
# -------------------------
def gauss_legendre_on_01(n: int):
    x, w = leggauss(n)
    e = 0.5 * (x + 1.0)
    w = 0.5 * w
    return e.astype(np.float64), w.astype(np.float64)

GH_x, GH_w = hermgauss(cfg['N_gh'])  # for Z
GL_e, GL_w = gauss_legendre_on_01(cfg['N_gl'])

# -------------------------
# 条件密度 f_{ε|ZL}(e | zL)
# 式(13)
# -------------------------
def f_eps_cond(e, zL, rhoL, a_beta, b_beta):
    e = np.clip(e, 1e-14, 1 - 1e-14)
    f_b = beta.pdf(e, a_beta, b_beta)
    u = beta.cdf(e, a_beta, b_beta)
    u = np.clip(u, 1e-16, 1 - 1e-16)
    v = norm.ppf(u)
    s = np.sqrt(max(1.0 - rhoL**2, 1e-14))
    w = (v - rhoL * zL) / s
    num = norm.pdf(w)
    denom = norm.pdf(v) * s
    dens = np.where(denom > 1e-300, f_b * num / denom, 0.0)
    return dens

# -------------------------
# M_{ε|ZL}(t) 及其导数（含 u_i）
# Lemma 3.1 / (18)
# -------------------------
def M_eps_and_derivs(t, zL, rhoL_i, a_beta, b_beta, u_i):
    e_nodes, e_weights = GL_e, GL_w
    fvals = f_eps_cond(e_nodes, zL, rhoL_i, a_beta, b_beta)
    te = np.clip(t * u_i * e_nodes, -700.0, 700.0)
    et = np.exp(te)
    ue = u_i * e_nodes

    M0 = np.sum(e_weights * et * fvals)
    M1 = np.sum(e_weights * (ue)        * et * fvals)
    M2 = np.sum(e_weights * (ue**2.0)   * et * fvals)
    M3 = np.sum(e_weights * (ue**3.0)   * et * fvals)
    M4 = np.sum(e_weights * (ue**4.0)   * et * fvals)
    return M0, M1, M2, M3, M4

# -------------------------
# I_i 及其导数（含 u_i）
# I_i(t|z) = 1 - p_i(zD) + p_i(zD) * M_{L_i|ZL}(t|zL)
# -------------------------
def I_i_and_derivs(t, zD, zL, rhoDi, rhoLi, xdi, a_beta, b_beta, u_i):
    sD = np.sqrt(max(1.0 - rhoDi**2, 1e-14))
    arg = (xdi - rhoDi * zD) / sD
    pz = 1.0 - norm.cdf(arg)  # p_i(zD)

    M0, M1, M2, M3, M4 = M_eps_and_derivs(t, zL, rhoLi, a_beta, b_beta, u_i)
    A0 = 1.0 - pz + pz * M0   # I
    A1 = pz * M1              # I'
    A2 = pz * M2              # I''
    A3 = pz * M3              # I^{(3)}
    A4 = pz * M4              # I^{(4)}
    return A0, A1, A2, A3, A4

# -------------------------
# K 及其导数（log-安全 + 正确 K⁽⁴⁾）
# K = Σ log(I_i)；K'..K⁽⁴⁾ 用 r_k = I^{(k)}/I
# -------------------------
def K_and_derivs(t, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta, u_vec):
    n = len(xds)
    A0s = np.empty(n, dtype=np.float64)
    A1s = np.empty(n, dtype=np.float64)
    A2s = np.empty(n, dtype=np.float64)
    A3s = np.empty(n, dtype=np.float64)
    A4s = np.empty(n, dtype=np.float64)

    for i in range(n):
        A0, A1, A2, A3, A4 = I_i_and_derivs(
            t, zD, zL,
            rhoD_vec[i], rhoL_vec[i], xds[i],
            a_beta, b_beta, u_vec[i]
        )
        A0s[i] = A0
        A1s[i] = A1
        A2s[i] = A2
        A3s[i] = A3
        A4s[i] = A4

    # log-安全（只在log时钳制）
    A0s_safe = np.where(A0s > 0.0, A0s, 1e-300)
    with np.errstate(divide='ignore', invalid='ignore'):
        r1 = np.nan_to_num(A1s / A0s_safe)
        r2 = np.nan_to_num(A2s / A0s_safe)
        r3 = np.nan_to_num(A3s / A0s_safe)
        r4 = np.nan_to_num(A4s / A0s_safe)

    K0 = np.sum(np.log(A0s_safe))
    K1 = np.sum(r1)
    K2 = np.sum(r2 - r1**2)
    K3 = np.sum(r3 - 3.0 * r1 * r2 + 2.0 * r1**3)
    # 正确的 K⁽⁴⁾（+3 r2^2）
    K4 = np.sum(r4 - 4.0 * r1 * r3 + 3.0 * r2**2 + 12.0 * (r1**2) * r2 - 6.0 * r1**4)
    return K0, K1, K2, K3, K4

# -------------------------
# 先扫描找变号区间，再二分
# -------------------------
def bracket_root(a_target, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta, u_vec, lo=-64.0, hi=64.0, m=65):
    grid = np.linspace(lo, hi, m)
    prev_t, prev_val = None, None
    for t in grid:
        _, K1, _, _, _ = K_and_derivs(t, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta, u_vec)
        val = K1 - a_target
        if prev_val is not None and val * prev_val <= 0:
            return prev_t, t
        prev_t, prev_val = t, val
    return -50.0, 50.0  # 兜底

# -------------------------
# 求鞍点：K'(t)=a（warm-start + bracket + 二分）
# -------------------------
def solve_saddlepoint(a_target, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta, u_vec, t0=0.1, max_iter=50):
    # 先用少步牛顿
    t = t0
    for _ in range(max_iter):
        try:
            _, K1, K2, _, _ = K_and_derivs(t, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta, u_vec)
        except (OverflowError, ValueError):
            break
        diff = K1 - a_target
        if abs(diff) < 1e-11:
            return t
        if K2 <= 1e-12 or not np.isfinite(K2):
            break
        step = np.sign(diff) * min(abs(diff / K2), 1.0)
        t_new = t - step
        if not np.isfinite(t_new):
            break
        t = t_new

    # 再 bracket + 二分
    lo, hi = bracket_root(a_target, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta, u_vec)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        try:
            _, K1_mid, _, _, _ = K_and_derivs(mid, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta, u_vec)
            if np.isnan(K1_mid):
                hi = mid
                continue
            if K1_mid > a_target:
                hi = mid
            else:
                lo = mid
        except (OverflowError, ValueError):
            hi = mid
    return 0.5 * (lo + hi)

# -------------------------
# 条件 SPA 密度 f_{L|Z}(a | z)
# (19) + Edgeworth 修正（裁剪）
# 带 warm start（跨节点连续）
# -------------------------
def spa_conditional_density(a_target, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta, u_vec, corr_clip=(0.8, 1.2)):
    # warm start
    if not hasattr(spa_conditional_density, "_twarm"):
        spa_conditional_density._twarm = 0.1
    t_hat = solve_saddlepoint(a_target, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta, u_vec, t0=spa_conditional_density._twarm)
    spa_conditional_density._twarm = t_hat

    K0, _, K2, K3, K4 = K_and_derivs(t_hat, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta, u_vec)
    if K2 <= 1e-14 or not np.isfinite(K2):
        return 0.0, True  # 标记K2退化

    with np.errstate(divide='ignore', invalid='ignore'):
        lam3 = np.nan_to_num(K3 / (K2**1.5))
        lam4 = np.nan_to_num(K4 / (K2**2))
    pref = np.exp(K0 - t_hat * a_target) / np.sqrt(2.0 * np.pi * K2)

    corr = 1.0 + 0.125 * (lam4 - (5.0 / 3.0) * (lam3**2))
    if corr_clip is not None:
        lo, hi = corr_clip
        corr = np.clip(corr, lo, hi)

    dens = pref * corr
    return float(max(0.0, dens)), False  # 非负化 + 无退化

# -------------------------
# 非条件 f_L(a) via 2D GH on correlated Z（(21)）
# -------------------------
def f_L_via_SPA_correlated(a_target, rho_S, rhoD_vec, rhoL_vec, xds, a_beta, b_beta, u_vec, n_gh, corr_clip):
    Sigma = np.array([[1.0, rho_S], [rho_S, 1.0]], dtype=np.float64)
    L = np.linalg.cholesky(Sigma)
    x_nodes, w_nodes = GH_x, GH_w
    total = 0.0
    bad_k2 = 0
    factor = 1.0 / np.pi  # (1/√π)^2

    # 每次计算前重置 warm start，避免跨 a 污染
    if hasattr(spa_conditional_density, "_twarm"):
        delattr(spa_conditional_density, "_twarm")

    for i in range(n_gh):
        for j in range(n_gh):
            xvec = np.array([x_nodes[i], x_nodes[j]])
            z = np.sqrt(2.0) * (L @ xvec)
            zD, zL = z[0], z[1]
            w = w_nodes[i] * w_nodes[j] * factor
            dens, bad = spa_conditional_density(a_target, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta, u_vec, corr_clip)
            if bad:
                bad_k2 += 1
            if np.isfinite(dens):
                total += w * dens
    return total, bad_k2

# -------------------------
# 分子 A_i（仿真-解析，CRN+反抽样）
# 命题 3.1 + IBP + Sort-Search
# -------------------------
def compute_safa_numerator_CRN(N_outer, N_inner, rho_S, rhoD_vec, rhoL_vec, a_target, a_beta, b_beta, xds, u_vec, base_ss: np.random.SeedSequence):
    """
    使用 CRN + 反抽样：
      - 外层：Z ~ N(0, Σ)，取 (z, -z) 成对；共 N_outer 个样本
      - 内层：每个外层样本，都用固定子种子生成 η_L, η_D，保证跨不同 a 可复现
    """
    cov = np.array([[1.0, rho_S], [rho_S, 1.0]], dtype=np.float64)
    Lc = np.linalg.cholesky(cov)
    N = len(xds)
    A_accum = np.zeros(N, dtype=np.float64)

    # 为每个“外层索引”和“符号”预分配子种子
    n_pairs = N_outer // 2
    odd = (N_outer % 2 == 1)

    ss_pairs = base_ss.spawn(n_pairs + (1 if odd else 0))
    # 计算循环
    used = 0
    for pair_idx in range(n_pairs + (1 if odd else 0)):
        rng_z = np.random.default_rng(ss_pairs[pair_idx])
        z_base = rng_z.standard_normal(2) @ Lc  # 一个 z
        signs = (1.0, -1.0) if (used + 2) <= N_outer else (1.0,)  # 尾数为奇时只用一个符号
        for sign in signs:
            used += 1
            zD, zL = (sign * z_base[0], sign * z_base[1])

            # 内层使用可复现的独立子种子
            ss_inner = ss_pairs[pair_idx].spawn(2)
            rng_L = np.random.default_rng(ss_inner[0])
            rng_D = np.random.default_rng(ss_inner[1])

            eta_L = rng_L.standard_normal((N, N_inner))
            eta_D = rng_D.standard_normal((N, N_inner))

            # 生成 eps 与 D
            Y = rhoL_vec[:, None] * zL + np.sqrt(1 - rhoL_vec[:, None]**2) * eta_L
            U = norm.cdf(Y)
            eps = beta.ppf(np.clip(U, 1e-16, 1 - 1e-16), a_beta, b_beta)

            X = rhoD_vec[:, None] * zD + np.sqrt(1 - rhoD_vec[:, None]**2) * eta_D
            D = (X > xds[:, None]).astype(np.float64)

            # 损失与 L^{-i}
            losses = (u_vec[:, None] * eps) * D
            L_total = np.sum(losses, axis=0)              # (N_inner,)
            L_minus = L_total[None, :] - losses           # (N, N_inner)
            sorted_L_minus = np.sort(L_minus, axis=1)     # each i row-sorted

            # 条件违约概率及 ∏(1-p)
            arg = (xds - rhoD_vec * zD) / np.sqrt(1 - rhoD_vec**2)
            p_cond = 1.0 - norm.cdf(arg)
            prod_all = np.prod(1.0 - p_cond)

            inner_vals = np.zeros(N, dtype=np.float64)

            # 逐 i 计算（这部分若要更快可以半向量化）
            for i in range(N):
                # 重新从 f_{ε_i|Z_L} 采样（独立于上面的 eps）
                eta_L_i = rng_L.standard_normal(N_inner)
                Yi = rhoL_vec[i] * zL + np.sqrt(1 - rhoL_vec[i]**2) * eta_L_i
                Ui = norm.cdf(Yi)
                eps_smp = beta.ppf(np.clip(Ui, 1e-16, 1 - 1e-16), a_beta, b_beta)
                eps_smp = np.clip(eps_smp, 1e-10, 1 - 1e-10)

                # f 与 f'（式(13)(14)）
                f_beta_vals = beta.pdf(eps_smp, a_beta, b_beta)
                u_ = beta.cdf(eps_smp, a_beta, b_beta)
                v = norm.ppf(np.clip(u_, 1e-16, 1 - 1e-16))
                s = np.sqrt(1 - rhoL_vec[i]**2)
                wv = (v - rhoL_vec[i] * zL) / s
                phi_v = norm.pdf(v)
                phi_w = norm.pdf(wv)

                f_cond = np.zeros_like(eps_smp)
                mask = phi_v > 1e-300
                f_cond[mask] = f_beta_vals[mask] * phi_w[mask] / (phi_v[mask] * s)

                term1 = np.zeros_like(eps_smp)
                term1[mask] = f_beta_vals[mask] / phi_v[mask] * (v[mask] - wv[mask] / s)
                term2 = (a_beta - 1.0) / eps_smp - (b_beta - 1.0) / (1.0 - eps_smp)
                f_prime = f_cond * (term1 + term2)

                # g(e) = 1 + e * f'/f
                g = np.ones_like(eps_smp)
                good = f_cond > 1e-300
                g[good] = 1.0 + eps_smp[good] * (f_prime[good] / f_cond[good])

                # F_{L^{-i}|Z}(a - u_i e)
                thresholds = a_target - u_vec[i] * eps_smp
                F = np.searchsorted(sorted_L_minus[i], thresholds, side='right') / N_inner

                # 边界项：仅当 a/u_i < 1
                boundary_i = 0.0
                ei_cap = min(1.0, a_target / max(u_vec[i], 1e-300))
                if ei_cap < 1.0:
                    ei_a = float(np.clip(ei_cap, 1e-10, 1 - 1e-10))
                    f_beta_a = beta.pdf(ei_a, a_beta, b_beta)
                    u_a = beta.cdf(ei_a, a_beta, b_beta)
                    v_a = norm.ppf(np.clip(u_a, 1e-16, 1 - 1e-16))
                    sL = np.sqrt(1 - rhoL_vec[i]**2)
                    w_a = (v_a - rhoL_vec[i] * zL) / sL
                    phi_v_a = norm.pdf(v_a)
                    phi_w_a = norm.pdf(w_a)
                    f_cond_a = (f_beta_a * phi_w_a / (phi_v_a * sL)) if phi_v_a > 1e-300 else 0.0

                    denom_i = max(1.0 - p_cond[i], 1e-300)
                    prod_except_i = prod_all / denom_i

                    boundary_i = - a_target * f_cond_a * prod_except_i

                # 命题 3.1 积分核前的 u_i
                inner_vals[i] = np.mean(F * (u_vec[i] * g)) + boundary_i

            A_accum += (p_cond * inner_vals)

    return A_accum / N_outer

# -------------------------
# 主流程：SASPA（CRN 跨 a + 诊断）
# -------------------------
def compute_saspa(a_values, cfg):
    results = []
    rng_master = np.random.SeedSequence(cfg['seed'])
    # 为（reps）生成主子种子
    ss_reps = rng_master.spawn(cfg['reps'])

    for a in a_values:
        VaRC_reps = []
        rep_times = []
        print(f"\n=== SASPA for VaR a = {a:.9f} ===")

        # 分母：SPA（每个 a 只算一次）
        t0_den = time.time()
        f_den, badK2 = f_L_via_SPA_correlated(
            a, cfg['rho_S'], rho_D, rho_L, x_d,
            cfg['alpha_beta'], cfg['beta_beta'], u_vec,
            cfg['N_gh'], cfg['corr_clip']
        )
        t1_den = time.time()
        print(f"[Den] f_L(a) via SPA = {f_den:.6g} | time {t1_den - t0_den:.2f}s | badK2={badK2}")

        for r in range(cfg['reps']):
            t0 = time.time()
            # 为该 rep 生成 CRN 基种子（跨 a 重复）
            ss_base = ss_reps[r]
            numer = compute_safa_numerator_CRN(
                cfg['N_outer'], cfg['N_inner'],
                cfg['rho_S'], rho_D, rho_L,
                a, cfg['alpha_beta'], cfg['beta_beta'],
                x_d, u_vec, ss_base
            )
            VaRC = numer / max(f_den, 1e-300)
            VaRC_reps.append(VaRC)
            t1 = time.time()
            rep_times.append(t1 - t0)
            print(f"[Num] rep {r+1}/{cfg['reps']}  time {t1 - t0:.2f}s")

        VaRC_reps = np.array(VaRC_reps, dtype=np.float64)
        mean_VaRC = VaRC_reps.mean(axis=0)
        se_VaRC = VaRC_reps.std(axis=0, ddof=1) / np.sqrt(cfg['reps'])

        # Full allocation 自检
        sum_VaRC = float(np.sum(mean_VaRC))
        rel_err = abs(sum_VaRC - a) / max(a, 1e-12)
        print(f"[Chk] ΣVaRC = {sum_VaRC:.6f}, a = {a:.6f}, rel_err = {100*rel_err:.3f}%")

        results.append({
            'a': a,
            'mean_VaRC': mean_VaRC,
            'se_VaRC': se_VaRC,
            'rep_times': rep_times,
            'den_time': t1_den - t0_den,
            'num_time_mean': float(np.mean(rep_times)),
            'rel_error_full_alloc': rel_err,
            'badK2_count': badK2
        })
    return results

# -------------------------
# 运行 & 导出
# -------------------------
if __name__ == "__main__":
    saspa_results = compute_saspa(a_VaR_values, cfg)

    # 汇总表（含 CPU & 误差诊断）
    cols = {}
    for i, result in enumerate(saspa_results):
        alpha = alpha_values[i]
        cols[f'VaRC {alpha}'] = result['mean_VaRC']
        cols[f'VaRC S.E. {alpha}'] = result['se_VaRC']
        cols[f'CPU Num Mean {alpha} (s)'] = np.full_like(result['mean_VaRC'], result['num_time_mean'], dtype=np.float64)
        cols[f'CPU Den {alpha} (s)'] = np.full_like(result['mean_VaRC'], result['den_time'], dtype=np.float64)
        cols[f'Alloc Err {alpha} (%)'] = np.full_like(result['mean_VaRC'], 100*result['rel_error_full_alloc'], dtype=np.float64)
        cols[f'Bad K2 {alpha} (cnt)'] = np.full_like(result['mean_VaRC'], result['badK2_count'], dtype=np.float64)

    Risk_Contributions = pd.DataFrame(
        cols,
        index=pd.Index([f'Obligor {i+1}' for i in range(N)])
    ).T

    print("\n--- Final Risk Contributions (SASPA, CRN+Antithetic) ---")
    print(Risk_Contributions)
    # Risk_Contributions.to_csv('RhoS=0.5_VaRC_SASPA_CRN.csv', float_format="%.10f")
