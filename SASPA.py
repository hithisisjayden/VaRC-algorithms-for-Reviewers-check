#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 23:27:36 2025

@author: jaydenwang
"""


import pandas as pd
import numpy as np
from scipy.stats import norm, beta
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
import time
from tqdm import tqdm


np.random.seed(42)


# Parameters
N = 10  # Portfolio size
tau = 0.5  # Correlation between Z_D and Z_L
rho_D = np.sqrt(0.5) * np.ones(N)  # Factor loadings for default, uniform
rho_L = np.sqrt(0.5) * np.ones(N)  # Factor loadings for LGD, uniform
alpha_beta = 2  # Shape parameter alpha for Beta
beta_beta = 5   # Shape parameter beta for Beta
p = np.array([0.01 * (i+1) for i in range(N)])  # Unconditional PDs: 0.01 to 0.10
x_d = norm.ppf(1 - p)  # Default thresholds


# Given VaR values for alphas [0.95,0.96,0.97,0.98,0.99]
alpha_values = [0.95, 0.96, 0.97, 0.98, 0.99]
a_VaR_values = [1.143161852, 1.309826942, 1.53969503, 1.873290874, 2.462527131]
a_VaR_values = [1.1432, 1.3098, 1.5397, 1.8733, 2.4625] 


# # Demo For testing with a single value
# a_VaR_values = [1.1432]


# Simulation parameters (modifiable)
N_outer = 6000  # Number of outer simulations for Z (e.g., 1000 for testing, increase for accuracy)
N_inner = 1000  # Number of inner simulations for L_{-i} and epsilon_i (e.g., 10000)
reps = 10  # Number of repetitions for averaging and SE

# Quadrature params
N_gh = 16 # Enough, 16 fast, 32 slow
N_gl = 16


# Numerical quadrature helpers
def gauss_legendre_on_01(n):
    x, w = leggauss(n)
    e = 0.5 * (x + 1.0)
    w = 0.5 * w
    return e, w

gh_x, gh_w = hermgauss(N_gh)


# Conditional density f_{eps|ZL}(e; zL)
def f_eps_cond(e, zL, rhoL, a_beta, b_beta):
    e = np.clip(e, 1e-14, 1 - 1e-14)
    f_b = beta.pdf(e, a_beta, b_beta)
    u = beta.cdf(e, a_beta, b_beta)
    u = np.clip(u, 1e-16, 1.0 - 1e-16)  # 防止 norm.ppf 溢出
    v = norm.ppf(u)
    s = np.sqrt(max(1.0 - rhoL**2, 1e-14))
    w = (v - rhoL * zL) / s
    num = norm.pdf(w)
    denom = norm.pdf(v) * s
    dens = np.where(denom > 1e-300, f_b * num / denom, 0.0)
    return dens


def M_eps_and_derivs(t, zL, rhoL, a_beta, b_beta, n_gl=N_gl):
    e_nodes, e_weights = gauss_legendre_on_01(n_gl)
    fvals = f_eps_cond(e_nodes, zL, rhoL, a_beta, b_beta)
    t_clipped = np.clip(t, -700, 700)
    et = np.exp(t_clipped * e_nodes)
    M0 = np.sum(e_weights * et * fvals)
    M1 = np.sum(e_weights * e_nodes * et * fvals)
    M2 = np.sum(e_weights * (e_nodes**2) * et * fvals)
    M3 = np.sum(e_weights * (e_nodes**3) * et * fvals)
    M4 = np.sum(e_weights * (e_nodes**4) * et * fvals)
    return M0, M1, M2, M3, M4


# Ai and its derivatives
def A_i_and_derivs(t, zD, zL, rhoDi, rhoLi, xdi, a_beta, b_beta):
    sD = np.sqrt(max(1.0 - rhoDi**2, 1e-14))
    arg = (xdi - rhoDi * zD) / sD
    pz = 1.0 - norm.cdf(arg)
    M0, M1, M2, M3, M4 = M_eps_and_derivs(t, zL, rhoLi, a_beta, b_beta)
    A0 = 1.0 - pz + pz * M0
    A1 = pz * M1
    A2 = pz * M2
    A3 = pz * M3
    A4 = pz * M4
    return A0, A1, A2, A3, A4


# K and derivatives 
def K_and_derivs(t, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta):
    A0s = np.empty(len(xds))
    A1s = np.empty(len(xds))
    A2s = np.empty(len(xds))
    A3s = np.empty(len(xds))
    A4s = np.empty(len(xds))
    for i in range(len(xds)):
        A0, A1, A2, A3, A4 = A_i_and_derivs(t, zD, zL, rhoD_vec[i], rhoL_vec[i], xds[i], a_beta, b_beta)
        A0s[i] = max(A0, 1e-300)
        A1s[i] = A1
        A2s[i] = A2
        A3s[i] = A3
        A4s[i] = A4

    with np.errstate(divide='ignore', invalid='ignore'):
        r1 = np.nan_to_num(A1s / A0s)
        r2 = np.nan_to_num(A2s / A0s)
        r3 = np.nan_to_num(A3s / A0s)
        r4 = np.nan_to_num(A4s / A0s)

    K0 = np.sum(np.log(A0s))
    K1 = np.sum(r1)
    K2 = np.sum(r2 - r1**2)
    K3 = np.sum(r3 - 3 * r1 * r2 + 2 * r1**3)
    K4 = np.sum(r4 - 4 * r1 * r3 - 3 * r2**2 + 12 * (r1**2) * r2 - 6 * r1**4)
    return K0, K1, K2, K3, K4


# Solve saddlepoint K'(t)=a 
def solve_saddlepoint(a_target, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta, t0=0.1, max_iter=60):
    t = t0

    for _ in range(max_iter):
        try:
            _, K1, K2, _, _ = K_and_derivs(t, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta)
        except (OverflowError, ValueError):
            break
        diff = K1 - a_target
        if abs(diff) < 1e-11:
            return t
        if K2 <= 1e-12:
            break
        step = np.sign(diff) * min(abs(diff / K2), 1.0)
        t_new = t - step
        if not np.isfinite(t_new):
            break
        t = t_new

    lo, hi = -50.0, 50.0
    try:
        _, K1_lo, _, _, _ = K_and_derivs(lo, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta)
        if K1_lo > a_target:
            return lo
        _, K1_hi, _, _, _ = K_and_derivs(hi, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta)
        if K1_hi < a_target:
            return hi
    except (OverflowError, ValueError):
        return t0

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mid == lo or mid == hi:
            break
        try:
            _, K1_mid, _, _, _ = K_and_derivs(mid, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta)
            if np.isnan(K1_mid):
                hi = mid if a_target < K1_mid else lo
                continue
            if K1_mid > a_target:
                hi = mid
            else:
                lo = mid
        except (OverflowError, ValueError):
            hi = mid
    return 0.5 * (lo + hi)


# Conditional SPA density
def spa_conditional_density(a_target, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta):
    t_hat = solve_saddlepoint(a_target, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta, t0=0.1)
    K0, _, K2, K3, K4 = K_and_derivs(t_hat, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta)
    if K2 <= 1e-12:
        return 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        lam3 = np.nan_to_num(K3 / (K2**1.5))
        lam4 = np.nan_to_num(K4 / (K2**2))
    pref = np.exp(K0 - t_hat * a_target) / np.sqrt(2.0 * np.pi * K2)
    corr = 1.0 + 0.125 * (lam4 - (5.0 / 3.0) * (lam3**2))
    return max(0.0, pref * corr)


# Unconditional f_L(a) via GH
def f_L_via_SPA(a, tau, rhoD_vec, rhoL_vec, xds, a_beta, b_beta, n_gh=N_gh):
    Sigma = np.array([[1.0, tau], [tau, 1.0]])
    L = np.linalg.cholesky(Sigma)
    x_nodes, w_nodes = hermgauss(n_gh)
    total = 0.0
    factor = 1.0 / np.pi  # 2D GH, 1/(√π)^2 = 1/π
    for i in range(n_gh):
        for j in range(n_gh):
            xvec = np.array([x_nodes[i], x_nodes[j]])
            z = np.sqrt(2.0) * (L @ xvec)
            zL, zD = z[0], z[1]
            w = w_nodes[i] * w_nodes[j] * factor
            dens = spa_conditional_density(a, zD, zL, rhoD_vec, rhoL_vec, xds, a_beta, b_beta)
            if np.isfinite(dens):
                total += w * dens
    return total


def compute_saspa_varc(a_VaR_values, N_outer, N_inner, reps):
    cov_matrix = np.array([[1, tau], [tau, 1]])
    
    results = []  # List to store for each a: mean_VaRC, se_VaRC, rep_times, total_time
    
    for a in a_VaR_values:
        VaRC_reps = []  # List of VaRC arrays for each rep
        rep_times = []  # List of times for each rep
        
        total_start = time.time()
        
        # Denominator f_L(a) via SPA; executed only once
        f_L_a_spa_start = time.time()
        
        f_L_a = f_L_via_SPA(a, tau, rho_D, rho_L, x_d, alpha_beta, beta_beta, n_gh=N_gh)
        
        f_L_a_spa_end = time.time()
        f_L_a_duration = f_L_a_spa_end - f_L_a_spa_start
        print(f"f_L(a) via SPA = {f_L_a:.6g} (calculated once in {f_L_a_duration:.2f}s)")

        
        for r in tqdm(range(reps), desc="Replications"):
            # rep_start = time.time()
            
            A = np.zeros(N)  # Numerators A_i for each obligor
        
            # Outer loop for Z
            for _ in range(N_outer):
                # Sample Z = (Z_L, Z_D)
                Z = np.random.multivariate_normal([0, 0], cov_matrix)
                z_L, z_D = Z[0], Z[1]
                
                # Sample inner paths for all obligors
                eta_L = np.random.randn(N, N_inner)
                eta_D = np.random.randn(N, N_inner)
                
                # Compute Y_j, epsilon_j
                Y = rho_L[:, np.newaxis] * z_L + np.sqrt(1 - rho_L[:, np.newaxis]**2) * eta_L
                U = norm.cdf(Y)
                epsilon = beta.ppf(U, alpha_beta, beta_beta)  # (N, N_inner)
                
                # Compute X_j, D_j
                X = rho_D[:, np.newaxis] * z_D + np.sqrt(1 - rho_D[:, np.newaxis]**2) * eta_D
                D = (X > x_d[:, np.newaxis]).astype(float)  # (N, N_inner)
                
                # Individual losses
                losses = epsilon * D  # (N, N_inner)
                
                # Total L
                L_total = np.sum(losses, axis=0)  # (N_inner,)
                
                # For each i, L_{-i} = L_total - losses_i
                L_minus = L_total[np.newaxis, :] - losses  # (N, N_inner)
                
                # Sort L_{-i} for each i for binary search
                sorted_L_minus = np.sort(L_minus, axis=1)  # (N, N_inner)
                
                # Sample M = N_inner independent epsilon_i | z_L (shared since rho_L same)
                eta_L_i = np.random.randn(N_inner)
                Y_i = rho_L[0] * z_L + np.sqrt(1 - rho_L[0]**2) * eta_L_i  # rho_L same for all
                U_i = norm.cdf(Y_i)
                epsilon_samples = beta.ppf(U_i, alpha_beta, beta_beta)  # (N_inner,)
                
                # Clip epsilon_samples to avoid extreme values
                epsilon_samples = np.clip(epsilon_samples, 1e-9, 1 - 1e-9)
                
                # Vectorized computation for f_cond, f_prime_cond, g
                f_beta = beta.pdf(epsilon_samples, alpha_beta, beta_beta)
                u = beta.cdf(epsilon_samples, alpha_beta, beta_beta)
                v = norm.ppf(u)
                w = (v - rho_L[0] * z_L) / np.sqrt(1 - rho_L[0]**2)
                phi_v = norm.pdf(v)
                phi_w = norm.pdf(w)
                s = np.sqrt(1 - rho_L[0]**2)
                f_cond = np.zeros(N_inner)
                mask_phi = phi_v > 0
                f_cond[mask_phi] = f_beta[mask_phi] * phi_w[mask_phi] / (phi_v[mask_phi] * s)
                
                term1 = np.zeros(N_inner)
                term1[mask_phi] = f_beta[mask_phi] / phi_v[mask_phi] * (v[mask_phi] - w[mask_phi] / s)
                term2 = (alpha_beta - 1) / epsilon_samples - (beta_beta - 1) / (1 - epsilon_samples)
                f_prime_cond = f_cond * (term1 + term2)
                
                g = np.zeros(N_inner)
                mask_f = f_cond > 0
                g[mask_f] = 1 + epsilon_samples[mask_f] * (f_prime_cond[mask_f] / f_cond[mask_f])
                
                arg = (x_d - rho_D * z_D) / np.sqrt(1 - rho_D**2)
                p_cond = 1 - norm.cdf(arg)
                
                # Compute boundary
                boundary = np.zeros(N)
                if a < 1:
                    # Compute f_cond_a
                    ei_a = a
                    ei_a = np.clip(ei_a, 1e-9, 1 - 1e-9)
                    f_beta_a = beta.pdf(ei_a, alpha_beta, beta_beta)
                    u_a = beta.cdf(ei_a, alpha_beta, beta_beta)
                    v_a = norm.ppf(u_a)
                    w_a = (v_a - rho_L[0] * z_L) / s
                    phi_v_a = norm.pdf(v_a)
                    phi_w_a = norm.pdf(w_a)
                    f_cond_a = f_beta_a * phi_w_a / (phi_v_a * s) if phi_v_a > 0 else 0
                    
                    prod_all = np.prod(1 - p_cond)
                    for i in range(N):
                        denom = (1 - p_cond[i])
                        prod_i = prod_all / denom if denom != 0 else 0
                        boundary[i] = - a * f_cond_a * prod_i
                
                inner_values = np.zeros(N)
                
                for i in range(N):
                    # Compute F for all epsilon_samples
                    thresholds = a - epsilon_samples
                    F = np.searchsorted(sorted_L_minus[i], thresholds, side='right') / N_inner
                    
                    sum_term = np.sum(F * g)
                    inner_avg = sum_term / N_inner + boundary[i]
                    inner_values[i] = inner_avg
                
                # Accumulate to A
                A += (p_cond * inner_values) 
                # A += inner_values

            # Divide by N_outer after the loop for efficiency
            A = A / N_outer
            # A = p * A / N_outer
            
            # VaRC_i = A_i / f_L_a
            VaRC = A / f_L_a if f_L_a != 0 else np.zeros(N)
            
            VaRC_reps.append(VaRC)
            
            # rep_end = time.time()
            # rep_duration = rep_end - rep_start
            # rep_times.append(rep_duration)
            # print(f"Rep {r+1} for VaR={a}: Time = {rep_duration:.2f} seconds")
        
        total_end = time.time()
        total_duration = total_end - total_start
        
        # Compute mean and SE
        VaRC_array = np.array(VaRC_reps)  # (reps, N)
        mean_VaRC = np.mean(VaRC_array, axis=0)
        se_VaRC = np.std(VaRC_array, axis=0) / np.sqrt(reps)
        
        results.append({
            'a': a,
            'mean_VaRC': mean_VaRC,
            'se_VaRC': se_VaRC,
            'rep_times': rep_times,
            'total_time': total_duration
        })
    
    return results


# Run the computation
saspa_results = compute_saspa_varc(a_VaR_values, N_outer, N_inner, reps) 


# Print final results
rows = []
for r in saspa_results:
    a_val = r['a']
    mean_v = r['mean_VaRC']
    se_v   = r['se_VaRC']
    cpu    = r['total_time']

    rows.append((f'VaRC (a={a_val:.4f})', mean_v))
    rows.append((f'VaRC S.E. (a={a_val:.4f})', se_v))
    rows.append((f'CPU (a={a_val:.4f})', np.full_like(mean_v, cpu, dtype=float)))

# columns: Obligor 1..N
cols = [f'Obligor {i+1}' for i in range(N)]
# convert rows to dict -> DataFrame
table_dict = {row_name: row_values for row_name, row_values in rows}
Risk_Contributions = pd.DataFrame(table_dict, index=cols).T

Risk_Contributions = Risk_Contributions.round(4)
print(Risk_Contributions)
# Risk_Contributions.to_csv('VaRC SASPA.csv')

