#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iterative CE with RQMC — Per‑Obligor VaRC and SE
-------------------------------------------------
Uses a FIXED, user‑provided loss_threshold (no E[L] or internal threshold calculation).
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple, Optional

from scipy.stats import norm, beta, qmc
from tqdm import tqdm


np.random.seed(42)


@dataclass
class PortfolioSpec:
    u: np.ndarray            # EADs, shape (N,)
    p: np.ndarray            # Unconditional PDs, shape (N,)
    rho_D: np.ndarray        # factor loadings (default), shape (N,)
    rho_L: np.ndarray        # factor loadings (LGD), shape (N,)
    alpha: float = 2.0       # Beta(alpha, beta) for LGD
    beta: float = 5.0
    tau: float = 0.5         # Corr(Z_D, Z_L)


def mean_se_vec(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    R = X.shape[0]
    mu = np.mean(X, axis=0)
    if R > 1:
        se = np.std(X, axis=0, ddof=1) / np.sqrt(R)
    else:
        se = np.zeros_like(mu)
    return mu, se


def cholesky_from_tau(tau: float) -> np.ndarray:
    Sigma = np.array([[1.0, tau], [tau, 1.0]], dtype=float)
    return np.linalg.cholesky(Sigma)


def default_thresholds(p: np.ndarray) -> np.ndarray:
    return norm.ppf(1.0 - np.asarray(p, dtype=float))


def halton_normals(n: int, dim: int, seed: Optional[int] = None) -> np.ndarray:
    engine = qmc.Halton(d=dim, scramble=True, seed=seed)
    u = engine.random(n)
    return norm.ppf(u.clip(1e-12, 1-1e-12))


def simulate_L_and_Li(z: np.ndarray,
                      spec: PortfolioSpec,
                      x_d: np.ndarray,
                      rng: np.random.Generator) -> Tuple[float, np.ndarray]:
    Z_L, Z_D = float(z[0]), float(z[1])
    u, p, rho_D, rho_L = spec.u, spec.p, spec.rho_D, spec.rho_L
    N = len(u)

    eta_D = rng.standard_normal(N)
    eta_L = rng.standard_normal(N)

    X = rho_D * Z_D + np.sqrt(1.0 - rho_D**2) * eta_D
    D = (X > x_d).astype(np.int32)

    Y = rho_L * Z_L + np.sqrt(1.0 - rho_L**2) * eta_L
    U = norm.cdf(Y).clip(1e-12, 1-1e-12)
    eps = beta.ppf(U, a=spec.alpha, b=spec.beta)

    Li = u * eps * D
    L = float(np.sum(Li))
    return L, Li.astype(float)


def ce_pilot_mu(spec: PortfolioSpec,
                loss_threshold: float,
                q_tail: float = 0.5,
                P: int = 2_000_000,
                max_iter: int = 5,
                seed: int = 2025) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chol = cholesky_from_tau(spec.tau)
    mu = np.zeros(2, dtype=float)
    x_d = default_thresholds(spec.p)

    for t in tqdm(range(max_iter), desc="Pilot iterations"):
        n_std = halton_normals(P, dim=2, seed=seed + t)
        z_all = mu + (n_std @ chol.T)

        Ls = np.empty(P, dtype=float)
        for i in range(P):
            L, _ = simulate_L_and_Li(z_all[i], spec, x_d, rng)
            Ls[i] = L

        q_val = np.quantile(Ls, 1.0 - q_tail)
        if q_val > loss_threshold:
            break

        indicators = (Ls > loss_threshold).astype(np.int32)
        if indicators.sum() == 0:
            mu += np.array([0.05, 0.05])
            continue

        Sigma = chol @ chol.T
        Sigma_inv = np.linalg.inv(Sigma)
        mul = (mu @ Sigma_inv)
        lr_prefactor = 0.5 * (mu @ Sigma_inv @ mu)

        log_xi = -(z_all @ mul) + lr_prefactor
        w = indicators * np.exp(log_xi)
        w_sum = w.sum()
        if w_sum <= 0:
            mu += np.array([0.05, 0.05])
            continue
        mu = (w[:, None] * z_all).sum(axis=0) / w_sum

    return mu


def ce_varc_per_obligor_once(spec: PortfolioSpec,
                             a_var: float,
                             delta: float,
                             mu_star: np.ndarray,
                             K: int = 10_000_000,
                             seed: int = 909) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chol = cholesky_from_tau(spec.tau)
    x_d = default_thresholds(spec.p)

    n_std = halton_normals(K, dim=2, seed=seed)
    z_all = mu_star + (n_std @ chol.T)

    Sigma = chol @ chol.T
    Sigma_inv = np.linalg.inv(Sigma)
    mul = (mu_star @ Sigma_inv)
    lr_prefactor = 0.5 * (mu_star @ Sigma_inv @ mu_star)

    N = len(spec.u)
    nume = np.zeros(N, dtype=float)
    deno = 0.0

    for i in range(K):
        z = z_all[i]
        L, Li = simulate_L_and_Li(z, spec, x_d, rng)

        if abs(L - a_var) > delta:
            continue

        log_xi = -(z @ mul) + lr_prefactor
        xi = float(np.exp(log_xi))

        nume += xi * Li
        deno += xi

    if deno == 0.0:
        return np.full(N, np.nan, dtype=float)

    return nume / deno


def run_ce_rqmc_per_obligor(
    N: int = 10,
    p_low: float = 0.01,
    p_high: float = 0.10,
    u_val: float = 1.0,
    rho_val: float = np.sqrt(0.5),
    tau: float = 0.5,
    alpha_lgd: float = 2.0,
    beta_lgd: float = 5.0,
    a_var: float = 1.1432,
    delta: float = 0.005,
    q_tail: float = 0.5,
    loss_threshold: float = 0.6035668192876804,  # FIXED threshold from user
    P: int = 2_000_000,
    K: int = 10_000_000,
    max_iter: int = 20,
    R: int = 10,
    seed: int = 2025
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.linspace(p_low, p_high, N)
    u = np.full(N, u_val)
    rho_D = np.full(N, rho_val)
    rho_L = np.full(N, rho_val)

    spec = PortfolioSpec(u=u, p=p, rho_D=rho_D, rho_L=rho_L,
                         alpha=alpha_lgd, beta=beta_lgd, tau=tau)

    start = time.time()
    mu_star = ce_pilot_mu(spec, loss_threshold=loss_threshold, q_tail=q_tail,
                          P=P, max_iter=max_iter, seed=seed)

    varc_mat = np.zeros((R, N), dtype=float)
    for r in tqdm(range(R), desc="Final replications"):
        varc_mat[r, :] = ce_varc_per_obligor_once(
            spec=spec, a_var=a_var, delta=delta, mu_star=mu_star,
            K=K, seed=seed + 1000 + r
        )

    VaRC_mean, VaRC_se = mean_se_vec(varc_mat)
    end = time.time()
    print(f"Time taken for CE (per-obligor VaRC, R={R}): {end - start:.2f} seconds")
    print(f"mu* = {mu_star}")

    return VaRC_mean, VaRC_se, mu_star


if __name__ == "__main__":
    VaRC_mean, VaRC_se, mu_star = run_ce_rqmc_per_obligor(
        N=10,
        a_var=1.1432,
        delta=0.005,
        loss_threshold=0.6035668192876804,  # use provided l
        P=2_000_000,
        K=10_000_000,
        # demo
        # P=2_000_0,
        # K=10_000_0,
        max_iter=5,
        R=10
    )
    np.set_printoptions(precision=4, suppress=True)
    print("VaRC per obligor (mean):", VaRC_mean)
    print("VaRC per obligor (SE):  ", VaRC_se)
