#!/usr/bin/env python3
"""Analog (continuous-variable) QKD simulation — minimal prepare-and-measure model.

This script simulates Gaussian-modulated coherent-state prepare-and-measure CV-QKD
over a lossy+noisy channel. It computes empirical correlations and estimates
mutual informations using the Gaussian assumption:

  I(X;Y) = -0.5 * log2(1 - rho^2)

for Pearson correlation rho between variables.

Reverse-reconciliation key rate estimate (very simplified):
  K = beta * I(A;B) - I(B;E)

This is a teaching / prototyping tool — not for production use.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

import numpy as np


@dataclass
class CVChannelParams:
    T: float = 0.6  # channel transmittance (0..1)
    excess_noise: float = 0.01  # added noise variance at receiver (in shot-noise units)
    V_A: float = 10.0  # variance of Alice's Gaussian modulation (SNU)
    beta: float = 0.95  # reconciliation efficiency


def simulate_samples(params: CVChannelParams, n: int, seed: int | None = None):
    rng = np.random.default_rng(seed)
    # Alice prepares X ~ N(0, V_A)
    X = rng.normal(0.0, math.sqrt(params.V_A), size=n)

    # Channel: attenuation sqrt(T) and additive Gaussian noise at Bob
    # Noise variance (shot-noise units): N0 = (1 - T) + T * excess_noise
    # Here we model channel loss as coupling to Eve (pure-loss) plus receiver excess noise
    channel_noise_var = params.excess_noise * params.T
    bob_noise_var = (1.0 - params.T) + channel_noise_var
    N_b = rng.normal(0.0, math.sqrt(bob_noise_var), size=n)
    Y = math.sqrt(params.T) * X + N_b

    # Eve's accessible mode for pure-loss (simple beam-splitter attack)
    # Z = sqrt(1-T) * X + N_e (neglecting additional noise)
    N_e = rng.normal(0.0, 1e-12, size=n)  # negligible thermal noise placeholder
    Z = math.sqrt(max(0.0, 1.0 - params.T)) * X + N_e

    return X, Y, Z


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    num = (x * y).sum()
    den = math.sqrt((x * x).sum() * (y * y).sum())
    return float(num / den) if den != 0 else 0.0


def gaussian_mutual_info_from_rho(rho: float) -> float:
    # I = -0.5 * log2(1 - rho^2)
    rho2 = rho * rho
    if rho2 >= 1.0:
        return float('inf')
    return -0.5 * math.log2(1.0 - rho2)


def symplectic_eigenvalues(gamma: np.ndarray) -> np.ndarray:
    # gamma: real symmetric covariance matrix (2N x 2N)
    N2 = gamma.shape[0]
    assert N2 % 2 == 0
    N = N2 // 2
    # symplectic form
    Omega = np.zeros((N2, N2))
    for i in range(N):
        Omega[2 * i, 2 * i + 1] = 1.0
        Omega[2 * i + 1, 2 * i] = -1.0
    # compute eigenvalues of i Omega gamma
    eigvals = np.linalg.eigvals(1j * Omega.dot(gamma))
    # take absolute values and unique positive ones
    vals = np.abs(eigvals)
    # keep positive real parts (numerical noise may produce tiny imag parts)
    vals = np.real_if_close(vals, tol=1e5)
    # sort and take first N
    vals_sorted = np.sort(vals)
    return vals_sorted[-N:]


def von_neumann_entropy_from_symplectic(nus: np.ndarray) -> float:
    # nus: array of symplectic eigenvalues > 1
    def g(x):
        return (x + 1.0) * math.log2(x + 1.0) - x * math.log2(x)

    s = 0.0
    for nu in np.atleast_1d(nus):
        x = (nu - 1.0) / 2.0
        s += g(x)
    return s


def holevo_bound_heterodyne(params: CVChannelParams) -> dict:
    # Entanglement-based representation parameters
    V = params.V_A + 1.0
    a = V
    c = math.sqrt(max(0.0, V * V - 1.0))
    # channel parameters
    T = params.T
    xi = params.excess_noise  # excess noise referred to channel input (SNU)
    # Bob's variance after channel
    b = T * a + (1.0 - T) + T * xi
    c_prime = math.sqrt(max(0.0, T)) * c

    # build covariance matrix gamma_AB (4x4)
    # block ordering: A (2x2), B (2x2)
    A = np.array([[a, 0.0], [0.0, a]])
    B = np.array([[b, 0.0], [0.0, b]])
    C = np.array([[c_prime, 0.0], [0.0, -c_prime]])
    gamma = np.block([[A, C], [C.T, B]])

    # total entropy S(AB)
    nus = symplectic_eigenvalues(gamma)
    S_AB = von_neumann_entropy_from_symplectic(nus)

    # Conditional CM of A given Bob's heterodyne measurement: A|B = A - C (B + I)^-1 C^T
    I2 = np.eye(2)
    B_plus = B + I2
    inv_Bp = np.linalg.inv(B_plus)
    A_cond = A - C.dot(inv_Bp).dot(C.T)

    # symplectic eigenvalue of single-mode A_cond is sqrt(det(A_cond))
    detAcond = float(np.linalg.det(A_cond))
    nu_cond = math.sqrt(max(1e-16, detAcond))
    S_A_cond = von_neumann_entropy_from_symplectic(np.array([nu_cond]))

    # Holevo between Bob and Eve: chi_BE = S(AB) - S(A|B)
    chi_BE = S_AB - S_A_cond

    # Classical mutual information I(A;B) for heterodyne (Gaussian) in bits
    # using standard formula: I = log2((V + chi_tot)/(1 + chi_tot))
    # where chi_line = (1 - T)/T + xi is channel-added noise referred to input
    chi_line = (1.0 - T) / T + xi
    Vtot = V
    I_AB = math.log2((Vtot + chi_line) / (1.0 + chi_line))

    key_rate = params.beta * I_AB - chi_BE
    return {
        'I_AB': I_AB,
        'chi_BE': chi_BE,
        'S_AB': S_AB,
        'S_A_cond': S_A_cond,
        'key_rate_holevo': key_rate,
    }


def estimate_key_rate(params: CVChannelParams, samples: int = 100_000, seed: int | None = None):
    X, Y, Z = simulate_samples(params, samples, seed=seed)
    rho_AB = pearson_r(X, Y)
    rho_BE = pearson_r(Y, Z)

    I_AB = gaussian_mutual_info_from_rho(rho_AB)
    I_BE = gaussian_mutual_info_from_rho(rho_BE)

    key_rate = params.beta * I_AB - I_BE
    return {
        'I_AB': I_AB,
        'I_BE': I_BE,
        'rho_AB': rho_AB,
        'rho_BE': rho_BE,
        'key_rate': key_rate,
    }


def estimate_key_rate_with_holevo(params: CVChannelParams) -> dict:
    """Estimate key rate using Gaussian Holevo bound (heterodyne model).

    Returns a dict with `I_AB`, `chi_BE`, and `key_rate_holevo`.
    """
    return holevo_bound_heterodyne(params)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Simple analog (CV) QKD simulator")
    p.add_argument('--transmittance', '-T', type=float, default=0.6, help='Channel transmittance')
    p.add_argument('--excess-noise', '-e', type=float, default=0.01, help='Receiver excess noise (SNU)')
    p.add_argument('--mod-variance', '-V', type=float, default=10.0, help="Alice's modulation variance (SNU)")
    p.add_argument('--beta', type=float, default=0.95, help='Reconciliation efficiency')
    p.add_argument('--samples', type=int, default=100000, help='Number of Monte Carlo samples')
    p.add_argument('--seed', type=int, default=None, help='RNG seed')
    p.add_argument('--use-holevo', action='store_true', help='Use Holevo-bound key-rate estimator (heterodyne model)')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    params = CVChannelParams(T=args.transmittance, excess_noise=args.excess_noise, V_A=args.mod_variance, beta=args.beta)
    print('Running CV-QKD simulation (Monte Carlo)')
    print(f'  T={params.T}, excess_noise={params.excess_noise}, V_A={params.V_A}, beta={params.beta}')
    if args.use_holevo:
        res = estimate_key_rate_with_holevo(params)
        print('Results (Holevo-bound, heterodyne model):')
        print(f"  I_AB = {res['I_AB']:.6f} bits")
        print(f"  chi_BE = {res['chi_BE']:.6f} bits")
        print(f"  Estimated key rate (RR) = {res['key_rate_holevo']:.6f} bits/use")
        if res['key_rate_holevo'] <= 0:
            print('  Warning: non-positive estimated key rate (no secure key with these params).')
    else:
        res = estimate_key_rate(params, samples=args.samples, seed=args.seed)
        print('Results (Monte-Carlo classical estimate):')
        print(f"  rho_AB = {res['rho_AB']:.6f}")
        print(f"  rho_BE = {res['rho_BE']:.6f}")
        print(f"  I_AB  = {res['I_AB']:.6f} bits")
        print(f"  I_BE  = {res['I_BE']:.6f} bits")
        print(f"  Estimated key rate (RR) = {res['key_rate']:.6f} bits/use")
        if res['key_rate'] <= 0:
            print('  Warning: non-positive estimated key rate (no secure key with these params).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
