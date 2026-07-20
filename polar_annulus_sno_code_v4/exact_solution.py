from __future__ import annotations

import numpy as np
from scipy.special import iv, ivp, kv, kvp


def exact_annulus_fourier_solution(
    r: np.ndarray,
    theta: np.ndarray,
    k: float,
    mode: int = 1,
    phase: float = 0.0,
    amplitude: float = 1.0,
    r_inner: float = 0.2,
    r_outer: float = 1.0,
) -> np.ndarray:
    """Exact solution for a single Fourier Neumann mode.

    The solved problem is

        Delta P - k^2 P = 0,
        P(r_outer, theta) = 0,
        -P_r(r_inner, theta) = amplitude*cos(mode*theta + phase).

    The radial factor is normalized so the inner-boundary flux has unit
    amplitude before applying ``amplitude``.
    """
    r = np.asarray(r, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    if mode < 0:
        raise ValueError("mode must be non-negative.")
    if not r_outer > r_inner > 0.0:
        raise ValueError("Require 0 < r_inner < r_outer.")
    if np.any(r < r_inner) or np.any(r > r_outer):
        raise ValueError("All radii must lie inside the annulus.")

    k_abs = abs(float(k))
    if k_abs < 1.0e-10:
        if mode == 0:
            radial = r_inner * np.log(r_outer / r)
        else:
            numerator = r_outer ** (2 * mode) * r ** (-mode) - r**mode
            denominator = mode * (
                r_outer ** (2 * mode) * r_inner ** (-mode - 1)
                + r_inner ** (mode - 1)
            )
            radial = numerator / denominator
    else:
        kb = k_abs * r_outer
        ka = k_abs * r_inner
        radial_numerator = (
            iv(mode, k_abs * r) * kv(mode, kb)
            - kv(mode, k_abs * r) * iv(mode, kb)
        )
        radial_derivative_at_inner = k_abs * (
            ivp(mode, ka, 1) * kv(mode, kb)
            - kvp(mode, ka, 1) * iv(mode, kb)
        )
        if not np.isfinite(radial_derivative_at_inner) or abs(
            radial_derivative_at_inner
        ) < np.finfo(np.float64).tiny:
            raise FloatingPointError(
                "The Bessel normalization is singular or numerically unstable."
            )
        radial = -radial_numerator / radial_derivative_at_inner

    angular = np.cos(mode * theta + phase)
    result = float(amplitude) * radial * angular
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Exact solution contains NaN or Inf.")
    return result


def exact_annulus_solution(
    r: np.ndarray,
    theta: np.ndarray,
    k: float,
    r_inner: float = 0.2,
    r_outer: float = 1.0,
) -> np.ndarray:
    """Backward-compatible mode-one cosine solution used by v1-v3."""
    return exact_annulus_fourier_solution(
        r,
        theta,
        k,
        mode=1,
        phase=0.0,
        amplitude=1.0,
        r_inner=r_inner,
        r_outer=r_outer,
    )
