import numpy as np
from scipy.special import iv, kv, ivp, kvp


def exact_annulus_solution(
    r: np.ndarray,
    theta: np.ndarray,
    k: float,
    r_inner: float = 0.2,
    r_outer: float = 1.0,
) -> np.ndarray:
    """
    Exact solution of

        ΔP - k^2 P = 0,

    on r_inner <= r <= r_outer, with

        P(r_outer, theta) = 0,
        -P_r(r_inner, theta) = cos(theta).
    """
    r = np.asarray(r)
    theta = np.asarray(theta)

    if abs(k) < 1.0e-10:
        radial = (
            r_inner**2
            / (r_inner**2 + r_outer**2)
            * (r_outer**2 / r - r)
        )
        return radial * np.cos(theta)

    kb = k * r_outer
    ka = k * r_inner

    radial_numerator = (
        iv(1, k * r) * kv(1, kb)
        - kv(1, k * r) * iv(1, kb)
    )

    radial_derivative_at_inner = k * (
        ivp(1, ka, 1) * kv(1, kb)
        - kvp(1, ka, 1) * iv(1, kb)
    )

    radial = -radial_numerator / radial_derivative_at_inner

    return radial * np.cos(theta)