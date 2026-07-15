from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from config_polar import PolarAnnulusConfig
from data_polar import (
    evaluate_polar_prior,
    inner_boundary_coords,
    outer_boundary_coords,
    r_to_hat,
    sample_bnn_params,
    theta_to_hat,
)
from run_prior_ablation import _rms_preflight


def _pressure_scalar(params, coord, sigma_r, config):
    radius, theta = coord
    phase = (
        params.w1[0, 0] * jnp.sin(theta)
        + params.w1[0, 1] * jnp.cos(theta)
        + params.w1[0, 2] * r_to_hat(radius, config)
        + params.b1[0]
        - jnp.pi / 4.0
    )
    u = jnp.sqrt(2.0 / params.w1.shape[-1]) * jnp.sum(
        params.w2[0] * jnp.cos(phase)
    )
    return config.pressure_scale(sigma_r) * (radius - config.r_outer) * u


def test_registered_rms_coefficients() -> None:
    config = PolarAnnulusConfig(pressure_prior_scaling="boundary_rms")
    expected = (0.4472136, 0.1643990, 0.0995037)
    actual = tuple(config.pressure_scale(sr) for sr in (1.0, 3.0, 5.0))
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5.0e-8)


def test_outer_boundary_and_analytic_derivatives() -> None:
    config = replace(
        PolarAnnulusConfig(), hidden_bnn=64, theta_size=32, cond_chunks=8
    )
    sigma_theta, sigma_r = 1.5, 3.0
    params = sample_bnn_params(
        jax.random.PRNGKey(101), 1, sigma_theta, sigma_r, config
    )
    k_values = jnp.asarray([1.0], dtype=jnp.float32)
    outer = evaluate_polar_prior(
        params, outer_boundary_coords(config), k_values, sigma_r, config
    )
    assert float(jnp.max(jnp.abs(outer.p))) <= 1.0e-7

    physical = jnp.asarray(
        [[0.26, 0.2], [0.47, 1.4], [0.79, 3.3], [0.94, 5.8]],
        dtype=jnp.float32,
    )
    coords = jnp.stack(
        [theta_to_hat(physical[:, 1]), r_to_hat(physical[:, 0], config)], axis=-1
    )
    analytic = evaluate_polar_prior(params, coords, k_values, sigma_r, config)
    scalar = lambda coord: _pressure_scalar(params, coord, sigma_r, config)
    gradients = jax.vmap(jax.grad(scalar))(physical)
    hessians = jax.vmap(jax.hessian(scalar))(physical)
    values = jax.vmap(scalar)(physical)
    p_r = gradients[:, 0]
    p_rr = hessians[:, 0, 0]
    p_theta2 = hessians[:, 1, 1]
    f = p_rr + p_r / physical[:, 0] + p_theta2 / physical[:, 0] ** 2 - values
    np.testing.assert_allclose(analytic.q[0], p_r, rtol=1.0e-6, atol=2.0e-5)
    np.testing.assert_allclose(analytic.q_r[0], p_rr, rtol=1.0e-6, atol=2.0e-5)
    np.testing.assert_allclose(
        analytic.p_theta2[0], p_theta2, rtol=1.0e-6, atol=2.0e-5
    )
    # The polar 1/r^2 term amplifies float32 derivative roundoff near r_inner;
    # the component derivatives above retain the stricter 2e-5 acceptance gate.
    np.testing.assert_allclose(analytic.f[0], f, rtol=1.0e-5, atol=1.0e-4)

    boundary = inner_boundary_coords(config)
    analytic_boundary = evaluate_polar_prior(
        params, boundary, k_values, sigma_r, config
    )
    theta = jnp.pi * (boundary[:, 0] + 1.0)
    physical_boundary = jnp.stack(
        [jnp.full_like(theta, config.r_inner), theta], axis=-1
    )
    outward_flux_ad = -jax.vmap(jax.grad(scalar))(physical_boundary)[:, 0]
    np.testing.assert_allclose(
        -analytic_boundary.q[0], outward_flux_ad, rtol=1.0e-6, atol=2.0e-5
    )


def test_boundary_rms_calibration_matches_direct_q_prior() -> None:
    config = replace(
        PolarAnnulusConfig(pressure_prior_scaling="boundary_rms"),
        hidden_bnn=256,
    )
    records = _rms_preflight(config)
    assert len(records) == len(config.prior_scale_pairs)
    for record in records:
        assert record["diagnostic_samples"] == 2_048
        assert record["relative_mismatch"] <= 0.05, record
