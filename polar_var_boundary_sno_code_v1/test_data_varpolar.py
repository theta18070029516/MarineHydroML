from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from config_varpolar import VarPolarConfig
from data_varpolar import (
    GeometryParams,
    evaluate_geometry_single,
    evaluate_prior,
    inner_boundary_operator_from_derivatives,
    physical_polar_to_reference_single,
    reference_to_physical_single,
    sample_geometry_params,
    sample_k_values,
    sample_prior_params,
    sample_sigma_pairs,
    target_boundary_from_geometry,
    theta_from_hat,
    transformed_derivatives,
)


def small_data_config() -> VarPolarConfig:
    return replace(
        VarPolarConfig(),
        hidden_geom_bnn=16,
        hidden_bnn=16,
        theta_size=8,
        radial_size=4,
        random_probe_points=8,
        sample_size=2,
        prior_generation_chunk_size=1,
        prior_point_chunk_size=16,
        n_basis=8,
        seq_chunks=2,
        cond_chunks=2,
    )


def _single_geometry(batch_geometry: GeometryParams) -> GeometryParams:
    return GeometryParams(*(value[0] for value in batch_geometry))


def test_geometry_derivatives_and_periodic_closure():
    config = small_data_config()
    geometry = _single_geometry(
        sample_geometry_params(jax.random.PRNGKey(1), 1, config)
    )
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 17)
    analytic = evaluate_geometry_single(geometry, theta, config)

    radius = lambda value: evaluate_geometry_single(geometry, value, config).a
    derivative = jax.vmap(jax.grad(radius))(theta)
    derivative2 = jax.vmap(jax.grad(jax.grad(radius)))(theta)

    np.testing.assert_allclose(analytic.a_theta, derivative, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(analytic.a_theta2, derivative2, rtol=5e-5, atol=5e-5)
    np.testing.assert_allclose(analytic.a[0], analytic.a[-1], atol=2e-6)
    np.testing.assert_allclose(analytic.a_theta[0], analytic.a_theta[-1], atol=2e-5)
    assert np.all(np.asarray(analytic.a) > 0.0)
    assert np.all(np.asarray(analytic.a) < 0.2)


def test_coordinate_roundtrip_and_chain_rule_laplacian():
    config = small_data_config()
    geometry = _single_geometry(
        sample_geometry_params(jax.random.PRNGKey(2), 1, config)
    )
    coords = jnp.asarray([0.23, -0.37], dtype=jnp.float32)
    xy = reference_to_physical_single(coords, geometry, config)
    theta = theta_from_hat(coords[0])
    radius = jnp.linalg.norm(xy)
    recovered = physical_polar_to_reference_single(theta, radius, geometry, config)
    np.testing.assert_allclose(recovered, coords, rtol=2e-6, atol=2e-6)

    def physical_pressure(polar):
        t, r = polar
        return jnp.sin(2.0 * t) * r**2 + 0.3 * jnp.cos(t) * r + 0.2 * r**3

    def reference_pressure(reference):
        t = theta_from_hat(reference[0])
        a = evaluate_geometry_single(geometry, t, config).a
        r = a * (3.0 + 2.0 * reference[1])
        return physical_pressure(jnp.stack([t, r]))

    gradient_hat = jax.grad(reference_pressure)(coords)
    hessian_hat = jax.hessian(reference_pressure)(coords)
    geom = evaluate_geometry_single(geometry, theta, config)
    p_r, p_theta, laplacian = transformed_derivatives(
        gradient_hat[0],
        gradient_hat[1],
        hessian_hat[0, 0],
        hessian_hat[0, 1],
        hessian_hat[1, 1],
        geom.a,
        geom.a_theta,
        geom.a_theta2,
        coords[1],
    )

    polar = jnp.stack([theta, radius])
    gradient_physical = jax.grad(physical_pressure)(polar)
    hessian_physical = jax.hessian(physical_pressure)(polar)
    laplacian_direct = (
        hessian_physical[1, 1]
        + gradient_physical[1] / radius
        + hessian_physical[0, 0] / radius**2
    )
    np.testing.assert_allclose(p_r, gradient_physical[1], rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(p_theta, gradient_physical[0], rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(laplacian, laplacian_direct, rtol=2e-4, atol=2e-4)


def test_inner_boundary_operator_matches_cartesian_gradient():
    config = small_data_config()
    geometry = _single_geometry(
        sample_geometry_params(jax.random.PRNGKey(3), 1, config)
    )
    coords = jnp.asarray([-0.61, -1.0], dtype=jnp.float32)
    theta = theta_from_hat(coords[0])
    geom = evaluate_geometry_single(geometry, theta, config)
    h = geom.a_theta / geom.a

    def pressure_xy(xy):
        radius = jnp.linalg.norm(xy)
        angle = jnp.arctan2(xy[1], xy[0])
        return jnp.sin(2.0 * angle) * radius**2 + 0.2 * jnp.cos(angle) * radius

    def pressure_hat(reference):
        xy = reference_to_physical_single(reference, geometry, config)
        return pressure_xy(xy)

    gradient_hat = jax.grad(pressure_hat)(coords)
    transformed = inner_boundary_operator_from_derivatives(
        gradient_hat[0], gradient_hat[1], geom.a, h
    )
    xy = reference_to_physical_single(coords, geometry, config)
    gradient_xy = jax.grad(pressure_xy)(xy)
    e_r = jnp.asarray([jnp.cos(theta), jnp.sin(theta)])
    e_theta = jnp.asarray([-jnp.sin(theta), jnp.cos(theta)])
    cartesian = jnp.dot(-e_r + h * e_theta, gradient_xy)
    np.testing.assert_allclose(transformed, cartesian, rtol=8e-5, atol=8e-5)


def test_pi_prior_outer_boundary_physics_and_circle_limit():
    config = small_data_config()
    key_geometry, key_sigma, key_prior = jax.random.split(jax.random.PRNGKey(4), 3)
    geometry_batch = sample_geometry_params(key_geometry, 1, config)
    geometry = _single_geometry(geometry_batch)
    sigma_theta, sigma_r = sample_sigma_pairs(key_sigma, 1, config)
    prior = sample_prior_params(
        key_prior, 1, sigma_theta, sigma_r, config
    )
    k_value = jnp.asarray([1.1], dtype=jnp.float32)
    theta = jnp.asarray(1.17, dtype=jnp.float32)
    eta = jnp.asarray(0.13, dtype=jnp.float32)
    a = evaluate_geometry_single(geometry, theta, config).a
    radius = a * (3.0 + 2.0 * eta)
    coords = physical_polar_to_reference_single(theta, radius, geometry, config)
    evaluated = evaluate_prior(prior, geometry_batch, coords[None, :], k_value, config)

    def pressure_from_polar(polar):
        reference = physical_polar_to_reference_single(
            polar[0], polar[1], geometry, config
        )
        return evaluate_prior(
            prior, geometry_batch, reference[None, :], k_value, config
        ).p[0, 0]

    polar = jnp.stack([theta, radius])
    gradient = jax.grad(pressure_from_polar)(polar)
    hessian = jax.hessian(pressure_from_polar)(polar)
    direct_laplacian = (
        hessian[1, 1] + gradient[1] / radius + hessian[0, 0] / radius**2
    )
    np.testing.assert_allclose(evaluated.q[0, 0], gradient[1], rtol=3e-4, atol=3e-4)
    np.testing.assert_allclose(
        evaluated.f[0, 0],
        direct_laplacian - k_value[0] ** 2 * evaluated.p[0, 0],
        rtol=1e-3,
        atol=1e-3,
    )

    outer = jnp.stack(
        [jnp.linspace(-1.0, 1.0, 9, endpoint=False), jnp.ones(9)], axis=-1
    )
    outer_values = evaluate_prior(prior, geometry_batch, outer, k_value, config).p
    np.testing.assert_allclose(outer_values, 0.0, atol=2e-6)

    zero_geometry = GeometryParams(
        w1=jnp.zeros_like(geometry_batch.w1),
        b1=jnp.zeros_like(geometry_batch.b1),
        w2=jnp.zeros_like(geometry_batch.w2),
    )
    boundary = jnp.asarray([[0.0, -1.0]], dtype=jnp.float32)
    circle_eval = evaluate_prior(prior, zero_geometry, boundary, k_value, config)
    np.testing.assert_allclose(
        circle_eval.boundary_operator,
        -circle_eval.q,
        rtol=2e-6,
        atol=2e-6,
    )


def test_independent_uniform_parameter_distributions():
    config = small_data_config()
    key_sigma, key_k = jax.random.split(jax.random.PRNGKey(5))
    sigma_theta, sigma_r = sample_sigma_pairs(key_sigma, 4096, config)
    k_values = sample_k_values(key_k, 4096, config)
    theta_np = np.asarray(sigma_theta)
    radial_np = np.asarray(sigma_r)
    k_np = np.asarray(k_values)

    assert theta_np.min() >= 0.5 and theta_np.max() < 2.0
    assert radial_np.min() >= 0.5 and radial_np.max() < 5.0
    assert k_np.min() >= 0.2 and k_np.max() < 2.0
    assert abs(np.corrcoef(theta_np, radial_np)[0, 1]) < 0.08


def test_target_load_and_unit_fem_flux_are_reversible():
    config = small_data_config()
    geometry = sample_geometry_params(jax.random.PRNGKey(6), 2, config)
    _, h, target, unit_flux = target_boundary_from_geometry(geometry, config)
    np.testing.assert_allclose(
        unit_flux * jnp.sqrt(1.0 + h**2), target, rtol=2e-6, atol=2e-6
    )
