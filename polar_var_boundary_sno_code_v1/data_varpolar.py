from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from scipy.stats import qmc

from config_varpolar import VarPolarConfig


Array = jax.Array


class GeometryParams(NamedTuple):
    w1: Array  # [B, 2, H]
    b1: Array  # [B, H]
    w2: Array  # [B, H]


class GeometryEvaluation(NamedTuple):
    a: Array
    a_theta: Array
    a_theta2: Array


class PolarPriorParams(NamedTuple):
    w1: Array  # [B, 3, H] for [sin(theta), cos(theta), r_hat]
    b1: Array  # [B, H]
    w2: Array  # [B, H]


class PriorEvaluation(NamedTuple):
    p: Array
    q: Array
    f: Array
    p_xi: Array
    p_eta: Array
    p_xixi: Array
    p_xieta: Array
    p_etaeta: Array
    boundary_operator: Array


class SampleBatch(NamedTuple):
    # Shared reference coordinates [theta_hat, r_hat].
    pod_coords: Array
    probe_coords: Array
    boundary_coords: Array

    # Pullbacks of physical fields to the shared reference rectangle.
    p_pod: Array
    f_pod: Array
    p_probe: Array
    f_probe: Array

    # Geometry and the induced non-unit boundary load B_a P.
    boundary_a: Array
    boundary_h: Array
    boundary_load: Array
    geometry_params: GeometryParams

    k_values: Array
    sigma_theta: Array
    sigma_r: Array


class FieldConditionNormalizer(NamedTuple):
    mean_p: Array
    std_p: Array
    mean_f: Array
    std_f: Array
    mean_h: Array
    std_h: Array
    mean_g: Array
    std_g: Array


def theta_to_hat(theta: Array) -> Array:
    return theta / jnp.pi - 1.0


def theta_from_hat(theta_hat: Array) -> Array:
    return jnp.pi * (theta_hat + 1.0)


def make_theta(config: VarPolarConfig) -> Array:
    return jnp.linspace(0.0, 2.0 * jnp.pi, config.theta_size, endpoint=False)


def make_reference_grid(config: VarPolarConfig) -> Array:
    """Regular [Nr, Nt] grid flattened with theta as the fast index."""
    theta = make_theta(config)
    eta = jnp.linspace(-1.0, 1.0, config.radial_size)
    ee, tt = jnp.meshgrid(eta, theta, indexing="ij")
    return jnp.stack([theta_to_hat(tt), ee], axis=-1).reshape(-1, 2)


def sobol_reference_points(
    key: Array,
    n_points: int,
    config: VarPolarConfig,
) -> Array:
    del config
    seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    values = jnp.asarray(sampler.random(n_points), dtype=jnp.float32)
    return 2.0 * values - 1.0


def inner_boundary_coords(config: VarPolarConfig) -> Array:
    theta = make_theta(config)
    return jnp.stack([theta_to_hat(theta), -jnp.ones_like(theta)], axis=-1)


def outer_boundary_coords(config: VarPolarConfig) -> Array:
    theta = make_theta(config)
    return jnp.stack([theta_to_hat(theta), jnp.ones_like(theta)], axis=-1)


def periodic_reference_features(coords_hat: Array) -> Array:
    theta = theta_from_hat(coords_hat[..., 0])
    eta = coords_hat[..., 1]
    return jnp.stack([jnp.sin(theta), jnp.cos(theta), eta], axis=-1)


def sample_geometry_params(
    key: Array,
    batch_size: int,
    config: VarPolarConfig,
) -> GeometryParams:
    key_w1, key_b1, key_w2 = jax.random.split(key, 3)
    hidden = config.hidden_geom_bnn
    return GeometryParams(
        w1=jax.random.normal(key_w1, (batch_size, 2, hidden))
        * config.geom_sigma,
        b1=jax.random.normal(key_b1, (batch_size, hidden)),
        w2=jax.random.normal(key_w2, (batch_size, hidden)),
    )


def evaluate_geometry_single(
    params: GeometryParams,
    theta: Array,
    config: VarPolarConfig,
) -> GeometryEvaluation:
    """Evaluate a(theta), a'(theta), and a''(theta) analytically."""
    theta = jnp.asarray(theta)
    sin_theta = jnp.sin(theta)[..., None]
    cos_theta = jnp.cos(theta)[..., None]
    w_s = params.w1[0]
    w_c = params.w1[1]

    z = sin_theta * w_s + cos_theta * w_c + params.b1
    z_theta = cos_theta * w_s - sin_theta * w_c
    z_theta2 = -sin_theta * w_s - cos_theta * w_c
    phase = z - jnp.pi / 4.0
    scale = 1.0 / jnp.sqrt(params.w1.shape[-1])

    hidden = jnp.sqrt(2.0) * jnp.cos(phase)
    hidden_theta = -jnp.sqrt(2.0) * jnp.sin(phase) * z_theta
    hidden_theta2 = -jnp.sqrt(2.0) * (
        jnp.cos(phase) * z_theta**2 + jnp.sin(phase) * z_theta2
    )

    raw = scale * jnp.sum(hidden * params.w2, axis=-1)
    raw_theta = scale * jnp.sum(hidden_theta * params.w2, axis=-1)
    raw_theta2 = scale * jnp.sum(hidden_theta2 * params.w2, axis=-1)

    c = config.geom_tanh_scale
    tanh_raw = jnp.tanh(c * raw)
    sech2 = 1.0 - tanh_raw**2
    a = config.geom_base + config.geom_amp * tanh_raw
    a_theta = config.geom_amp * c * sech2 * raw_theta
    a_theta2 = config.geom_amp * c * (
        sech2 * raw_theta2
        - 2.0 * c * tanh_raw * sech2 * raw_theta**2
    )
    return GeometryEvaluation(a=a, a_theta=a_theta, a_theta2=a_theta2)


def evaluate_geometry(
    params: GeometryParams,
    theta: Array,
    config: VarPolarConfig,
) -> GeometryEvaluation:
    return jax.vmap(lambda p: evaluate_geometry_single(p, theta, config))(params)


def reference_to_physical_single(
    coords_hat: Array,
    geometry: GeometryParams,
    config: VarPolarConfig,
) -> Array:
    theta = theta_from_hat(coords_hat[..., 0])
    eta = coords_hat[..., 1]
    a = evaluate_geometry_single(geometry, theta, config).a
    radius = a * (3.0 + 2.0 * eta)
    return jnp.stack(
        [radius * jnp.cos(theta), radius * jnp.sin(theta)], axis=-1
    )


def physical_polar_to_reference_single(
    theta: Array,
    radius: Array,
    geometry: GeometryParams,
    config: VarPolarConfig,
) -> Array:
    a = evaluate_geometry_single(geometry, theta, config).a
    eta = 0.5 * (radius / a - 3.0)
    return jnp.stack([theta_to_hat(theta), eta], axis=-1)


def transformed_derivatives(
    p_xi: Array,
    p_eta: Array,
    p_xixi: Array,
    p_xieta: Array,
    p_etaeta: Array,
    a: Array,
    a_theta: Array,
    a_theta2: Array,
    eta: Array,
) -> tuple[Array, Array, Array]:
    """Return ``P_r``, ``P_theta|r`` and the physical polar Laplacian.

    All arguments may be broadcast arrays. Derivatives of ``P`` are with
    respect to the normalized reference coordinates ``(xi, eta)``. Keeping
    this transform public makes the variable-coefficient physics available to
    tests and diagnostics even when ``fe_phys_weight`` is zero.
    """
    s_coord = 3.0 + 2.0 * eta
    h = a_theta / a
    beta = -0.5 * h * s_coord
    p_r = p_eta / (2.0 * a)
    p_theta_fixed_r = p_xi / jnp.pi + beta * p_eta
    p_theta2_fixed_r = (
        p_xixi / (jnp.pi**2)
        + 2.0 * beta * p_xieta / jnp.pi
        + beta**2 * p_etaeta
        + s_coord * (h**2 - a_theta2 / (2.0 * a)) * p_eta
    )
    laplacian = (
        p_etaeta / (4.0 * a**2)
        + p_eta / (2.0 * a**2 * s_coord)
        + p_theta2_fixed_r / (a**2 * s_coord**2)
    )
    return p_r, p_theta_fixed_r, laplacian


def inner_boundary_operator_from_derivatives(
    p_xi: Array,
    p_eta: Array,
    a: Array,
    h: Array,
) -> Array:
    """Evaluate the selected non-unit ``B_a P`` at ``eta=-1``."""
    return h / (a * jnp.pi) * p_xi - (1.0 + h**2) / (2.0 * a) * p_eta


def _uniform_or_constant(
    key: Array,
    batch_size: int,
    bounds: tuple[float, float],
) -> Array:
    lower, upper = bounds
    if upper == lower:
        return jnp.full((batch_size,), lower, dtype=jnp.float32)
    return jax.random.uniform(
        key,
        (batch_size,),
        minval=lower,
        maxval=upper,
        dtype=jnp.float32,
    )


def sample_sigma_pairs(
    key: Array,
    batch_size: int,
    config: VarPolarConfig,
) -> tuple[Array, Array]:
    key_theta, key_r = jax.random.split(key)
    return (
        _uniform_or_constant(key_theta, batch_size, config.sigma_theta_range),
        _uniform_or_constant(key_r, batch_size, config.sigma_r_range),
    )


def sample_k_values(
    key: Array,
    batch_size: int,
    config: VarPolarConfig,
) -> Array:
    """Draw PDE parameters independently from the configured uniform law."""
    return _uniform_or_constant(
        key, batch_size, (config.k_min, config.k_max)
    )


def sample_prior_params(
    key: Array,
    batch_size: int,
    sigma_theta: Array,
    sigma_r: Array,
    config: VarPolarConfig,
) -> PolarPriorParams:
    key_w1, key_b1, key_w2 = jax.random.split(key, 3)
    feature_sigmas = jnp.stack([sigma_theta, sigma_theta, sigma_r], axis=-1)
    bias_sigma = jnp.mean(feature_sigmas, axis=-1)
    return PolarPriorParams(
        w1=jax.random.normal(key_w1, (batch_size, 3, config.hidden_bnn))
        * feature_sigmas[:, :, None],
        b1=jax.random.normal(key_b1, (batch_size, config.hidden_bnn))
        * bias_sigma[:, None],
        w2=jax.random.normal(key_w2, (batch_size, config.hidden_bnn))
        * config.bnn_output_sigma,
    )


def _sinx_over_x(x: Array) -> Array:
    return jnp.sinc(x / jnp.pi)


def _evaluate_prior_points(
    params: PolarPriorParams,
    geometry: GeometryParams,
    coords_hat: Array,
    k_values: Array,
    config: VarPolarConfig,
) -> PriorEvaluation:
    """Analytic q-integral and transformed physical operator on one point chunk."""
    theta = theta_from_hat(coords_hat[:, 0])
    eta = coords_hat[:, 1]
    s_coord = 3.0 + 2.0 * eta
    geom = evaluate_geometry(geometry, theta, config)
    a = geom.a
    a_theta = geom.a_theta
    a_theta2 = geom.a_theta2
    h = a_theta / a

    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    w_s = params.w1[:, 0, :]
    w_c = params.w1[:, 1, :]
    w_eta = params.w1[:, 2, :]
    out = params.w2

    base_phase = (
        w_s[:, None, :] * sin_theta[None, :, None]
        + w_c[:, None, :] * cos_theta[None, :, None]
        + params.b1[:, None, :]
        - jnp.pi / 4.0
    )
    phase = base_phase + w_eta[:, None, :] * eta[None, :, None]
    phase_theta = (
        w_s[:, None, :] * cos_theta[None, :, None]
        - w_c[:, None, :] * sin_theta[None, :, None]
    )
    phase_theta2 = (
        -w_s[:, None, :] * sin_theta[None, :, None]
        - w_c[:, None, :] * cos_theta[None, :, None]
    )

    prior_scale = jnp.sqrt(2.0 / params.w1.shape[-1])
    weighted_out = out[:, None, :]
    cos_phase = jnp.cos(phase)
    sin_phase = jnp.sin(phase)

    q = prior_scale * jnp.sum(weighted_out * cos_phase, axis=-1)
    q_eta = prior_scale * jnp.sum(
        weighted_out * (-sin_phase) * w_eta[:, None, :], axis=-1
    )
    q_theta = prior_scale * jnp.sum(
        weighted_out * (-sin_phase) * phase_theta, axis=-1
    )

    delta = eta[None, :, None] - 1.0
    half_argument = 0.5 * w_eta[:, None, :] * delta
    sinc_term = _sinx_over_x(half_argument)
    midpoint_phase = base_phase + 0.5 * w_eta[:, None, :] * (
        eta[None, :, None] + 1.0
    )
    f_sine = delta * jnp.cos(midpoint_phase) * sinc_term
    g_cosine = -delta * jnp.sin(midpoint_phase) * sinc_term

    integral = prior_scale * jnp.sum(weighted_out * f_sine, axis=-1)
    integral_theta = prior_scale * jnp.sum(
        weighted_out * phase_theta * g_cosine, axis=-1
    )
    integral_theta2 = prior_scale * jnp.sum(
        weighted_out
        * (
            -f_sine * phase_theta**2
            + g_cosine * phase_theta2
        ),
        axis=-1,
    )

    jacobian_r = 2.0 * a
    p = jacobian_r * integral
    p_eta = jacobian_r * q
    p_etaeta = jacobian_r * q_eta
    p_theta_eta = 2.0 * a_theta * q + jacobian_r * q_theta
    p_theta = 2.0 * a_theta * integral + jacobian_r * integral_theta
    p_theta2 = (
        2.0 * a_theta2 * integral
        + 4.0 * a_theta * integral_theta
        + jacobian_r * integral_theta2
    )

    p_xi = jnp.pi * p_theta
    p_xieta = jnp.pi * p_theta_eta
    p_xixi = (jnp.pi**2) * p_theta2

    p_r, p_theta_fixed_r, laplacian = transformed_derivatives(
        p_xi,
        p_eta,
        p_xixi,
        p_xieta,
        p_etaeta,
        a,
        a_theta,
        a_theta2,
        eta[None, :],
    )
    f = laplacian - (k_values[:, None] ** 2) * p

    boundary_operator = -p_r + (h / (a * s_coord[None, :])) * p_theta_fixed_r

    return PriorEvaluation(
        p=p,
        q=q,
        f=f,
        p_xi=p_xi,
        p_eta=p_eta,
        p_xixi=p_xixi,
        p_xieta=p_xieta,
        p_etaeta=p_etaeta,
        boundary_operator=boundary_operator,
    )


def evaluate_prior(
    params: PolarPriorParams,
    geometry: GeometryParams,
    coords_hat: Array,
    k_values: Array,
    config: VarPolarConfig,
) -> PriorEvaluation:
    """Evaluate the prior with point chunking to bound the B*N*H workspace."""
    outputs: list[PriorEvaluation] = []
    chunk_size = min(config.prior_point_chunk_size, coords_hat.shape[0])
    for start in range(0, coords_hat.shape[0], chunk_size):
        outputs.append(
            _evaluate_prior_points(
                params,
                geometry,
                coords_hat[start : start + chunk_size],
                k_values,
                config,
            )
        )
    return PriorEvaluation(
        *(jnp.concatenate([getattr(item, field) for item in outputs], axis=1)
          for field in PriorEvaluation._fields)
    )


def _slice_named_tuple(value, start: int, stop: int):
    return type(value)(*(field[start:stop] for field in value))


def sample_batch(key: Array, config: VarPolarConfig) -> SampleBatch:
    """Generate a self-consistent batch on a shared reference rectangle."""
    key_probe, key_geom, key_sigma, key_prior, key_k = jax.random.split(key, 5)
    pod_coords = make_reference_grid(config)
    probe_coords = sobol_reference_points(
        key_probe, config.random_probe_points, config
    )
    boundary_coords = inner_boundary_coords(config)

    batch_size = config.sample_size
    geometry = sample_geometry_params(key_geom, batch_size, config)
    sigma_theta, sigma_r = sample_sigma_pairs(key_sigma, batch_size, config)
    params = sample_prior_params(
        key_prior,
        batch_size,
        sigma_theta,
        sigma_r,
        config,
    )
    k_values = sample_k_values(key_k, batch_size, config)

    p_pod_parts = []
    f_pod_parts = []
    p_probe_parts = []
    f_probe_parts = []
    a_parts = []
    h_parts = []
    load_parts = []
    theta_boundary = theta_from_hat(boundary_coords[:, 0])

    chunk_size = min(batch_size, config.prior_generation_chunk_size)
    for start in range(0, batch_size, chunk_size):
        stop = min(start + chunk_size, batch_size)
        geom_chunk = _slice_named_tuple(geometry, start, stop)
        param_chunk = _slice_named_tuple(params, start, stop)
        k_chunk = k_values[start:stop]
        pod_eval = evaluate_prior(
            param_chunk, geom_chunk, pod_coords, k_chunk, config
        )
        probe_eval = evaluate_prior(
            param_chunk, geom_chunk, probe_coords, k_chunk, config
        )
        boundary_eval = evaluate_prior(
            param_chunk, geom_chunk, boundary_coords, k_chunk, config
        )
        geom_boundary = evaluate_geometry(
            geom_chunk, theta_boundary, config
        )

        p_pod_parts.append(pod_eval.p)
        f_pod_parts.append(pod_eval.f)
        p_probe_parts.append(probe_eval.p)
        f_probe_parts.append(probe_eval.f)
        a_parts.append(geom_boundary.a)
        h_parts.append(geom_boundary.a_theta / geom_boundary.a)
        load_parts.append(boundary_eval.boundary_operator)

    return SampleBatch(
        pod_coords=pod_coords,
        probe_coords=probe_coords,
        boundary_coords=boundary_coords,
        p_pod=jnp.concatenate(p_pod_parts, axis=0),
        f_pod=jnp.concatenate(f_pod_parts, axis=0),
        p_probe=jnp.concatenate(p_probe_parts, axis=0),
        f_probe=jnp.concatenate(f_probe_parts, axis=0),
        boundary_a=jnp.concatenate(a_parts, axis=0),
        boundary_h=jnp.concatenate(h_parts, axis=0),
        boundary_load=jnp.concatenate(load_parts, axis=0),
        geometry_params=geometry,
        k_values=k_values,
        sigma_theta=sigma_theta,
        sigma_r=sigma_r,
    )


def normalize_p(p: Array, normalizer: FieldConditionNormalizer) -> Array:
    return p / normalizer.std_p


def denormalize_p(p_norm: Array, normalizer: FieldConditionNormalizer) -> Array:
    return p_norm * normalizer.std_p


def normalize_f(f: Array, normalizer: FieldConditionNormalizer) -> Array:
    return (f - normalizer.mean_f) / normalizer.std_f


def denormalize_f(f_norm: Array, normalizer: FieldConditionNormalizer) -> Array:
    return f_norm * normalizer.std_f + normalizer.mean_f


def normalize_h(h: Array, normalizer: FieldConditionNormalizer) -> Array:
    return (h - normalizer.mean_h) / normalizer.std_h


def normalize_g(g: Array, normalizer: FieldConditionNormalizer) -> Array:
    return (g - normalizer.mean_g) / normalizer.std_g


def build_normalizer_online(
    key: Array,
    config: VarPolarConfig,
) -> tuple[FieldConditionNormalizer, Array]:
    sums = {name: jnp.asarray(0.0, dtype=jnp.float32) for name in (
        "p2", "f", "f2", "h", "h2", "g", "g2"
    )}
    counts = {"p": 0, "f": 0, "h": 0, "g": 0}
    for _ in range(config.normalizer_batches):
        key, subkey = jax.random.split(key)
        batch = sample_batch(subkey, config)
        p = jnp.concatenate([batch.p_pod.reshape(-1), batch.p_probe.reshape(-1)])
        f = jnp.concatenate([batch.f_pod.reshape(-1), batch.f_probe.reshape(-1)])
        h = batch.boundary_h.reshape(-1)
        g = batch.boundary_load.reshape(-1)
        sums["p2"] += jnp.sum(p**2)
        sums["f"] += jnp.sum(f)
        sums["f2"] += jnp.sum(f**2)
        sums["h"] += jnp.sum(h)
        sums["h2"] += jnp.sum(h**2)
        sums["g"] += jnp.sum(g)
        sums["g2"] += jnp.sum(g**2)
        counts["p"] += p.size
        counts["f"] += f.size
        counts["h"] += h.size
        counts["g"] += g.size

    eps = jnp.asarray(config.normalizer_eps, dtype=jnp.float32)
    mean_f = sums["f"] / counts["f"]
    mean_h = sums["h"] / counts["h"]
    mean_g = sums["g"] / counts["g"]
    normalizer = FieldConditionNormalizer(
        mean_p=jnp.asarray(0.0, dtype=jnp.float32),
        std_p=jnp.maximum(jnp.sqrt(sums["p2"] / counts["p"]), eps),
        mean_f=mean_f,
        std_f=jnp.maximum(
            jnp.sqrt(jnp.maximum(sums["f2"] / counts["f"] - mean_f**2, 0.0)),
            eps,
        ),
        mean_h=mean_h,
        std_h=jnp.maximum(
            jnp.sqrt(jnp.maximum(sums["h2"] / counts["h"] - mean_h**2, 0.0)),
            eps,
        ),
        mean_g=mean_g,
        std_g=jnp.maximum(
            jnp.sqrt(jnp.maximum(sums["g2"] / counts["g"] - mean_g**2, 0.0)),
            eps,
        ),
    )
    return normalizer, key


def make_source_tokens(latent_f: Array, config: VarPolarConfig) -> Array:
    return latent_f.reshape(
        latent_f.shape[0], config.seq_chunks, config.seq_chunk_width
    )


def make_condition_tokens_from_arrays(
    boundary_coords: Array,
    boundary_a: Array,
    boundary_h: Array,
    boundary_load: Array,
    normalizer: FieldConditionNormalizer,
    config: VarPolarConfig,
) -> Array:
    if boundary_coords.shape != (config.theta_size, 2):
        raise ValueError(
            f"boundary_coords must have shape ({config.theta_size}, 2)."
        )
    if boundary_a.shape != boundary_h.shape or boundary_a.shape != boundary_load.shape:
        raise ValueError("boundary geometry and load arrays must share shape [B, Nt].")
    theta = theta_from_hat(boundary_coords[:, 0])
    batch_size = boundary_a.shape[0]
    sin_theta = jnp.broadcast_to(jnp.sin(theta)[None, :], boundary_a.shape)
    cos_theta = jnp.broadcast_to(jnp.cos(theta)[None, :], boundary_a.shape)
    a_hat = (boundary_a - config.geom_base) / config.geom_amp
    features = jnp.stack(
        [
            sin_theta,
            cos_theta,
            a_hat,
            normalize_h(boundary_h, normalizer),
            normalize_g(boundary_load, normalizer),
        ],
        axis=-1,
    )
    return features.reshape(
        batch_size,
        config.cond_chunks,
        config.cond_chunk_width,
    )


def make_condition_tokens(
    batch: SampleBatch,
    normalizer: FieldConditionNormalizer,
    config: VarPolarConfig,
) -> Array:
    return make_condition_tokens_from_arrays(
        batch.boundary_coords,
        batch.boundary_a,
        batch.boundary_h,
        batch.boundary_load,
        normalizer,
        config,
    )


def target_boundary_from_geometry(
    geometry: GeometryParams,
    config: VarPolarConfig,
) -> tuple[Array, Array, Array, Array]:
    coords = inner_boundary_coords(config)
    theta = theta_from_hat(coords[:, 0])
    values = evaluate_geometry(geometry, theta, config)
    h = values.a_theta / values.a
    target = jnp.cos(theta)[None, :] + h * jnp.sin(theta)[None, :]
    unit_flux = target / jnp.sqrt(1.0 + h**2)
    return values.a, h, target, unit_flux
