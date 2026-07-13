from __future__ import annotations

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
from scipy.stats import qmc

from config_polar import PolarAnnulusConfig


Array = jax.Array


class PolarBNNParams(NamedTuple):
    """Parameters of q(r, theta) = P_r.

    w1: [B, 3, H] for features [sin(theta), cos(theta), r_hat]
    b1: [B, H]
    w2: [B, H]
    """

    w1: Array
    b1: Array
    w2: Array


class PriorEvaluation(NamedTuple):
    p: Array          # [B, N]
    q: Array          # P_r, [B, N]
    q_r: Array        # P_rr, [B, N]
    p_theta2: Array   # P_{theta theta}, [B, N]
    f: Array          # Delta P - k^2 P, [B, N]


class SampleBatch(NamedTuple):
    # Normalized polar coordinates [theta_hat, r_hat].
    boundary_coords: Array  # [B, Nt, 2]
    boundary_flux: Array    # g_n = -P_r on r=r_inner, [B, Nt]
    pod_coords: Array       # [Npod, 2]
    probe_coords: Array     # [Nprobe, 2]

    p_pod: Array            # [B, Npod]
    f_pod: Array            # [B, Npod]
    p_probe: Array          # [B, Nprobe]
    f_probe: Array          # [B, Nprobe]
    k_values: Array         # [B]


class FieldNormalizer(NamedTuple):
    mean_p: Array
    std_p: Array
    mean_f: Array
    std_f: Array


def theta_to_hat(theta: Array) -> Array:
    """Map physical theta in [0, 2*pi] to theta_hat in [-1, 1]."""
    return theta / jnp.pi - 1.0


def theta_from_hat(theta_hat: Array) -> Array:
    return jnp.pi * (theta_hat + 1.0)


def r_to_hat(r: Array, config: PolarAnnulusConfig) -> Array:
    return 2.0 * (r - config.r_inner) / config.radial_length - 1.0


def r_from_hat(r_hat: Array, config: PolarAnnulusConfig) -> Array:
    return config.r_inner + 0.5 * config.radial_length * (r_hat + 1.0)


def periodic_polar_features(coords_hat: Array) -> Array:
    """Convert [theta_hat, r_hat] to [sin(theta), cos(theta), r_hat]."""
    theta_hat = coords_hat[..., 0]
    r_hat = coords_hat[..., 1]
    theta = theta_from_hat(theta_hat)
    return jnp.stack([jnp.sin(theta), jnp.cos(theta), r_hat], axis=-1)


def make_theta(config: PolarAnnulusConfig) -> Array:
    return jnp.linspace(
        0.0,
        2.0 * jnp.pi,
        config.theta_size,
        endpoint=False,
    )


def make_polar_grid(config: PolarAnnulusConfig) -> Array:
    """Regular normalized polar grid, flattened in [Nr, Nt] order."""
    theta = make_theta(config)
    radial = jnp.linspace(config.r_inner, config.r_outer, config.radial_size)
    rr, tt = jnp.meshgrid(radial, theta, indexing="ij")
    return jnp.stack([theta_to_hat(tt), r_to_hat(rr, config)], axis=-1).reshape(-1, 2)


def sobol_polar_points(
    key: Array,
    n_points: int,
    config: PolarAnnulusConfig,
) -> Array:
    """Sobol points uniformly distributed in the normalized rectangle [-1,1]^2.

    This means theta is uniform and the physical radius r is uniform, matching the
    current annulus convention rather than physical-area-uniform sampling.
    """
    seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    u = jnp.asarray(sampler.random(n_points), dtype=jnp.float32)
    theta_hat = 2.0 * u[:, 0] - 1.0
    r_hat = 2.0 * u[:, 1] - 1.0
    return jnp.stack([theta_hat, r_hat], axis=-1)


def inner_boundary_coords(config: PolarAnnulusConfig) -> Array:
    theta = make_theta(config)
    theta_hat = theta_to_hat(theta)
    r_hat = -jnp.ones_like(theta_hat)
    return jnp.stack([theta_hat, r_hat], axis=-1)


def outer_boundary_coords(config: PolarAnnulusConfig) -> Array:
    theta = make_theta(config)
    theta_hat = theta_to_hat(theta)
    r_hat = jnp.ones_like(theta_hat)
    return jnp.stack([theta_hat, r_hat], axis=-1)


def sample_bnn_params(
    key: Array,
    batch_size: int,
    sigma_theta: float,
    sigma_r: float,
    config: PolarAnnulusConfig,
) -> PolarBNNParams:
    """Sample a single-hidden-layer Fourier BNN for q=P_r."""
    key_w1, key_b1, key_w2 = jax.random.split(key, 3)
    feature_sigmas = jnp.asarray(
        [sigma_theta, sigma_theta, sigma_r],
        dtype=jnp.float32,
    )
    w1 = jax.random.normal(
        key_w1,
        (batch_size, 3, config.hidden_bnn),
    ) * feature_sigmas[None, :, None]
    bias_sigma = jnp.mean(feature_sigmas)
    b1 = (
        jax.random.normal(key_b1, (batch_size, config.hidden_bnn))
        * bias_sigma
    )
    w2 = (
        jax.random.normal(key_w2, (batch_size, config.hidden_bnn))
        * config.bnn_output_sigma
    )
    return PolarBNNParams(w1=w1, b1=b1, w2=w2)


def _sinx_over_x(x: Array) -> Array:
    """Stable sin(x)/x, including the exact limit at x=0."""
    # jnp.sinc(y) = sin(pi*y)/(pi*y), hence y=x/pi gives sin(x)/x.
    return jnp.sinc(x / jnp.pi)


def evaluate_polar_prior(
    params: PolarBNNParams,
    coords_hat: Array,
    k_values: Array,
    config: PolarAnnulusConfig,
) -> PriorEvaluation:
    r"""Evaluate q=P_r, its analytic radial integral P, and the PDE source.

    The BNN is

        q(r,theta) = sqrt(2/H) sum_j a_j cos(A_j(theta)+w_rj*r_hat),

    where

        A_j(theta)=w_sj sin(theta)+w_cj cos(theta)+b_j-pi/4.

    Since r_hat is affine in r, P=int_{r_outer}^r q(s,theta) ds has a closed form.
    The implementation uses the stable identity

        [sin(A+w*r_hat)-sin(A+w)]/w
        = (r_hat-1) cos(A+w(r_hat+1)/2)
          sinc(w(r_hat-1)/2),

    so w=0 is handled by its exact limiting form without a numerical branch.
    """
    if coords_hat.ndim != 2 or coords_hat.shape[-1] != 2:
        raise ValueError("coords_hat must have shape [N, 2].")

    batch_size = params.w1.shape[0]
    if k_values.shape != (batch_size,):
        raise ValueError(
            f"k_values must have shape ({batch_size},), got {k_values.shape}."
        )

    theta_hat = coords_hat[:, 0]
    r_hat = coords_hat[:, 1]
    theta = theta_from_hat(theta_hat)
    radius = r_from_hat(r_hat, config)

    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)

    w_s = params.w1[:, 0, :]  # [B,H]
    w_c = params.w1[:, 1, :]
    w_r = params.w1[:, 2, :]
    out = params.w2

    base_phase = (
        w_s[:, None, :] * sin_theta[None, :, None]
        + w_c[:, None, :] * cos_theta[None, :, None]
        + params.b1[:, None, :]
        - jnp.pi / 4.0
    )
    phase = base_phase + w_r[:, None, :] * r_hat[None, :, None]

    scale = jnp.sqrt(2.0 / params.w1.shape[-1])
    weighted_out = out[:, None, :]

    cos_phase = jnp.cos(phase)
    sin_phase = jnp.sin(phase)

    # q is defined directly as the physical derivative P_r.
    q = scale * jnp.sum(weighted_out * cos_phase, axis=-1)

    # d r_hat / dr supplies the physical radial chain-rule factor.
    q_r = scale * jnp.sum(
        weighted_out
        * (-sin_phase)
        * w_r[:, None, :]
        * config.drhat_dr,
        axis=-1,
    )

    delta = r_hat[None, :, None] - 1.0
    half_argument = 0.5 * w_r[:, None, :] * delta
    sinc_term = _sinx_over_x(half_argument)
    midpoint_phase = (
        base_phase
        + 0.5 * w_r[:, None, :] * (r_hat[None, :, None] + 1.0)
    )

    # Stable difference quotients:
    # F = [sin(A+w*r_hat)-sin(A+w)]/w
    # G = [cos(A+w*r_hat)-cos(A+w)]/w
    f_sine = delta * jnp.cos(midpoint_phase) * sinc_term
    g_cosine = -delta * jnp.sin(midpoint_phase) * sinc_term

    integral_scale = 1.0 / config.drhat_dr
    p = scale * integral_scale * jnp.sum(
        weighted_out * f_sine,
        axis=-1,
    )

    phase_theta = (
        w_s[:, None, :] * cos_theta[None, :, None]
        - w_c[:, None, :] * sin_theta[None, :, None]
    )
    phase_theta2 = (
        -w_s[:, None, :] * sin_theta[None, :, None]
        - w_c[:, None, :] * cos_theta[None, :, None]
    )

    p_theta2 = scale * integral_scale * jnp.sum(
        weighted_out
        * (
            -f_sine * phase_theta**2
            + g_cosine * phase_theta2
        ),
        axis=-1,
    )

    f = (
        q_r
        + q / radius[None, :]
        + p_theta2 / (radius[None, :] ** 2)
        - (k_values[:, None] ** 2) * p
    )

    return PriorEvaluation(p=p, q=q, q_r=q_r, p_theta2=p_theta2, f=f)


def sample_batch(key: Array, config: PolarAnnulusConfig) -> SampleBatch:
    """Generate a complete PI-sampler batch.

    For every prior scale pair and every repeat, this function samples an
    independent BNN realization. The same BNN parameters are evaluated on the POD,
    probe, and inner-boundary coordinates, guaranteeing exact sample consistency.
    """
    key_probe, key_groups = jax.random.split(key)
    pod_coords = make_polar_grid(config)
    probe_coords = sobol_polar_points(
        key_probe,
        config.random_probe_points,
        config,
    )
    bnd_coords_single = inner_boundary_coords(config)

    group_specs: list[tuple[float, float]] = []
    for scale_pair in config.prior_scale_pairs:
        group_specs.extend([scale_pair] * config.repeats_per_scale)

    group_keys = jax.random.split(key_groups, len(group_specs))

    p_pod_list = []
    f_pod_list = []
    p_probe_list = []
    f_probe_list = []
    k_list = []
    flux_list = []

    for group_key, (sigma_theta, sigma_r) in zip(group_keys, group_specs):
        key_params, key_k = jax.random.split(group_key)
        params = sample_bnn_params(
            key_params,
            config.sample_size,
            sigma_theta,
            sigma_r,
            config,
        )
        if config.k_max == config.k_min:
            k_values = jnp.full(
                (config.sample_size,),
                config.k_min,
                dtype=jnp.float32,
            )
        else:
            k_values = jax.random.uniform(
                key_k,
                (config.sample_size,),
                minval=config.k_min,
                maxval=config.k_max,
            )

        pod_eval = evaluate_polar_prior(params, pod_coords, k_values, config)
        probe_eval = evaluate_polar_prior(params, probe_coords, k_values, config)
        bnd_eval = evaluate_polar_prior(params, bnd_coords_single, k_values, config)

        p_pod_list.append(pod_eval.p)
        f_pod_list.append(pod_eval.f)
        p_probe_list.append(probe_eval.p)
        f_probe_list.append(probe_eval.f)
        k_list.append(k_values)

        # On the inner boundary the outward normal of the annular domain is -e_r.
        # Therefore g_n = dP/dn = -P_r = -q.
        flux_list.append(-bnd_eval.q)

    p_pod = jnp.concatenate(p_pod_list, axis=0)
    f_pod = jnp.concatenate(f_pod_list, axis=0)
    p_probe = jnp.concatenate(p_probe_list, axis=0)
    f_probe = jnp.concatenate(f_probe_list, axis=0)
    k_values = jnp.concatenate(k_list, axis=0)
    boundary_flux = jnp.concatenate(flux_list, axis=0)

    boundary_coords = jnp.broadcast_to(
        bnd_coords_single[None, :, :],
        (config.effective_batch_size, config.theta_size, 2),
    )

    return SampleBatch(
        boundary_coords=boundary_coords,
        boundary_flux=boundary_flux,
        pod_coords=pod_coords,
        probe_coords=probe_coords,
        p_pod=p_pod,
        f_pod=f_pod,
        p_probe=p_probe,
        f_probe=f_probe,
        k_values=k_values,
    )


def normalize_p(p: Array, normalizer: FieldNormalizer) -> Array:
    return (p - normalizer.mean_p) / normalizer.std_p


def denormalize_p(p_norm: Array, normalizer: FieldNormalizer) -> Array:
    return p_norm * normalizer.std_p + normalizer.mean_p


def normalize_f(f: Array, normalizer: FieldNormalizer) -> Array:
    return (f - normalizer.mean_f) / normalizer.std_f


def denormalize_f(f_norm: Array, normalizer: FieldNormalizer) -> Array:
    return f_norm * normalizer.std_f + normalizer.mean_f


def build_field_normalizer_from_batches(
    batches: Sequence[SampleBatch],
    eps: float = 1.0e-6,
) -> FieldNormalizer:
    if not batches:
        raise ValueError("batches must not be empty.")

    p_all = jnp.concatenate([batch.p_pod for batch in batches], axis=0)
    f_all = jnp.concatenate([batch.f_pod for batch in batches], axis=0)

    # P uses scale-only normalization. Keeping mean_p exactly zero is deliberate:
    # the P decoder has a multiplicative outer-boundary mask, so a zero normalized
    # prediction must also correspond to zero physical pressure at r=r_outer.
    mean_p = jnp.asarray(0.0, dtype=p_all.dtype)
    std_p = jnp.maximum(jnp.sqrt(jnp.mean(p_all**2)), eps)

    mean_f = jnp.mean(f_all)
    std_f = jnp.maximum(jnp.std(f_all), eps)
    return FieldNormalizer(mean_p, std_p, mean_f, std_f)


def build_field_normalizer(
    config: PolarAnnulusConfig,
    key: Array,
    num_batches: int | None = None,
    eps: float = 1.0e-6,
) -> FieldNormalizer:
    n_batches = config.normalizer_batches if num_batches is None else num_batches
    if n_batches <= 0:
        raise ValueError("num_batches must be positive.")

    sum_p = jnp.asarray(0.0)
    sum_p2 = jnp.asarray(0.0)
    sum_f = jnp.asarray(0.0)
    sum_f2 = jnp.asarray(0.0)
    count = 0

    for _ in range(n_batches):
        key, subkey = jax.random.split(key)
        batch = sample_batch(subkey, config)
        sum_p = sum_p + jnp.sum(batch.p_pod)
        sum_p2 = sum_p2 + jnp.sum(batch.p_pod**2)
        sum_f = sum_f + jnp.sum(batch.f_pod)
        sum_f2 = sum_f2 + jnp.sum(batch.f_pod**2)
        count += batch.p_pod.size

    # Scale-only normalization for P preserves the exact zero Dirichlet value
    # after decoding and denormalization.
    mean_p = jnp.asarray(0.0, dtype=sum_p.dtype)
    mean_f = sum_f / count
    rms_p2 = jnp.maximum(sum_p2 / count, eps**2)
    var_f = jnp.maximum(sum_f2 / count - mean_f**2, eps**2)
    return FieldNormalizer(
        mean_p=mean_p,
        std_p=jnp.sqrt(rms_p2),
        mean_f=mean_f,
        std_f=jnp.sqrt(var_f),
    )


def build_field_normalizer_online(
    key: Array,
    config: PolarAnnulusConfig,
) -> tuple[FieldNormalizer, Array]:
    """Estimate fixed normalization statistics without retaining batches."""

    p_sum_sq = jnp.asarray(0.0, dtype=jnp.float64)

    f_sum = jnp.asarray(0.0, dtype=jnp.float64)
    f_sum_sq = jnp.asarray(0.0, dtype=jnp.float64)

    p_count = 0
    f_count = 0

    for _ in range(config.normalizer_batches):
        key, subkey = jax.random.split(key)
        batch = sample_batch(subkey, config)

        # 可以同时使用 POD 和 probe 数据估计统计量
        p_values = jnp.concatenate(
            [
                batch.p_pod.reshape(-1),
                batch.p_probe.reshape(-1),
            ],
            axis=0,
        )

        f_values = jnp.concatenate(
            [
                batch.f_pod.reshape(-1),
                batch.f_probe.reshape(-1),
            ],
            axis=0,
        )

        p_values = p_values.astype(jnp.float64)
        f_values = f_values.astype(jnp.float64)

        p_sum_sq = p_sum_sq + jnp.sum(
            jnp.square(p_values)
        )
        p_count += p_values.size

        f_sum = f_sum + jnp.sum(f_values)
        f_sum_sq = f_sum_sq + jnp.sum(
            jnp.square(f_values)
        )
        f_count += f_values.size

    # P 使用 scale-only normalization
    mean_p = jnp.asarray(0.0, dtype=jnp.float32)
    std_p = jnp.sqrt(
        p_sum_sq / float(p_count)
    ).astype(jnp.float32)

    # f 使用 mean/std normalization
    mean_f_64 = f_sum / float(f_count)

    variance_f_64 = (
        f_sum_sq / float(f_count)
        - jnp.square(mean_f_64)
    )

    variance_f_64 = jnp.maximum(
        variance_f_64,
        0.0,
    )

    mean_f = mean_f_64.astype(jnp.float32)
    std_f = jnp.sqrt(
        variance_f_64
    ).astype(jnp.float32)

    eps = jnp.asarray(
        config.normalizer_eps,
        dtype=jnp.float32,
    )

    std_p = jnp.maximum(std_p, eps)
    std_f = jnp.maximum(std_f, eps)

    normalizer = FieldNormalizer(
        mean_p=mean_p,
        std_p=std_p,
        mean_f=mean_f,
        std_f=std_f,
    )

    return normalizer, key


def build_batch_pool(
    config: PolarAnnulusConfig,
    key: Array,
    pool_size: int | None = None,
) -> list[SampleBatch]:
    n_batches = config.pool_size if pool_size is None else pool_size
    if n_batches <= 0:
        raise ValueError("pool_size must be positive.")
    batches = []
    for _ in range(n_batches):
        key, subkey = jax.random.split(key)
        batches.append(sample_batch(subkey, config))
    return batches


def make_source_tokens(latent_f: Array, config: PolarAnnulusConfig) -> Array:
    return latent_f.reshape(
        latent_f.shape[0],
        config.seq_chunks,
        config.seq_chunk_width,
    )


def make_condition_tokens_from_arrays(
    boundary_coords: Array,
    boundary_flux: Array,
    config: PolarAnnulusConfig,
) -> Array:
    """Build periodic boundary tokens.

    Input coordinates remain normalized polar [theta_hat, r_hat], but the token
    exposes [sin(theta), cos(theta), r_hat, g_n] to avoid an angular seam.
    """
    if boundary_coords.ndim != 3 or boundary_coords.shape[-1] != 2:
        raise ValueError("boundary_coords must have shape [B, Nt, 2].")
    if boundary_flux.shape != boundary_coords.shape[:2]:
        raise ValueError("boundary_flux must have shape [B, Nt].")

    theta = theta_from_hat(boundary_coords[..., 0])
    r_hat = boundary_coords[..., 1]
    features = jnp.stack(
        [
            jnp.sin(theta),
            jnp.cos(theta),
            r_hat,
            boundary_flux,
        ],
        axis=-1,
    )
    batch_size, _, feature_dim = features.shape
    return features.reshape(
        batch_size,
        config.cond_chunks,
        config.boundary_chunk_size * feature_dim,
    )


def make_condition_tokens(
    batch: SampleBatch,
    config: PolarAnnulusConfig,
) -> Array:
    return make_condition_tokens_from_arrays(
        batch.boundary_coords,
        batch.boundary_flux,
        config,
    )


def make_target_cosine_boundary(
    config: PolarAnnulusConfig,
    batch_size: int,
) -> tuple[Array, Array]:
    """Return the target condition g_n=-P_r=cos(theta)."""
    coords_single = inner_boundary_coords(config)
    theta = theta_from_hat(coords_single[:, 0])
    flux_single = jnp.cos(theta)
    coords = jnp.broadcast_to(
        coords_single[None, :, :],
        (batch_size, config.theta_size, 2),
    )
    flux = jnp.broadcast_to(
        flux_single[None, :],
        (batch_size, config.theta_size),
    )
    return coords, flux
