from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import qmc

import pickle
import gzip
from pathlib import Path

from config_varboundary import VarBoundaryConfig

Array = jax.Array


class GeometryParams(NamedTuple):
    w1: Array  # [B, 2, H]
    b1: Array  # [B, 1, H]
    w2: Array  # [B, H, 1]


class RawBNNParams(NamedTuple):
    w1: Array  # [B, 2, H]
    b1: Array  # [B, 1, H]
    w2: Array  # [B, H, 1]


class SampleBatch(NamedTuple):
    # Canonical coordinates shared by all geometries.
    pod_coords: Array           # [Npod, 2], canonical Cartesian
    probe_coords: Array         # [Nprobe, 2], canonical Cartesian

    # Physical coordinates after geometry mapping; these vary per sample.
    pod_phys_coords: Array      # [B, Npod, 2]
    probe_phys_coords: Array    # [B, Nprobe, 2]

    # Inner boundary condition tokens. Geometry is embedded through boundary_coords.
    boundary_coords: Array      # [B, Nt, 2], physical inner boundary coordinates
    boundary_flux: Array        # [B, Nt], induced flux from constructed P

    # Fields pulled back to the canonical domain.
    u_pod: Array                # [B, Npod]
    f_pod: Array                # [B, Npod]
    u_probe: Array              # [B, Nprobe]
    f_probe: Array              # [B, Nprobe]
    k_values: Array             # [B]

    # Geometry parameters for physics loss / inference diagnostics.
    geom_params: GeometryParams


class FieldNormalizer(NamedTuple):
    mean_u: Array
    std_u: Array
    mean_f: Array
    std_f: Array


def normalize_u(u: Array, normalizer: FieldNormalizer) -> Array:
    return (u - normalizer.mean_u) / normalizer.std_u


def denormalize_u(u_norm: Array, normalizer: FieldNormalizer) -> Array:
    return u_norm * normalizer.std_u + normalizer.mean_u


def normalize_f(f: Array, normalizer: FieldNormalizer) -> Array:
    return (f - normalizer.mean_f) / normalizer.std_f


def denormalize_f(f_norm: Array, normalizer: FieldNormalizer) -> Array:
    return f_norm * normalizer.std_f + normalizer.mean_f


def make_theta(config: VarBoundaryConfig) -> Array:
    return jnp.linspace(0.0, 2.0 * jnp.pi, config.theta_size, endpoint=False)[:, None]


def make_canonical_grid(config: VarBoundaryConfig) -> Array:
    theta = make_theta(config)[:, 0]
    rho = jnp.linspace(config.canonical_r_inner, config.canonical_r_outer, config.radial_size)
    rr, tt = jnp.meshgrid(rho, theta, indexing="ij")
    x = rr * jnp.cos(tt)
    y = rr * jnp.sin(tt)
    return jnp.stack([x, y], axis=-1).reshape(-1, 2)


def canonical_polar(coords_hat: Array) -> tuple[Array, Array]:
    rho = jnp.linalg.norm(coords_hat, axis=-1)
    theta = jnp.arctan2(coords_hat[..., 1], coords_hat[..., 0])
    return rho, theta


def physical_polar(coords: Array) -> tuple[Array, Array]:
    r = jnp.linalg.norm(coords, axis=-1)
    theta = jnp.arctan2(coords[..., 1], coords[..., 0])
    return r, theta


def sample_canonical_probe_points(key: Array, n_points: int, config: VarBoundaryConfig) -> Array:
    """Sobol points in the canonical annulus with direct rho-uniform sampling.

    This intentionally matches the current annulus convention: rho is sampled
    uniformly in the radial coordinate, not area-uniformly.
    """
    seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    u = jnp.asarray(sampler.random(n_points), dtype=jnp.float32)

    rho = config.canonical_r_inner + u[:, 0] * (
        config.canonical_r_outer - config.canonical_r_inner
    )
    theta = 2.0 * jnp.pi * u[:, 1]
    return jnp.stack([rho * jnp.cos(theta), rho * jnp.sin(theta)], axis=-1)


def sample_geometry_params(key: Array, batch_size: int, config: VarBoundaryConfig) -> GeometryParams:
    """Sample a periodic BNN for a(theta).

    The BNN input is [sin(theta), cos(theta)], ensuring periodicity.
    """
    key_w1, key_b1, key_w2 = jax.random.split(key, 3)
    H = config.hidden_geom_bnn
    w1 = jax.random.normal(key_w1, (batch_size, 2, H)) * config.geom_sigma
    b1 = jax.random.normal(key_b1, (batch_size, 1, H))
    w2 = jax.random.normal(key_w2, (batch_size, H, 1))
    return GeometryParams(w1=w1, b1=b1, w2=w2)


def eval_radius_single(params: GeometryParams, theta: Array, config: VarBoundaryConfig) -> Array:
    """Evaluate physical inner radius a(theta) for one geometry.

    theta can be scalar or array. Return shape follows theta.
    """
    theta = jnp.asarray(theta)
    inp = jnp.stack([jnp.sin(theta), jnp.cos(theta)], axis=-1)  # [..., 2]
    z = jnp.einsum("...i,ih->...h", inp, params.w1) + params.b1[0]
    h = jnp.sqrt(2.0) * jnp.cos(z - jnp.pi / 4.0)
    raw = jnp.einsum("...h,ho->...o", h, params.w2) / jnp.sqrt(h.shape[-1])
    raw = raw[..., 0]
    return config.geom_base + config.geom_amp * jnp.tanh(config.geom_tanh_scale * raw)


def eval_radius(params: GeometryParams, theta: Array, config: VarBoundaryConfig) -> Array:
    return jax.vmap(lambda p: eval_radius_single(p, theta, config))(params)


def radius_theta_derivative_single(params: GeometryParams, theta: Array, config: VarBoundaryConfig) -> Array:
    """Compute da/dtheta for one geometry and vector theta via AD."""
    def scalar_radius(t):
        return eval_radius_single(params, t, config)

    return jax.vmap(jax.grad(scalar_radius))(theta)


def canonical_to_physical_single(coords_hat: Array, geom: GeometryParams, config: VarBoundaryConfig) -> Array:
    rho, theta = canonical_polar(coords_hat)
    a = eval_radius_single(geom, theta, config)
    r = config.outer_scale * rho * a
    return jnp.stack([r * jnp.cos(theta), r * jnp.sin(theta)], axis=-1)


def canonical_to_physical(coords_hat: Array, geom_params: GeometryParams, config: VarBoundaryConfig) -> Array:
    return jax.vmap(lambda g: canonical_to_physical_single(coords_hat, g, config))(geom_params)


def physical_to_canonical_single(coords: Array, geom: GeometryParams, config: VarBoundaryConfig) -> Array:
    """Inverse map for b(theta)=outer_scale*a(theta).

    Since r = outer_scale * rho * a(theta), theta is unchanged.
    """
    r, theta = physical_polar(coords)
    a = eval_radius_single(geom, theta, config)
    rho = r / (config.outer_scale * a)
    return jnp.stack([rho * jnp.cos(theta), rho * jnp.sin(theta)], axis=-1)


def sample_raw_bnn_params(key: Array, sigma_xy: Array, config: VarBoundaryConfig) -> RawBNNParams:
    """Sample raw P_BNN parameters.

    sigma_xy: [B, 2]
    """
    B = sigma_xy.shape[0]
    H = config.hidden_bnn
    key_w1, key_b1, key_w2 = jax.random.split(key, 3)
    w1 = jax.random.normal(key_w1, (B, 2, H)) * sigma_xy[:, :, None]
    b1 = jax.random.normal(key_b1, (B, 1, H)) * sigma_xy.mean(axis=1, keepdims=True)[:, None, :]
    w2 = jax.random.normal(key_w2, (B, H, 1))
    return RawBNNParams(w1=w1, b1=b1, w2=w2)


def raw_bnn_single(params: RawBNNParams, x: Array) -> Array:
    z = jnp.einsum("i,ih->h", x, params.w1) + params.b1[0]
    h = jnp.sqrt(2.0) * jnp.cos(z - jnp.pi / 4.0)
    return (jnp.einsum("h,ho->o", h, params.w2) / jnp.sqrt(h.shape[-1]))[0]


def constructed_solution_single(
    x: Array,
    raw_params: RawBNNParams,
    geom: GeometryParams,
    config: VarBoundaryConfig,
) -> Array:
    """P(x,y) = (r - b(theta)) P_BNN(x,y), b(theta)=outer_scale*a(theta)."""
    r, theta = physical_polar(x)
    a = eval_radius_single(geom, theta, config)
    b = config.outer_scale * a
    return (r - b) * raw_bnn_single(raw_params, x)


def eval_solution_source_single(
    coords_phys: Array,
    raw_params: RawBNNParams,
    geom: GeometryParams,
    k_value: Array,
    config: VarBoundaryConfig,
) -> tuple[Array, Array]:
    """Evaluate P and f=Delta_x P-k^2P on physical points using Cartesian AD."""
    def p_of_x(x):
        return constructed_solution_single(x, raw_params, geom, config)

    u = jax.vmap(p_of_x)(coords_phys)
    hess = jax.vmap(jax.hessian(p_of_x))(coords_phys)
    lap = jnp.trace(hess, axis1=-2, axis2=-1)
    f = lap - (k_value ** 2) * u
    return u, f


def induced_inner_flux_single(
    raw_params: RawBNNParams,
    geom: GeometryParams,
    theta: Array,
    config: VarBoundaryConfig,
) -> tuple[Array, Array]:
    """Induced boundary flux under the same boundary operator as the target problem.

    Problem-description operator:
        dP/dn = grad(P) dot [e_r - (a_dot/a) e_theta]

    This convention gives target cos(alpha)=cos(theta)+(a_dot/a)sin(theta).
    The vector is intentionally not normalized because the problem statement uses
    denominator a(theta), not sqrt(a^2+a_dot^2).
    """
    a = eval_radius_single(geom, theta, config)
    a_dot = radius_theta_derivative_single(geom, theta, config)

    xb = jnp.stack([a * jnp.cos(theta), a * jnp.sin(theta)], axis=-1)

    def p_of_x(x):
        return constructed_solution_single(x, raw_params, geom, config)

    grad_p = jax.vmap(jax.grad(p_of_x))(xb)

    er = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    et = jnp.stack([-jnp.sin(theta), jnp.cos(theta)], axis=-1)
    # normal_operator_vec = er - (a_dot / a)[:, None] * et
    normal_operator_vec = -er + (a_dot / a)[:, None] * et
    flux = jnp.sum(grad_p * normal_operator_vec, axis=-1)
    return xb, flux


def sample_subbatch(
    key: Array,
    config: VarBoundaryConfig,
    pod_coords: Array,
    probe_coords: Array,
) -> SampleBatch:
    """Generate one subbatch using externally supplied canonical coordinates.

    Important:
        pod_coords and probe_coords are shared across all subbatches inside
        sample_batch. This prevents coordinate-value mismatch after concatenation.
    """
    key_sigma, key_k, key_geom, key_raw = jax.random.split(key, 4)
    B = config.sample_size

    geom_params = sample_geometry_params(key_geom, B, config)

    # Same canonical coordinates, different physical coordinates per geometry.
    pod_phys_coords = canonical_to_physical(pod_coords, geom_params, config)
    probe_phys_coords = canonical_to_physical(probe_coords, geom_params, config)

    sigma_choices = jnp.asarray(config.sigma_array)
    sigma_idx = jax.random.randint(key_sigma, (B,), 0, sigma_choices.shape[0])
    sigma_scalar = sigma_choices[sigma_idx]
    sigma_xy = jnp.stack([sigma_scalar, sigma_scalar], axis=-1)

    raw_params = sample_raw_bnn_params(key_raw, sigma_xy, config)

    if config.k_max == config.k_min:
        k_values = jnp.full((B,), config.k_min)
    else:
        k_values = jax.random.uniform(
            key_k,
            (B,),
            minval=config.k_min,
            maxval=config.k_max,
        )

    u_pod, f_pod = jax.vmap(
        eval_solution_source_single,
        in_axes=(0, 0, 0, 0, None),
    )(
        pod_phys_coords,
        raw_params,
        geom_params,
        k_values,
        config,
    )

    u_probe, f_probe = jax.vmap(
        eval_solution_source_single,
        in_axes=(0, 0, 0, 0, None),
    )(
        probe_phys_coords,
        raw_params,
        geom_params,
        k_values,
        config,
    )

    theta = make_theta(config)[:, 0]

    boundary_coords, boundary_flux = jax.vmap(
        induced_inner_flux_single,
        in_axes=(0, 0, None, None),
    )(
        raw_params,
        geom_params,
        theta,
        config,
    )

    return SampleBatch(
        pod_coords=pod_coords,
        probe_coords=probe_coords,
        pod_phys_coords=pod_phys_coords,
        probe_phys_coords=probe_phys_coords,
        boundary_coords=boundary_coords,
        boundary_flux=boundary_flux,
        u_pod=u_pod,
        f_pod=f_pod,
        u_probe=u_probe,
        f_probe=f_probe,
        k_values=k_values,
        geom_params=geom_params,
    )


def sample_batch(key: Array, config: VarBoundaryConfig) -> SampleBatch:
    """Generate an effective batch of num_repeats * sample_size.

    Canonical pod/probe coordinates are generated once and shared by all
    subbatches. This is required because FE.reconstruct assumes a common
    coordinate set:
        latent: [B, p]
        coords: [N, 2]
        output: [B, N]

    Physical coordinates are still sample-dependent because each geometry
    maps the same canonical coordinates to a different physical domain.
    """
    key_coords, key_loop = jax.random.split(key)

    key_pod, key_probe = jax.random.split(key_coords)

    # Fixed canonical grid for branch input / trunk reconstruction.
    # This is deterministic, so key_pod is not actually needed here.
    pod_coords = make_canonical_grid(config)

    # One shared canonical probe set for the whole effective batch.
    probe_coords = sample_canonical_probe_points(
        key_probe,
        config.random_probe_points,
        config,
    )

    samples = []
    for _ in range(config.num_repeats):
        key_loop, subkey = jax.random.split(key_loop)
        samples.append(
            sample_subbatch(
                subkey,
                config,
                pod_coords=pod_coords,
                probe_coords=probe_coords,
            )
        )

    return SampleBatch(
        # Shared canonical coordinates: do not concatenate.
        pod_coords=pod_coords,
        probe_coords=probe_coords,

        # Sample-dependent physical coordinates: concatenate over batch.
        pod_phys_coords=jnp.concatenate(
            [s.pod_phys_coords for s in samples],
            axis=0,
        ),
        probe_phys_coords=jnp.concatenate(
            [s.probe_phys_coords for s in samples],
            axis=0,
        ),

        # Boundary tokens.
        boundary_coords=jnp.concatenate(
            [s.boundary_coords for s in samples],
            axis=0,
        ),
        boundary_flux=jnp.concatenate(
            [s.boundary_flux for s in samples],
            axis=0,
        ),

        # Fields.
        u_pod=jnp.concatenate(
            [s.u_pod for s in samples],
            axis=0,
        ),
        f_pod=jnp.concatenate(
            [s.f_pod for s in samples],
            axis=0,
        ),
        u_probe=jnp.concatenate(
            [s.u_probe for s in samples],
            axis=0,
        ),
        f_probe=jnp.concatenate(
            [s.f_probe for s in samples],
            axis=0,
        ),

        # PDE parameter.
        k_values=jnp.concatenate(
            [s.k_values for s in samples],
            axis=0,
        ),

        # Geometry BNN parameters.
        geom_params=GeometryParams(
            w1=jnp.concatenate([s.geom_params.w1 for s in samples], axis=0),
            b1=jnp.concatenate([s.geom_params.b1 for s in samples], axis=0),
            w2=jnp.concatenate([s.geom_params.w2 for s in samples], axis=0),
        ),
    )


def to_cpu_batch(batch: SampleBatch) -> SampleBatch:
    """Move one SampleBatch from device to CPU numpy arrays.

    The training pool should live in host memory, not GPU memory.
    """
    return jax.tree_util.tree_map(
        lambda x: np.asarray(jax.device_get(x)),
        batch,
    )


def build_batch_pool(
    config: VarBoundaryConfig,
    key: Array,
    pool_size: int | None = None,
) -> list[SampleBatch]:
    """Pre-generate a pool of SampleBatch objects on CPU memory.

    Each element in the pool has the same static shapes, so selecting any
    batch will not trigger JAX recompilation inside fe_train_step.

    Important:
        The pool is intentionally stored as CPU numpy arrays.
        Do not jax.device_put the whole pool.
    """
    if pool_size is None:
        pool_size = config.fe_pool_size

    pool = []
    for i in range(pool_size):
        key, subkey = jax.random.split(key)
        batch = sample_batch(subkey, config)

        # Force data generation to finish and move the result to CPU memory.
        batch_cpu = to_cpu_batch(batch)
        pool.append(batch_cpu)

        if (i + 1) % 10 == 0:
            print(f"[Pool] generated {i + 1}/{pool_size} batches")

    return pool


def build_field_normalizer_from_pool(
    pool: list[SampleBatch],
    eps: float = 1e-6,
) -> FieldNormalizer:
    """Compute global scalar normalizer from a pre-generated training pool.

    This uses u_pod and f_pod from the exact same data distribution used in FE
    training. The computation is done on CPU to avoid GPU memory pressure.
    """
    sum_u = 0.0
    sum_u2 = 0.0
    sum_f = 0.0
    sum_f2 = 0.0
    count = 0

    for batch in pool:
        u = np.asarray(batch.u_pod, dtype=np.float64)
        f = np.asarray(batch.f_pod, dtype=np.float64)

        sum_u += np.sum(u)
        sum_u2 += np.sum(u ** 2)

        sum_f += np.sum(f)
        sum_f2 += np.sum(f ** 2)

        count += u.size

    mean_u = sum_u / count
    mean_f = sum_f / count

    var_u = max(sum_u2 / count - mean_u ** 2, eps ** 2)
    var_f = max(sum_f2 / count - mean_f ** 2, eps ** 2)

    std_u = np.sqrt(var_u)
    std_f = np.sqrt(var_f)

    return FieldNormalizer(
        mean_u=jnp.asarray(mean_u, dtype=jnp.float32),
        std_u=jnp.asarray(std_u, dtype=jnp.float32),
        mean_f=jnp.asarray(mean_f, dtype=jnp.float32),
        std_f=jnp.asarray(std_f, dtype=jnp.float32),
    )

def build_field_normalizer(
    config: VarBoundaryConfig,
    key: Array,
    num_batches: int | None = None,
    eps: float = 1e-6,
) -> FieldNormalizer:
    if num_batches is None:
        num_batches = config.normalizer_batches

    sum_u = 0.0
    sum_u2 = 0.0
    sum_f = 0.0
    sum_f2 = 0.0
    count = 0
    for _ in range(num_batches):
        key, subkey = jax.random.split(key)
        batch = sample_batch(subkey, config)
        u = batch.u_pod
        f = batch.f_pod
        sum_u += jnp.sum(u)
        sum_u2 += jnp.sum(u**2)
        sum_f += jnp.sum(f)
        sum_f2 += jnp.sum(f**2)
        count += u.size

    mean_u = sum_u / count
    mean_f = sum_f / count
    std_u = jnp.sqrt(jnp.maximum(sum_u2 / count - mean_u**2, eps**2))
    std_f = jnp.sqrt(jnp.maximum(sum_f2 / count - mean_f**2, eps**2))
    return FieldNormalizer(mean_u=mean_u, std_u=std_u, mean_f=mean_f, std_f=std_f)


def save_batch_pool(
    pool: list[SampleBatch],
    path: str | Path,
) -> Path:
    """Save pre-generated FE training pool to disk.

    The pool should already be on CPU memory, typically generated by
    build_batch_pool(...), whose elements are numpy arrays.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(path, "wb") as f:
        pickle.dump(pool, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"[Pool] saved {len(pool)} batches to: {path}")
    return path


def load_batch_pool(
    path: str | Path,
) -> list[SampleBatch]:
    """Load pre-generated FE training pool from disk."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Cannot find pool file: {path}")

    with gzip.open(path, "rb") as f:
        pool = pickle.load(f)

    print(f"[Pool] loaded {len(pool)} batches from: {path}")
    return pool


def make_source_tokens(latent_f: Array, config: VarBoundaryConfig) -> Array:
    return latent_f.reshape(latent_f.shape[0], config.seq_chunks, config.seq_chunk_width)


def make_condition_tokens(batch: SampleBatch, config: VarBoundaryConfig) -> Array:
    bc = jnp.concatenate([batch.boundary_coords, batch.boundary_flux[..., None]], axis=-1)
    return bc.reshape(bc.shape[0], config.cond_chunks, config.cond_chunk_width)


def target_boundary_flux_from_problem(geom_params: GeometryParams, config: VarBoundaryConfig) -> tuple[Array, Array]:
    """For inference: g_target = cos(theta) + a_dot/a * sin(theta)."""
    theta = make_theta(config)[:, 0]

    def one_geom(g):
        a = eval_radius_single(g, theta, config)
        a_dot = radius_theta_derivative_single(g, theta, config)
        xb = jnp.stack([a * jnp.cos(theta), a * jnp.sin(theta)], axis=-1)
        g_target = jnp.cos(theta) + (a_dot / a) * jnp.sin(theta)
        return xb, g_target

    return jax.vmap(one_geom)(geom_params)
