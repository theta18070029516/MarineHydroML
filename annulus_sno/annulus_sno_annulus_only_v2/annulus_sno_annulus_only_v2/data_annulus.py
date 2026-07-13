from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from scipy.stats import qmc

from config import AnnulusConfig


Array = jax.Array


class SampleBatch(NamedTuple):
    boundary_coords: Array    # [B, Nt, 2]
    boundary_flux: Array      # [B, Nt]
    pod_coords: Array         # [Npod, 2]
    probe_coords: Array       # [Nprobe, 2]
    u_pod: Array              # [B, Npod]
    f_pod: Array              # [B, Npod]
    u_probe: Array            # [B, Nprobe]
    f_probe: Array            # [B, Nprobe]
    k_values: Array           # [B]


class PCAStats(NamedTuple):
    mean_u: Array
    modes_u: Array
    eigvals_u: Array
    mean_f: Array
    modes_f: Array
    eigvals_f: Array

class FieldNormalizer(NamedTuple):
    mean_u: Array
    std_u: Array
    mean_f: Array
    std_f: Array

def build_field_normalizer(
    config: AnnulusConfig,
    key: Array,
    num_batches: int | None = None,
    eps: float = 1e-6,
) -> FieldNormalizer:
    """Build global scalar mean/std for u and f from samples on the regular pod grid.

    The statistics are computed over:
        batch dimension × sample dimension × regular-grid points.

    This function must use the same sample_batch implementation as training.
    In particular, u_pod/f_pod and u_probe/f_probe must correspond to the same
    BNN realization when used inside sample_batch.
    """
    if num_batches is None:
        num_batches = config.pod_snapshots

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

        sum_u = sum_u + jnp.sum(u)
        sum_u2 = sum_u2 + jnp.sum(u ** 2)
        sum_f = sum_f + jnp.sum(f)
        sum_f2 = sum_f2 + jnp.sum(f ** 2)
        count = count + u.size

    mean_u = sum_u / count
    mean_f = sum_f / count

    var_u = sum_u2 / count - mean_u ** 2
    var_f = sum_f2 / count - mean_f ** 2

    std_u = jnp.sqrt(jnp.maximum(var_u, eps ** 2))
    std_f = jnp.sqrt(jnp.maximum(var_f, eps ** 2))

    return FieldNormalizer(
        mean_u=mean_u,
        std_u=std_u,
        mean_f=mean_f,
        std_f=std_f,
    )

def normalize_u(u: Array, normalizer: FieldNormalizer) -> Array:
    return (u - normalizer.mean_u) / normalizer.std_u


def denormalize_u(u_norm: Array, normalizer: FieldNormalizer) -> Array:
    return u_norm * normalizer.std_u + normalizer.mean_u


def normalize_f(f: Array, normalizer: FieldNormalizer) -> Array:
    return (f - normalizer.mean_f) / normalizer.std_f


def denormalize_f(f_norm: Array, normalizer: FieldNormalizer) -> Array:
    return f_norm * normalizer.std_f + normalizer.mean_f


def make_theta(config: AnnulusConfig) -> Array:
    return jnp.linspace(0.0, 2.0 * jnp.pi, config.theta_size, endpoint=False)[:, None]


def make_annulus_grid(config: AnnulusConfig) -> Array:
    theta = make_theta(config)[:, 0]
    radial = jnp.linspace(config.r_inner, config.r_outer, config.radial_size)
    rr, tt = jnp.meshgrid(radial, theta, indexing='ij')
    x = rr * jnp.cos(tt)
    y = rr * jnp.sin(tt)
    return jnp.stack([x, y], axis=-1).reshape(-1, 2)


# def sample_uniform_annulus(key: Array, n_points: int, r_inner: float, r_outer: float) -> Array:
#     key_r, key_t = jax.random.split(key)
#     radius2 = jax.random.uniform(key_r, (n_points, 1), minval=r_inner ** 2, maxval=r_outer ** 2)
#     radius = jnp.sqrt(radius2)
#     theta = jax.random.uniform(key_t, (n_points, 1), minval=0.0, maxval=2.0 * jnp.pi)
 
#     return jnp.concatenate([radius * jnp.cos(theta), radius * jnp.sin(theta)], axis=-1)

def sobol_annulus_points(key: Array, n_points: int, r_inner: float, r_outer: float, dim: int) -> Array:
    seed = int(jax.random.randint(key, (), 0, 2**31 - 1)) #int(jax.random.randint(key, (), 0, 2**31 - 1)) #0
    sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
    u = sampler.random(n_points)   # shape [n_points, 2]

    u1 = u[:, 0:1]
    u2 = u[:, 1:2]

    r = r_inner + u1 * (r_outer - r_inner)
    r = jnp.asarray(r)
    # r = jnp.sqrt(jnp.asarray(r2))
    theta = 2.0 * jnp.pi * jnp.asarray(u2)

    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    return jnp.concatenate([x, y], axis=-1)


def inner_boundary_coords(config: AnnulusConfig) -> Array:
    theta = make_theta(config)
    return jnp.concatenate([config.r_inner * jnp.cos(theta), config.r_inner * jnp.sin(theta)], axis=-1)


def inner_boundary_flux(config: AnnulusConfig) -> Array:
    theta = make_theta(config)[:, 0]
    return jnp.cos(theta)



def single_sample_forward_second_order_sin(w1: Array, b1: Array, w2: Array, x: Array) -> tuple[Array, Array, Array]:
    z = jnp.einsum('ih,ni->nh', w1, x) + b1
    sin_term = jnp.sin(z - jnp.pi / 4.0)
    cos_term = jnp.cos(z - jnp.pi / 4.0)
    hidden = jnp.sqrt(2.0) * cos_term

    u = jnp.einsum('ho,nh->no', w2, hidden) / jnp.sqrt(hidden.shape[-1])
    dh_dx = -jnp.sqrt(2.0) * jnp.einsum('nh,ih->nhi', sin_term, w1)
    du = jnp.einsum('nhi,ho->noi', dh_dx, w2) / jnp.sqrt(hidden.shape[-1])
    d2u = -jnp.sqrt(2.0) * jnp.einsum('nh,ih,jh,ho->noij', cos_term, w1, w1, w2)
    d2u = d2u / jnp.sqrt(hidden.shape[-1])
    return u, du, d2u


def bnn_sin(key: Array, sigma: Array, x: Array, sample_size: int, hidden_layers: int) -> tuple[Array, Array, Array]:
    key_w1, key_b1, key_w2 = jax.random.split(key, 3)
    in_dim = x.shape[-1]
    out_dim = 1

    sigma = jnp.asarray(sigma)
    if sigma.ndim == 0:
        sigma = jnp.full((sample_size, in_dim), sigma)
    elif sigma.ndim == 1:
        sigma = jnp.broadcast_to(sigma[None, :], (sample_size, in_dim))
    elif sigma.ndim != 2:
        raise ValueError(f'Unsupported sigma shape: {sigma.shape}')

    w1 = jax.random.normal(key_w1, (sample_size, in_dim, hidden_layers)) * sigma[:, :, None]
    b1 = jax.random.normal(key_b1, (sample_size, 1, hidden_layers)) * sigma.mean(axis=1, keepdims=True)[:, None, :]
    w2 = jax.random.normal(key_w2, (sample_size, hidden_layers, out_dim))

    if x.ndim == 2:
        u, du, ddu = jax.vmap(single_sample_forward_second_order_sin, in_axes=(0, 0, 0, None))(w1, b1, w2, x)
    elif x.ndim == 3:
        u, du, ddu = jax.vmap(single_sample_forward_second_order_sin, in_axes=(0, 0, 0, 0))(w1, b1, w2, x)
    else:
        raise ValueError(f'Unsupported x shape: {x.shape}')

    lap = jnp.trace(ddu, axis1=-2, axis2=-1)
    return u[..., 0], du[..., 0, :], lap[..., 0]


def radial_theta(coords: Array) -> tuple[Array, Array]:
    r = jnp.linalg.norm(coords, axis=-1)
    theta = jnp.arctan2(coords[..., 1], coords[..., 0])
    return r, theta


def lifting_term(coords: Array, config: AnnulusConfig) -> Array:
    r, theta = radial_theta(coords)
    return -(r - config.r_outer) * jnp.cos(theta)


def lifting_laplacian(coords: Array, config: AnnulusConfig) -> Array:
    r, theta = radial_theta(coords)
    # For L(r,theta) = (R-r) cos(theta), ΔL = -(R/r^2) cos(theta)
    return -(config.r_outer / (r ** 2)) * jnp.cos(theta)


def mask_term(coords: Array, config: AnnulusConfig) -> tuple[Array, Array, Array]:
    r, _ = radial_theta(coords)
    a = config.r_inner
    R = config.r_outer
    mask = (r - R) * (r - a) ** 2
    dmask = 2 * R * a - 2 * R * r + a ** 2 - 4 * a * r + 3 * r ** 2
    d2mask = -2 * R - 4 * a + 6 * r
    lap_mask = d2mask + dmask / r
    return mask, dmask, lap_mask


def hard_bc_solution_and_source(raw_u: Array, grad_raw_u: Array, lap_raw_u: Array, coords: Array, k_values: Array, config: AnnulusConfig) -> tuple[Array, Array]:
    r, _ = radial_theta(coords)
    lift = lifting_term(coords, config)
    lap_lift = lifting_laplacian(coords, config)
    mask, dmask, lap_mask = mask_term(coords, config)

    radial_grad_raw_u = (coords[..., 0] * grad_raw_u[..., 0] + coords[..., 1] * grad_raw_u[..., 1]) / r
    u = lift + mask * raw_u
    lap_u = lap_lift + lap_mask * raw_u + 2.0 * dmask * radial_grad_raw_u + mask * lap_raw_u
    f = lap_u - (k_values[:, None] ** 2) * u
    return u, f

def outer_dirichlet_solution_and_source(
    raw_u: Array,
    grad_raw_u: Array,
    lap_raw_u: Array,
    coords: Array,
    k_values: Array,
    config: AnnulusConfig,
) -> tuple[Array, Array]:
    """
    Construct solution with only outer Dirichlet boundary enforced:

        u = (r - r_outer) * raw_u

    Then compute:

        f = Δu - k^2 u

    Shapes:
        raw_u:      [B, N]
        grad_raw_u: [B, N, 2]
        lap_raw_u:  [B, N]
        coords:     [B, N, 2]
        k_values:   [B]
    """
    r, _ = radial_theta(coords)

    m = r - config.r_outer

    radial_grad_raw_u = (
        coords[..., 0] * grad_raw_u[..., 0]
        + coords[..., 1] * grad_raw_u[..., 1]
    ) / r

    u = m * raw_u

    lap_u = (
        m * lap_raw_u
        + 2.0 * radial_grad_raw_u
        + raw_u / r
    )

    f = lap_u - (k_values[:, None] ** 2) * u

    return u, f

def outer_dirichlet_inner_flux(
    raw_u_bnd: Array,
    grad_raw_u_bnd: Array,
    boundary_coords: Array,
    config: AnnulusConfig,
) -> Array:
    """
    Compute inner Neumann data:

        g = ∂u/∂n

    for:

        u = (r - r_outer) raw_u

    on the inner boundary.

    Since n = -e_r on the inner boundary,

        g = -∂_r u
          = -[raw_u + (r_inner - r_outer) ∂_r raw_u]
    """
    r, _ = radial_theta(boundary_coords)

    radial_grad_raw_u = (
        boundary_coords[..., 0] * grad_raw_u_bnd[..., 0]
        + boundary_coords[..., 1] * grad_raw_u_bnd[..., 1]
    ) / r

    flux = -(
        raw_u_bnd
        + (config.r_inner - config.r_outer) * radial_grad_raw_u
    )

    return flux


# def sample_batch(key: Array, config: AnnulusConfig) -> SampleBatch:
#     key_sigma, key_k, key_probe, key_pod_field, key_probe_field = jax.random.split(key, 5)

#     pod_coords = make_annulus_grid(config)
#     probe_coords = sobol_annulus_points(key_probe, config.random_probe_points, config.r_inner, config.r_outer, config.dim)

#     pod_coords_b = jnp.broadcast_to(pod_coords[None, :, :], (config.sample_size, pod_coords.shape[0], 2))
#     probe_coords_b = jnp.broadcast_to(probe_coords[None, :, :], (config.sample_size, probe_coords.shape[0], 2))

#     sigma_choices = jnp.asarray(config.sigma_array)
#     sigma_idx = jax.random.randint(key_sigma, (config.sample_size,), 0, sigma_choices.shape[0])
#     sigma_scalar = sigma_choices[sigma_idx]
#     sigma_xy = jnp.stack([sigma_scalar, sigma_scalar], axis=-1)
#     k_values = jax.random.uniform(key_k, (config.sample_size,), minval=config.k_min, maxval=config.k_max)

#     raw_u_pod, grad_u_pod, lap_u_pod = bnn_sin(key_pod_field, sigma_xy, pod_coords_b, config.sample_size, config.hidden_bnn)
#     raw_u_probe, grad_u_probe, lap_u_probe = bnn_sin(key_pod_field, sigma_xy, probe_coords_b, config.sample_size, config.hidden_bnn)

#     u_pod, f_pod = hard_bc_solution_and_source(raw_u_pod, grad_u_pod, lap_u_pod, pod_coords_b, k_values, config)
#     u_probe, f_probe = hard_bc_solution_and_source(raw_u_probe, grad_u_probe, lap_u_probe, probe_coords_b, k_values, config)

#     boundary_coords = jnp.broadcast_to(inner_boundary_coords(config)[None, :, :], (config.sample_size, config.theta_size, 2))
#     boundary_flux = jnp.broadcast_to(inner_boundary_flux(config)[None, :], (config.sample_size, config.theta_size))

#     return SampleBatch(
#         boundary_coords=boundary_coords,
#         boundary_flux=boundary_flux,
#         pod_coords=pod_coords,
#         probe_coords=probe_coords,
#         u_pod=u_pod,
#         f_pod=f_pod,
#         u_probe=u_probe,
#         f_probe=f_probe,
#         k_values=k_values,
#     )

def sample_batch(key: Array, config: AnnulusConfig) -> SampleBatch:
    """
    Generate a SampleBatch with effective batch size:

        B_eff = num_repeats * config.sample_size

    Here config.sample_size is the size of each sub-batch.
    The PI-sampler is called num_repeats times and the resulting data are
    concatenated along the batch dimension.

    Output shapes:
        u_pod:    [3 * config.sample_size, Npod]
        f_pod:    [3 * config.sample_size, Npod]
        u_probe:  [3 * config.sample_size, Nprobe]
        f_probe:  [3 * config.sample_size, Nprobe]
        k_values: [3 * config.sample_size]
    """
    num_repeats = config.num_repeats
    sub_batch_size = config.sample_size
    total_batch_size = num_repeats * sub_batch_size

    key_sigma, key_k, key_probe, key_field = jax.random.split(key, 4)

    # ------------------------------------------------------------
    # 1. Coordinates are generated once and shared by all sub-batches.
    #    This keeps SampleBatch.probe_coords shape as [Nprobe, 2].
    # ------------------------------------------------------------
    pod_coords = make_annulus_grid(config)

    probe_coords = sobol_annulus_points(
        key_probe,
        config.random_probe_points,
        config.r_inner,
        config.r_outer,
        config.dim,
    )

    # ------------------------------------------------------------
    # 2. Split keys for each sub-batch.
    # ------------------------------------------------------------
    key_sigma_list = jax.random.split(key_sigma, num_repeats)
    key_k_list = jax.random.split(key_k, num_repeats)
    key_field_list = jax.random.split(key_field, num_repeats)

    sigma_choices = jnp.asarray(config.sigma_array)

    u_pod_list = []
    f_pod_list = []
    u_probe_list = []
    f_probe_list = []
    k_values_list = []
    boundary_flux_list = []

    # ------------------------------------------------------------
    # 3. Generate each sub-batch independently.
    # ------------------------------------------------------------
    for i in range(num_repeats):
        key_sigma_i = key_sigma_list[i]
        key_k_i = key_k_list[i]
        key_field_i = key_field_list[i]

        pod_coords_b = jnp.broadcast_to(
            pod_coords[None, :, :],
            (sub_batch_size, pod_coords.shape[0], 2),
        )

        probe_coords_b = jnp.broadcast_to(
            probe_coords[None, :, :],
            (sub_batch_size, probe_coords.shape[0], 2),
        )

        sigma_idx = jax.random.randint(
            key_sigma_i,
            (sub_batch_size,),
            0,
            sigma_choices.shape[0],
        )

        sigma_scalar = sigma_choices[sigma_idx]
        sigma_xy = jnp.stack([sigma_scalar, sigma_scalar], axis=-1)

        k_values_i = jax.random.uniform(
            key_k_i,
            (sub_batch_size,),
            minval=config.k_min,
            maxval=config.k_max,
        )

        # --------------------------------------------------------
        # Critical:
        # pod and probe must use the same key_field_i.
        # This guarantees they are evaluations of the same BNN
        # realization at different coordinates.
        # --------------------------------------------------------
        raw_u_pod, grad_u_pod, lap_u_pod = bnn_sin(
            key_field_i,
            sigma_xy,
            pod_coords_b,
            sub_batch_size,
            config.hidden_bnn,
        )

        raw_u_probe, grad_u_probe, lap_u_probe = bnn_sin(
            key_field_i,
            sigma_xy,
            probe_coords_b,
            sub_batch_size,
            config.hidden_bnn,
        )

        # u_pod_i, f_pod_i = hard_bc_solution_and_source(
        #     raw_u_pod,
        #     grad_u_pod,
        #     lap_u_pod,
        #     pod_coords_b,
        #     k_values_i,
        #     config,
        # )

        # u_probe_i, f_probe_i = hard_bc_solution_and_source(
        #     raw_u_probe,
        #     grad_u_probe,
        #     lap_u_probe,
        #     probe_coords_b,
        #     k_values_i,
        #     config,
        # )

        u_pod_i, f_pod_i = outer_dirichlet_solution_and_source(
            raw_u_pod,
            grad_u_pod,
            lap_u_pod,
            pod_coords_b,
            k_values_i,
            config,
        )
        
        u_probe_i, f_probe_i = outer_dirichlet_solution_and_source(
            raw_u_probe,
            grad_u_probe,
            lap_u_probe,
            probe_coords_b,
            k_values_i,
            config,
        )

        boundary_coords_single = inner_boundary_coords(config)
        boundary_coords_b = jnp.broadcast_to(
            boundary_coords_single[None, :, :],
            (sub_batch_size, config.theta_size, 2),
        )
        
        raw_u_bnd, grad_u_bnd, _ = bnn_sin(
            key_field_i,
            sigma_xy,
            boundary_coords_b,
            sub_batch_size,
            config.hidden_bnn,
        )
        
        boundary_flux_i = outer_dirichlet_inner_flux(
            raw_u_bnd,
            grad_u_bnd,
            boundary_coords_b,
            config,
        )

        u_pod_list.append(u_pod_i)
        f_pod_list.append(f_pod_i)
        u_probe_list.append(u_probe_i)
        f_probe_list.append(f_probe_i)
        k_values_list.append(k_values_i)
        boundary_flux_list.append(boundary_flux_i)

    # ------------------------------------------------------------
    # 4. Concatenate along batch dimension.
    # ------------------------------------------------------------
    u_pod = jnp.concatenate(u_pod_list, axis=0)
    f_pod = jnp.concatenate(f_pod_list, axis=0)
    u_probe = jnp.concatenate(u_probe_list, axis=0)
    f_probe = jnp.concatenate(f_probe_list, axis=0)
    k_values = jnp.concatenate(k_values_list, axis=0)
    boundary_flux = jnp.concatenate(boundary_flux_list, axis=0)

    # ------------------------------------------------------------
    # 5. Boundary information.
    #    Shape must match the effective batch size.
    # ------------------------------------------------------------
    boundary_coords = jnp.broadcast_to(
        inner_boundary_coords(config)[None, :, :],
        (total_batch_size, config.theta_size, 2),
    )

    # boundary_flux = jnp.broadcast_to(
    #     inner_boundary_flux(config)[None, :],
    #     (total_batch_size, config.theta_size),
    # )

    return SampleBatch(
        boundary_coords=boundary_coords,
        boundary_flux=boundary_flux,
        pod_coords=pod_coords,
        probe_coords=probe_coords,
        u_pod=u_pod,
        f_pod=f_pod,
        u_probe=u_probe,
        f_probe=f_probe,
        k_values=k_values,
    )

def build_batch_pool_list(
    config: AnnulusConfig,
    key: jax.Array,
    pool_size: int = 100,
):
    batches = []
    for _ in range(pool_size):
        key, subkey = jax.random.split(key)
        batches.append(sample_batch(subkey, config))
    return batches


def build_pca_basis_from_snapshots(snapshot_matrix: Array, n_basis: int) -> tuple[Array, Array, Array]:
    # snapshot_matrix: [Nsamples, Npoints]
    mean_vec = jnp.mean(snapshot_matrix, axis=0)
    centered = snapshot_matrix - mean_vec[None, :]
    U, S, _ = jnp.linalg.svd(centered.T, full_matrices=False)
    modes = U[:, :n_basis]
    eigvals = (S ** 2) / max(snapshot_matrix.shape[0] - 1, 1)
    eigvals = eigvals[:n_basis]
    # plt.plot(eigvals) #eigvals[:config.N_basis]
    return mean_vec, modes, eigvals


def project_with_pca(field_values: Array, mean_vec: Array, modes: Array) -> Array:
    return (field_values - mean_vec[None, :]) @ modes


def reconstruct_with_pca(coeffs: Array, mean_vec: Array, modes: Array) -> Array:
    return mean_vec[None, :] + coeffs @ modes.T


def build_pca_stats(config: AnnulusConfig, key: Array) -> PCAStats:
    u_snapshots = []
    f_snapshots = []
    for _ in range(config.pod_snapshots):
        key, subkey = jax.random.split(key)
        batch = sample_batch(subkey, config)
        u_snapshots.append(batch.u_pod)
        f_snapshots.append(batch.f_pod)

    u_matrix = jnp.concatenate(u_snapshots, axis=0)
    f_matrix = jnp.concatenate(f_snapshots, axis=0)

    mean_u, modes_u, eigvals_u = build_pca_basis_from_snapshots(u_matrix, config.n_basis)
    mean_f, modes_f, eigvals_f = build_pca_basis_from_snapshots(f_matrix, config.n_basis)
    return PCAStats(mean_u, modes_u, eigvals_u, mean_f, modes_f, eigvals_f)


def make_source_tokens(latent_f: Array, config: AnnulusConfig) -> Array:
    return latent_f.reshape(latent_f.shape[0], config.seq_chunks, config.seq_chunk_width)


def make_condition_tokens(batch: SampleBatch, config: AnnulusConfig) -> Array:
    cond = jnp.concatenate([batch.boundary_coords, batch.boundary_flux[..., None]], axis=-1)
    B, Nt, D = cond.shape
    chunk = config.boundary_chunk_size
    cond = cond.reshape(B, config.cond_chunks, chunk * D)
    return cond
