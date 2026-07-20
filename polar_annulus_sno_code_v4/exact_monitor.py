from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from config_polar import PolarAnnulusConfig
from data_polar import (
    FieldNormalizer,
    denormalize_f,
    denormalize_p,
    inner_boundary_coords,
    make_condition_tokens_from_arrays,
    make_polar_grid,
    make_source_tokens,
    normalize_f,
    normalize_p,
    outer_boundary_coords,
    r_from_hat,
    r_to_hat,
    theta_from_hat,
    theta_to_hat,
)
from exact_solution import exact_annulus_fourier_solution
from models_polar import FunctionEncoder


Array = jax.Array


class ExactBenchmark(NamedTuple):
    pod_coords: Array
    eval_coords: Array
    eval_radius: Array
    eval_theta: Array
    area_weights: Array
    p_pod: Array
    f_pod: Array
    p_eval: Array
    boundary_coords: Array
    boundary_flux: Array
    k_values: Array


def make_exact_benchmark(config: PolarAnnulusConfig) -> ExactBenchmark:
    """Build the fixed analytic monitor without using training samples."""
    pod_coords = make_polar_grid(config)
    pod_theta = theta_from_hat(pod_coords[:, 0])
    pod_radius = r_from_hat(pod_coords[:, 1], config)
    p_pod = exact_annulus_fourier_solution(
        np.asarray(pod_radius),
        np.asarray(pod_theta),
        config.exact_eval_k,
        mode=config.exact_eval_mode,
        phase=config.exact_eval_phase,
        amplitude=config.exact_eval_amplitude,
        r_inner=config.r_inner,
        r_outer=config.r_outer,
    )

    radial_edges = np.linspace(
        config.r_inner,
        config.r_outer,
        config.exact_eval_radial_size + 1,
        dtype=np.float64,
    )
    theta_edges = np.linspace(
        0.0,
        2.0 * np.pi,
        config.exact_eval_theta_size + 1,
        dtype=np.float64,
    )
    radial_centers = 0.5 * (radial_edges[:-1] + radial_edges[1:])
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    rr, tt = np.meshgrid(radial_centers, theta_centers, indexing="ij")
    eval_coords = np.stack(
        [
            np.asarray(theta_to_hat(jnp.asarray(tt))),
            np.asarray(r_to_hat(jnp.asarray(rr), config)),
        ],
        axis=-1,
    ).reshape(-1, 2)
    p_eval = exact_annulus_fourier_solution(
        rr,
        tt,
        config.exact_eval_k,
        mode=config.exact_eval_mode,
        phase=config.exact_eval_phase,
        amplitude=config.exact_eval_amplitude,
        r_inner=config.r_inner,
        r_outer=config.r_outer,
    )

    boundary_coords_single = inner_boundary_coords(config)
    boundary_theta = theta_from_hat(boundary_coords_single[:, 0])
    boundary_flux_single = config.exact_eval_amplitude * jnp.cos(
        config.exact_eval_mode * boundary_theta + config.exact_eval_phase
    )

    return ExactBenchmark(
        pod_coords=pod_coords,
        eval_coords=jnp.asarray(eval_coords, dtype=jnp.float32),
        eval_radius=jnp.asarray(rr, dtype=jnp.float32),
        eval_theta=jnp.asarray(tt, dtype=jnp.float32),
        area_weights=jnp.asarray(rr.reshape(1, -1), dtype=jnp.float32),
        p_pod=jnp.asarray(p_pod.reshape(1, -1), dtype=jnp.float32),
        f_pod=jnp.zeros((1, config.n_pod), dtype=jnp.float32),
        p_eval=jnp.asarray(p_eval.reshape(1, -1), dtype=jnp.float32),
        boundary_coords=jnp.broadcast_to(
            boundary_coords_single[None, :, :],
            (1, config.theta_size, 2),
        ),
        boundary_flux=boundary_flux_single[None, :],
        k_values=jnp.asarray([config.exact_eval_k], dtype=jnp.float32),
    )


def _field_metrics(
    pred: Array,
    ref: Array,
    area_weights: Array,
) -> dict[str, float]:
    error = pred - ref
    eps = jnp.finfo(pred.dtype).tiny
    grid_relative_l2 = jnp.linalg.norm(error) / jnp.maximum(
        jnp.linalg.norm(ref), eps
    )
    area_relative_l2 = jnp.sqrt(
        jnp.sum(area_weights * error**2)
        / jnp.maximum(jnp.sum(area_weights * ref**2), eps)
    )
    rmse = jnp.sqrt(jnp.mean(error**2))
    max_abs = jnp.max(jnp.abs(error))
    relative_linf = max_abs / jnp.maximum(jnp.max(jnp.abs(ref)), eps)
    return {
        "p_grid_relative_l2": float(grid_relative_l2),
        "p_area_relative_l2": float(area_relative_l2),
        "p_rmse": float(rmse),
        "p_relative_linf": float(relative_linf),
    }


def evaluate_fe_exact(
    fe_state,
    normalizer: FieldNormalizer,
    benchmark: ExactBenchmark,
    config: PolarAnnulusConfig,
) -> tuple[dict[str, float], Array]:
    """Evaluate FE reconstruction against the analytic pressure and zero source."""
    latent_p = fe_state.apply_fn(
        {"params": fe_state.params},
        normalize_p(benchmark.p_pod, normalizer),
        method=FunctionEncoder.encode_p,
    )
    latent_f = fe_state.apply_fn(
        {"params": fe_state.params},
        normalize_f(benchmark.f_pod, normalizer),
        method=FunctionEncoder.encode_f,
    )
    pred_p = denormalize_p(
        fe_state.apply_fn(
            {"params": fe_state.params},
            latent_p,
            benchmark.eval_coords,
            method=FunctionEncoder.reconstruct_p,
        ),
        normalizer,
    )
    pred_f = denormalize_f(
        fe_state.apply_fn(
            {"params": fe_state.params},
            latent_f,
            benchmark.eval_coords,
            method=FunctionEncoder.reconstruct_f,
        ),
        normalizer,
    )
    outer_p = denormalize_p(
        fe_state.apply_fn(
            {"params": fe_state.params},
            latent_p,
            outer_boundary_coords(config),
            method=FunctionEncoder.reconstruct_p,
        ),
        normalizer,
    )
    jax.block_until_ready(pred_f)
    metrics = _field_metrics(pred_p, benchmark.p_eval, benchmark.area_weights)
    metrics.update(
        {
            "f_zero_rmse": float(jnp.sqrt(jnp.mean(pred_f**2))),
            "f_zero_max_abs": float(jnp.max(jnp.abs(pred_f))),
            "outer_dirichlet_max_abs": float(jnp.max(jnp.abs(outer_p))),
        }
    )
    return metrics, pred_p


@partial(jax.jit, static_argnames=("apply_fn",))
def _decoded_inner_flux(
    params,
    latent_p: Array,
    boundary_coords: Array,
    std_p: Array,
    mean_p: Array,
    drhat_dr: float,
    apply_fn,
) -> Array:
    def decoded_pressure(coord_hat):
        p_norm = apply_fn(
            {"params": params},
            latent_p,
            coord_hat[None, :],
            method=FunctionEncoder.reconstruct_p,
        )[0, 0]
        return p_norm * std_p + mean_p

    gradient_hat = jax.vmap(jax.grad(decoded_pressure))(boundary_coords)
    return -gradient_hat[:, 1] * drhat_dr


def evaluate_operator_exact(
    ol_state,
    fe_state,
    normalizer: FieldNormalizer,
    benchmark: ExactBenchmark,
    config: PolarAnnulusConfig,
) -> tuple[dict[str, float], Array]:
    """Evaluate SNO prediction against the analytic benchmark."""
    latent_f = fe_state.apply_fn(
        {"params": fe_state.params},
        normalize_f(benchmark.f_pod, normalizer),
        method=FunctionEncoder.encode_f,
    )
    target_latent_p = fe_state.apply_fn(
        {"params": fe_state.params},
        normalize_p(benchmark.p_pod, normalizer),
        method=FunctionEncoder.encode_p,
    )
    pred_latent_p = ol_state.apply_fn(
        {"params": ol_state.params},
        make_source_tokens(latent_f, config),
        make_condition_tokens_from_arrays(
            benchmark.boundary_coords,
            benchmark.boundary_flux,
            config,
        ),
        benchmark.k_values,
    )
    pred_p = denormalize_p(
        fe_state.apply_fn(
            {"params": fe_state.params},
            pred_latent_p,
            benchmark.eval_coords,
            method=FunctionEncoder.reconstruct_p,
        ),
        normalizer,
    )
    outer_p = denormalize_p(
        fe_state.apply_fn(
            {"params": fe_state.params},
            pred_latent_p,
            outer_boundary_coords(config),
            method=FunctionEncoder.reconstruct_p,
        ),
        normalizer,
    )
    pred_inner_flux = _decoded_inner_flux(
        fe_state.params,
        pred_latent_p,
        benchmark.boundary_coords[0],
        normalizer.std_p,
        normalizer.mean_p,
        config.drhat_dr,
        fe_state.apply_fn,
    )
    jax.block_until_ready(pred_inner_flux)

    metrics = _field_metrics(pred_p, benchmark.p_eval, benchmark.area_weights)
    latent_denominator = jnp.maximum(jnp.linalg.norm(target_latent_p), 1.0e-12)
    flux_denominator = jnp.maximum(
        jnp.linalg.norm(benchmark.boundary_flux[0]), 1.0e-12
    )
    metrics.update(
        {
            "latent_relative_l2": float(
                jnp.linalg.norm(pred_latent_p - target_latent_p)
                / latent_denominator
            ),
            "outer_dirichlet_max_abs": float(jnp.max(jnp.abs(outer_p))),
            "inner_flux_relative_l2": float(
                jnp.linalg.norm(
                    pred_inner_flux - benchmark.boundary_flux[0]
                )
                / flux_denominator
            ),
        }
    )
    return metrics, pred_p


def save_exact_monitor_figure(
    benchmark: ExactBenchmark,
    pred_p: Array,
    path: str | Path,
    title: str,
) -> Path:
    """Save a headless three-panel monitor figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = benchmark.eval_radius.shape
    radius = np.asarray(benchmark.eval_radius)
    theta = np.asarray(benchmark.eval_theta)
    exact = np.asarray(benchmark.p_eval).reshape(shape)
    pred = np.asarray(jax.device_get(pred_p)).reshape(shape)
    error = np.abs(pred - exact)
    solution_limit = max(
        float(np.max(np.abs(exact))),
        float(np.max(np.abs(pred))),
        np.finfo(float).eps,
    )

    figure = Figure(figsize=(9.0, 3.1), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    fields = (exact, pred, error)
    panel_titles = ("Exact", "Prediction / reconstruction", "Absolute error")
    for index, (axis, field, panel_title) in enumerate(
        zip(axes, fields, panel_titles)
    ):
        if index < 2:
            mesh = axis.pcolormesh(
                theta,
                radius,
                field,
                shading="auto",
                cmap="RdBu_r",
                vmin=-solution_limit,
                vmax=solution_limit,
            )
        else:
            mesh = axis.pcolormesh(
                theta,
                radius,
                field,
                shading="auto",
                cmap="magma",
                vmin=0.0,
            )
        axis.set_xlabel("theta")
        axis.set_ylabel("r")
        axis.set_title(panel_title)
        figure.colorbar(mesh, ax=axis, shrink=0.82)
    figure.suptitle(title)
    figure.savefig(path, dpi=180, facecolor="white")
    return path
