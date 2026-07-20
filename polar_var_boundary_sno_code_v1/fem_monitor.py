from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.io import loadmat, savemat

from config_varpolar import VarPolarConfig
from data_varpolar import (
    FieldConditionNormalizer,
    GeometryParams,
    denormalize_f,
    denormalize_p,
    evaluate_geometry,
    inner_boundary_coords,
    make_condition_tokens_from_arrays,
    make_reference_grid,
    make_source_tokens,
    normalize_f,
    normalize_p,
    outer_boundary_coords,
    sample_geometry_params,
    sample_k_values,
    theta_from_hat,
)
from models_varpolar import FunctionEncoder


Array = jax.Array


class FEMMonitorSet(NamedTuple):
    pod_coords: Array
    eval_coords: Array
    boundary_coords: Array
    p_pod: Array
    p_eval: Array
    area_weights: Array
    boundary_a: Array
    boundary_h: Array
    boundary_load: Array
    boundary_unit_flux: Array
    k_values: Array
    convergence_error: Array
    mesh_level_used: Array
    pcg_relres: Array
    pcg_iterations: Array
    geometry_params: GeometryParams
    monitor_seed: Array


def _matlab_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''").replace("\\", "/")


def export_fem_manifest(
    config: VarPolarConfig,
    path: str | Path | None = None,
) -> Path:
    """Generate the fixed geometry/k manifest without producing FEM solutions."""
    target = Path(path) if path is not None else config.fem_manifest_path
    target.parent.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(config.fem_monitor_seed)
    key_geometry, key_k = jax.random.split(key)
    geometry = sample_geometry_params(
        key_geometry, config.fem_monitor_size, config
    )
    k_values = sample_k_values(key_k, config.fem_monitor_size, config)

    check_theta = jnp.linspace(0.0, 2.0 * jnp.pi, 33, endpoint=False)
    check = evaluate_geometry(geometry, check_theta, config)
    levels = np.asarray(config.fem_mesh_levels, dtype=np.int32)
    savemat(
        target,
        {
            "geometry_w1": np.asarray(geometry.w1, dtype=np.float64),
            "geometry_b1": np.asarray(geometry.b1, dtype=np.float64),
            "geometry_w2": np.asarray(geometry.w2, dtype=np.float64),
            "k_values": np.asarray(k_values, dtype=np.float64)[:, None],
            "geom_base": np.asarray([[config.geom_base]], dtype=np.float64),
            "geom_amp": np.asarray([[config.geom_amp]], dtype=np.float64),
            "geom_tanh_scale": np.asarray(
                [[config.geom_tanh_scale]], dtype=np.float64
            ),
            "outer_scale": np.asarray([[config.outer_scale]], dtype=np.float64),
            "theta_size": np.asarray([[config.theta_size]], dtype=np.int32),
            "radial_size": np.asarray([[config.radial_size]], dtype=np.int32),
            "eval_theta_size": np.asarray(
                [[config.fem_eval_theta_size]], dtype=np.int32
            ),
            "eval_radial_size": np.asarray(
                [[config.fem_eval_radial_size]], dtype=np.int32
            ),
            "mesh_levels": levels,
            "convergence_tol": np.asarray(
                [[config.fem_convergence_tol]], dtype=np.float64
            ),
            "pcg_tol": np.asarray([[config.fem_pcg_tol]], dtype=np.float64),
            "pcg_maxiter": np.asarray(
                [[config.fem_pcg_maxiter]], dtype=np.int32
            ),
            "monitor_seed": np.asarray(
                [[config.fem_monitor_seed]], dtype=np.int64
            ),
            "check_theta": np.asarray(check_theta, dtype=np.float64)[None, :],
            "check_a": np.asarray(check.a, dtype=np.float64),
            "check_a_theta": np.asarray(check.a_theta, dtype=np.float64),
        },
        do_compression=True,
    )
    return target


def run_matlab_fem_builder(
    config: VarPolarConfig,
    manifest_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    manifest = (
        Path(manifest_path) if manifest_path is not None else config.fem_manifest_path
    )
    output = Path(output_path) if output_path is not None else config.fem_monitor_path
    matlab_dir = Path(__file__).resolve().parent / "matlab"
    if not manifest.exists():
        raise FileNotFoundError(
            f"FEM manifest not found: {manifest}. Run export_fem_manifest first."
        )
    executable = Path(config.matlab_executable)
    if not executable.exists():
        raise FileNotFoundError(f"MATLAB executable not found: {executable}")
    expression = (
        f"addpath('{_matlab_path(matlab_dir)}'); "
        f"build_fem_monitor_set('{_matlab_path(manifest)}', "
        f"'{_matlab_path(output)}');"
    )
    subprocess.run(
        [str(executable), "-batch", expression],
        check=True,
        cwd=matlab_dir,
    )
    if not output.exists():
        raise RuntimeError(f"MATLAB completed without producing {output}.")
    return output


def _require_matrix(data: dict[str, Any], name: str) -> np.ndarray:
    if name not in data:
        raise KeyError(f"Missing {name!r} in FEM monitor file.")
    return np.asarray(data[name])


def load_fem_monitor(path: str | Path) -> FEMMonitorSet:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"FEM monitor set not found: {source}")
    data = loadmat(source)

    def array(name: str) -> Array:
        return jnp.asarray(_require_matrix(data, name), dtype=jnp.float32)

    k_values = array("k_values").reshape(-1)
    convergence = array("convergence_error").reshape(-1)
    level_used = array("mesh_level_used").reshape(-1)
    relres = array("pcg_relres").reshape(-1)
    iterations = array("pcg_iterations").reshape(-1)
    geometry = GeometryParams(
        w1=array("geometry_w1"),
        b1=array("geometry_b1"),
        w2=array("geometry_w2"),
    )
    monitor = FEMMonitorSet(
        pod_coords=array("pod_coords"),
        eval_coords=array("eval_coords"),
        boundary_coords=array("boundary_coords"),
        p_pod=array("p_pod"),
        p_eval=array("p_eval"),
        area_weights=array("area_weights"),
        boundary_a=array("boundary_a"),
        boundary_h=array("boundary_h"),
        boundary_load=array("boundary_load"),
        boundary_unit_flux=array("boundary_unit_flux"),
        k_values=k_values,
        convergence_error=convergence,
        mesh_level_used=level_used,
        pcg_relres=relres,
        pcg_iterations=iterations,
        geometry_params=geometry,
        monitor_seed=jnp.asarray(
            _require_matrix(data, "monitor_seed"), dtype=jnp.int32
        ).reshape(()),
    )
    batch_size = monitor.p_pod.shape[0]
    expected = {
        "p_eval": monitor.p_eval.shape[0],
        "area_weights": monitor.area_weights.shape[0],
        "boundary_a": monitor.boundary_a.shape[0],
        "boundary_h": monitor.boundary_h.shape[0],
        "boundary_load": monitor.boundary_load.shape[0],
        "k_values": monitor.k_values.shape[0],
        "geometry_w1": monitor.geometry_params.w1.shape[0],
        "geometry_b1": monitor.geometry_params.b1.shape[0],
        "geometry_w2": monitor.geometry_params.w2.shape[0],
    }
    if any(size != batch_size for size in expected.values()):
        raise ValueError(f"Inconsistent FEM monitor batch dimensions: {expected}")
    return monitor


def _decoded_boundary_load(
    fe_state,
    latent_p: Array,
    boundary_coords: Array,
    boundary_a: Array,
    boundary_h: Array,
    normalizer: FieldConditionNormalizer,
) -> Array:
    """Differentiate the continuous decoder and evaluate the chosen B_a."""

    def one_sample(z, a, h):
        def decoded_pressure(coord):
            p_norm = fe_state.apply_fn(
                {"params": fe_state.params},
                z[None, :],
                coord[None, :],
                method=FunctionEncoder.reconstruct_p,
            )[0, 0]
            return p_norm * normalizer.std_p

        gradient = jax.vmap(jax.grad(decoded_pressure))(boundary_coords)
        return (
            h / (a * jnp.pi) * gradient[:, 0]
            - (1.0 + h**2) / (2.0 * a) * gradient[:, 1]
        )

    return jax.vmap(one_sample)(latent_p, boundary_a, boundary_h)


def _field_metrics(
    prediction: Array,
    reference: Array,
    weights: Array,
) -> dict[str, np.ndarray]:
    error = prediction - reference
    eps = jnp.finfo(prediction.dtype).tiny
    grid_l2 = jnp.linalg.norm(error, axis=1) / jnp.maximum(
        jnp.linalg.norm(reference, axis=1), eps
    )
    area_l2 = jnp.sqrt(
        jnp.sum(weights * error**2, axis=1)
        / jnp.maximum(jnp.sum(weights * reference**2, axis=1), eps)
    )
    rmse = jnp.sqrt(jnp.mean(error**2, axis=1))
    rel_linf = jnp.max(jnp.abs(error), axis=1) / jnp.maximum(
        jnp.max(jnp.abs(reference), axis=1), eps
    )
    return {
        "p_grid_relative_l2": np.asarray(grid_l2),
        "p_area_relative_l2": np.asarray(area_l2),
        "p_rmse": np.asarray(rmse),
        "p_relative_linf": np.asarray(rel_linf),
    }


def _aggregate(per_sample: dict[str, np.ndarray]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, values in per_sample.items():
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        result[f"{name}_mean"] = float(np.mean(values))
        result[f"{name}_median"] = float(np.median(values))
        result[f"{name}_p95"] = float(np.percentile(values, 95.0))
        result[f"{name}_max"] = float(np.max(values))
    return result


def _slice_monitor(monitor: FEMMonitorSet, start: int, stop: int) -> FEMMonitorSet:
    return FEMMonitorSet(
        pod_coords=monitor.pod_coords,
        eval_coords=monitor.eval_coords,
        boundary_coords=monitor.boundary_coords,
        p_pod=monitor.p_pod[start:stop],
        p_eval=monitor.p_eval[start:stop],
        area_weights=monitor.area_weights[start:stop],
        boundary_a=monitor.boundary_a[start:stop],
        boundary_h=monitor.boundary_h[start:stop],
        boundary_load=monitor.boundary_load[start:stop],
        boundary_unit_flux=monitor.boundary_unit_flux[start:stop],
        k_values=monitor.k_values[start:stop],
        convergence_error=monitor.convergence_error[start:stop],
        mesh_level_used=monitor.mesh_level_used[start:stop],
        pcg_relres=monitor.pcg_relres[start:stop],
        pcg_iterations=monitor.pcg_iterations[start:stop],
        geometry_params=GeometryParams(
            *(value[start:stop] for value in monitor.geometry_params)
        ),
        monitor_seed=monitor.monitor_seed,
    )


def evaluate_fe_fem(
    fe_state,
    normalizer: FieldConditionNormalizer,
    monitor: FEMMonitorSet,
    config: VarPolarConfig,
) -> tuple[dict[str, float], dict[str, np.ndarray], Array]:
    predictions = []
    f_predictions = []
    outer_values = []
    boundary_values = []
    for start in range(0, monitor.p_pod.shape[0], config.fem_eval_chunk_size):
        part = _slice_monitor(
            monitor, start, min(start + config.fem_eval_chunk_size, monitor.p_pod.shape[0])
        )
        zeros_pod = jnp.zeros_like(part.p_pod)
        latent_p = fe_state.apply_fn(
            {"params": fe_state.params},
            normalize_p(part.p_pod, normalizer),
            method=FunctionEncoder.encode_p,
        )
        latent_f = fe_state.apply_fn(
            {"params": fe_state.params},
            normalize_f(zeros_pod, normalizer),
            method=FunctionEncoder.encode_f,
        )
        predictions.append(
            denormalize_p(
                fe_state.apply_fn(
                    {"params": fe_state.params},
                    latent_p,
                    part.eval_coords,
                    method=FunctionEncoder.reconstruct_p,
                ),
                normalizer,
            )
        )
        f_predictions.append(
            denormalize_f(
                fe_state.apply_fn(
                    {"params": fe_state.params},
                    latent_f,
                    part.eval_coords,
                    method=FunctionEncoder.reconstruct_f,
                ),
                normalizer,
            )
        )
        outer_values.append(
            denormalize_p(
                fe_state.apply_fn(
                    {"params": fe_state.params},
                    latent_p,
                    outer_boundary_coords(config),
                    method=FunctionEncoder.reconstruct_p,
                ),
                normalizer,
            )
        )
        boundary_values.append(
            _decoded_boundary_load(
                fe_state,
                latent_p,
                part.boundary_coords,
                part.boundary_a,
                part.boundary_h,
                normalizer,
            )
        )

    prediction = jnp.concatenate(predictions, axis=0)
    f_prediction = jnp.concatenate(f_predictions, axis=0)
    outer = jnp.concatenate(outer_values, axis=0)
    boundary = jnp.concatenate(boundary_values, axis=0)
    per_sample = _field_metrics(prediction, monitor.p_eval, monitor.area_weights)
    per_sample.update(
        {
            "f_zero_rmse": np.asarray(jnp.sqrt(jnp.mean(f_prediction**2, axis=1))),
            "f_zero_max_abs": np.asarray(jnp.max(jnp.abs(f_prediction), axis=1)),
            "outer_dirichlet_max_abs": np.asarray(jnp.max(jnp.abs(outer), axis=1)),
            "inner_load_relative_l2": np.asarray(
                jnp.linalg.norm(boundary - monitor.boundary_load, axis=1)
                / jnp.maximum(jnp.linalg.norm(monitor.boundary_load, axis=1), 1.0e-12)
            ),
        }
    )
    return _aggregate(per_sample), per_sample, prediction


def evaluate_operator_fem(
    ol_state,
    fe_state,
    normalizer: FieldConditionNormalizer,
    monitor: FEMMonitorSet,
    config: VarPolarConfig,
) -> tuple[dict[str, float], dict[str, np.ndarray], Array]:
    predictions = []
    latent_errors = []
    outer_values = []
    boundary_values = []
    for start in range(0, monitor.p_pod.shape[0], config.fem_eval_chunk_size):
        part = _slice_monitor(
            monitor, start, min(start + config.fem_eval_chunk_size, monitor.p_pod.shape[0])
        )
        zeros_pod = jnp.zeros_like(part.p_pod)
        latent_f = fe_state.apply_fn(
            {"params": fe_state.params},
            normalize_f(zeros_pod, normalizer),
            method=FunctionEncoder.encode_f,
        )
        target_latent = fe_state.apply_fn(
            {"params": fe_state.params},
            normalize_p(part.p_pod, normalizer),
            method=FunctionEncoder.encode_p,
        )
        pred_latent = ol_state.apply_fn(
            {"params": ol_state.params},
            make_source_tokens(latent_f, config),
            make_condition_tokens_from_arrays(
                part.boundary_coords,
                part.boundary_a,
                part.boundary_h,
                part.boundary_load,
                normalizer,
                config,
            ),
            part.k_values,
        )
        predictions.append(
            denormalize_p(
                fe_state.apply_fn(
                    {"params": fe_state.params},
                    pred_latent,
                    part.eval_coords,
                    method=FunctionEncoder.reconstruct_p,
                ),
                normalizer,
            )
        )
        latent_errors.append(
            jnp.linalg.norm(pred_latent - target_latent, axis=1)
            / jnp.maximum(jnp.linalg.norm(target_latent, axis=1), 1.0e-12)
        )
        outer_values.append(
            denormalize_p(
                fe_state.apply_fn(
                    {"params": fe_state.params},
                    pred_latent,
                    outer_boundary_coords(config),
                    method=FunctionEncoder.reconstruct_p,
                ),
                normalizer,
            )
        )
        boundary_values.append(
            _decoded_boundary_load(
                fe_state,
                pred_latent,
                part.boundary_coords,
                part.boundary_a,
                part.boundary_h,
                normalizer,
            )
        )

    prediction = jnp.concatenate(predictions, axis=0)
    outer = jnp.concatenate(outer_values, axis=0)
    boundary = jnp.concatenate(boundary_values, axis=0)
    per_sample = _field_metrics(prediction, monitor.p_eval, monitor.area_weights)
    per_sample.update(
        {
            "latent_relative_l2": np.asarray(jnp.concatenate(latent_errors)),
            "outer_dirichlet_max_abs": np.asarray(jnp.max(jnp.abs(outer), axis=1)),
            "inner_load_relative_l2": np.asarray(
                jnp.linalg.norm(boundary - monitor.boundary_load, axis=1)
                / jnp.maximum(jnp.linalg.norm(monitor.boundary_load, axis=1), 1.0e-12)
            ),
        }
    )
    return _aggregate(per_sample), per_sample, prediction


def make_synthetic_monitor(
    config: VarPolarConfig,
    batch_size: int = 2,
) -> FEMMonitorSet:
    """Shape-safe test fixture; it is never used as a physical reference."""
    pod = make_reference_grid(config)
    eval_coords = pod
    boundary = inner_boundary_coords(config)
    key = jax.random.PRNGKey(config.fem_monitor_seed + 1)
    geometry = sample_geometry_params(key, batch_size, config)
    theta = theta_from_hat(boundary[:, 0])
    values = evaluate_geometry(geometry, theta, config)
    h = values.a_theta / values.a
    load = jnp.cos(theta)[None, :] + h * jnp.sin(theta)[None, :]
    manufactured = (
        jnp.sin(theta_from_hat(pod[:, 0])) * 0.5 * (1.0 - pod[:, 1])
    )
    p = jnp.broadcast_to(manufactured[None, :], (batch_size, pod.shape[0]))
    eta = eval_coords[:, 1]
    theta_eval = theta_from_hat(eval_coords[:, 0])
    geom_eval = evaluate_geometry(geometry, theta_eval, config)
    radius = geom_eval.a * (3.0 + 2.0 * eta[None, :])
    weights = 2.0 * geom_eval.a * radius
    return FEMMonitorSet(
        pod_coords=pod,
        eval_coords=eval_coords,
        boundary_coords=boundary,
        p_pod=p,
        p_eval=p,
        area_weights=weights,
        boundary_a=values.a,
        boundary_h=h,
        boundary_load=load,
        boundary_unit_flux=load / jnp.sqrt(1.0 + h**2),
        k_values=jnp.linspace(config.k_min, config.k_max, batch_size),
        convergence_error=jnp.zeros((batch_size,)),
        mesh_level_used=jnp.ones((batch_size,)),
        pcg_relres=jnp.zeros((batch_size,)),
        pcg_iterations=jnp.zeros((batch_size,)),
        geometry_params=geometry,
        monitor_seed=jnp.asarray(config.fem_monitor_seed),
    )
