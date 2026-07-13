from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
import argparse
import csv
import json
import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from scipy.io import loadmat

from config_polar import PolarAnnulusConfig, make_ablation_config
from data_polar import (
    denormalize_p,
    make_condition_tokens_from_arrays,
    make_polar_grid,
    make_source_tokens,
    make_target_cosine_boundary,
    normalize_f,
    r_from_hat,
    sample_batch,
    sobol_polar_points,
    theta_from_hat,
)
from exact_solution import exact_annulus_solution
from models_polar import FunctionEncoder
from train_polar import (
    encode_operator_batch,
    fe_eval_step,
    load_trained_states,
    ol_eval_step,
)


ALPHAS = (0.25, 0.5, 1.0, 2.0, 4.0)
EVALUATION_SEED = 24_681_357


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _predict_case(config, fe_state, ol_state, normalizer, alpha: float):
    coords = make_polar_grid(config)
    zero_f = jnp.zeros((1, config.n_pod), dtype=jnp.float32)
    latent_f = fe_state.apply_fn(
        {"params": fe_state.params},
        normalize_f(zero_f, normalizer),
        method=FunctionEncoder.encode_f,
    )
    boundary_coords, boundary_flux = make_target_cosine_boundary(config, 1)
    boundary_flux = alpha * boundary_flux
    boundary_tokens = make_condition_tokens_from_arrays(
        boundary_coords, boundary_flux, config
    )
    latent_p = ol_state.apply_fn(
        {"params": ol_state.params},
        make_source_tokens(latent_f, config),
        boundary_tokens,
        jnp.asarray([config.k_min], dtype=jnp.float32),
    )
    p_norm = fe_state.apply_fn(
        {"params": fe_state.params},
        latent_p,
        coords,
        method=FunctionEncoder.reconstruct_p,
    )
    return coords, latent_p, denormalize_p(p_norm, normalizer)[0]


def _area_weights(config: PolarAnnulusConfig) -> np.ndarray:
    radius = np.linspace(config.r_inner, config.r_outer, config.radial_size)
    radial_quadrature = np.ones(config.radial_size)
    radial_quadrature[[0, -1]] = 0.5
    return np.repeat(radius * radial_quadrature, config.theta_size)


def _field_metrics(pred: np.ndarray, exact: np.ndarray, weights: np.ndarray) -> dict:
    error = pred - exact
    eps = 1.0e-12
    return {
        "area_weighted_relative_l2": float(
            np.sqrt(np.sum(weights * error**2) / max(np.sum(weights * exact**2), eps))
        ),
        "grid_relative_l2": float(
            np.linalg.norm(error) / max(np.linalg.norm(exact), eps)
        ),
        "relative_linf": float(
            np.max(np.abs(error)) / max(np.max(np.abs(exact)), eps)
        ),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def _pressure_scalar(fe_state, normalizer, latent_p, config, coord):
    value_norm = fe_state.apply_fn(
        {"params": fe_state.params},
        latent_p,
        coord[None, :],
        method=FunctionEncoder.reconstruct_p,
    )[0, 0]
    return value_norm * normalizer.std_p + normalizer.mean_p


def _physics_metrics(config, fe_state, normalizer, latent_p, alpha: float) -> dict:
    boundary_coords, _ = make_target_cosine_boundary(config, 1)
    boundary_coords = boundary_coords[0]
    scalar = lambda coord: _pressure_scalar(
        fe_state, normalizer, latent_p, config, coord
    )
    boundary_grad = jax.vmap(jax.grad(scalar))(boundary_coords)
    predicted_flux = -config.drhat_dr * boundary_grad[:, 1]
    theta = theta_from_hat(boundary_coords[:, 0])
    target_flux = alpha * jnp.cos(theta)
    flux_relative_l2 = jnp.linalg.norm(predicted_flux - target_flux) / jnp.maximum(
        jnp.linalg.norm(target_flux), 1.0e-12
    )

    outer_coords = boundary_coords.at[:, 1].set(1.0)
    outer_pressure = jax.vmap(scalar)(outer_coords)

    residual_coords = sobol_polar_points(
        jax.random.PRNGKey(EVALUATION_SEED + int(round(alpha * 100))),
        min(config.random_probe_points, 256),
        config,
    )
    values = jax.vmap(scalar)(residual_coords)
    gradients = jax.vmap(jax.grad(scalar))(residual_coords)
    hessians = jax.vmap(jax.hessian(scalar))(residual_coords)
    radius = r_from_hat(residual_coords[:, 1], config)
    p_r = config.drhat_dr * gradients[:, 1]
    p_rr = config.drhat_dr**2 * hessians[:, 1, 1]
    p_theta2 = config.dthetahat_dtheta**2 * hessians[:, 0, 0]
    residual = (
        p_rr
        + p_r / radius
        + p_theta2 / radius**2
        - config.k_min**2 * values
    )
    residual_rms = jnp.sqrt(jnp.mean(residual**2))
    reference_rms = jnp.sqrt(jnp.mean((config.k_min**2 * values) ** 2))
    jax.block_until_ready(residual_rms)
    return {
        "boundary_flux_relative_l2": float(flux_relative_l2),
        "outer_boundary_max_abs": float(jnp.max(jnp.abs(outer_pressure))),
        "pde_residual_rms": float(residual_rms),
        "pde_residual_relative_rms": float(
            residual_rms / jnp.maximum(reference_rms, 1.0e-12)
        ),
        "pde_residual_points": int(residual_coords.shape[0]),
    }


def _distribution_metrics(
    config,
    fe_state,
    ol_state,
    normalizer,
    samples_per_scale: int,
    chunk_size: int,
) -> list[dict]:
    if samples_per_scale <= 0 or chunk_size <= 0:
        return []
    key = jax.random.PRNGKey(EVALUATION_SEED)
    results = []
    for pair in config.prior_scale_pairs:
        remaining = samples_per_scale
        totals = {
            "fe_p_relative_l2": 0.0,
            "fe_f_relative_l2": 0.0,
            "ol_latent_mse": 0.0,
            "ol_latent_relative_l2": 0.0,
            "ol_physical_relative_l2": 0.0,
        }
        while remaining:
            current = min(chunk_size, remaining)
            eval_config = replace(
                config,
                prior_scale_pairs=(pair,),
                repeats_per_scale=1,
                sample_size=current,
            )
            key, batch_key = jax.random.split(key)
            batch = sample_batch(batch_key, eval_config)
            fe_p, fe_f = fe_eval_step(fe_state, batch, normalizer, eval_config)
            f_tokens, boundary_tokens, target_latent = encode_operator_batch(
                fe_state, batch, normalizer, eval_config
            )
            ol_metrics = ol_eval_step(
                ol_state.params,
                ol_state.apply_fn,
                fe_state.params,
                fe_state.apply_fn,
                f_tokens,
                boundary_tokens,
                batch.k_values,
                target_latent,
                batch.probe_coords,
                batch.p_probe,
                normalizer,
            )
            jax.block_until_ready(ol_metrics["physical_relative_l2"])
            totals["fe_p_relative_l2"] += current * float(fe_p)
            totals["fe_f_relative_l2"] += current * float(fe_f)
            totals["ol_latent_mse"] += current * float(ol_metrics["latent_mse"])
            totals["ol_latent_relative_l2"] += current * float(
                ol_metrics["latent_relative_l2"]
            )
            totals["ol_physical_relative_l2"] += current * float(
                ol_metrics["physical_relative_l2"]
            )
            remaining -= current
        results.append(
            {
                "sigma_theta": pair[0],
                "sigma_r": pair[1],
                "samples": samples_per_scale,
                **{name: value / samples_per_scale for name, value in totals.items()},
            }
        )
    return results


def _training_cost(config: PolarAnnulusConfig) -> dict:
    result = {}
    for stage in ("fe", "ol"):
        path = config.output_dir / f"{stage}_checkpoint_latest.json"
        if path.exists():
            result[stage] = json.loads(path.read_text(encoding="utf-8"))
    return result


def collect_historical_comparisons(project_root: Path) -> dict:
    """Read reference artifacts without importing or writing to old projects."""
    result: dict[str, Any] = {
        "comparison_type": "descriptive_only",
        "strict_causal_comparison": False,
        "polar_q_v2_policy": "compare descriptively only with new OL step 10000",
    }
    polar_root = project_root / "polar_annulus_sno_code" / "out_polar_annulus_sno" / "polar_v2"
    polar_metrics = polar_root / "sno_exact_evaluation" / "sno_vs_exact_k_1_metrics.json"
    if polar_metrics.exists():
        result["polar_q_v2_exact"] = json.loads(polar_metrics.read_text(encoding="utf-8"))
    history_path = polar_root / "operator_training_history.csv"
    if history_path.exists():
        with history_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            result["polar_q_v2_recorded_progress"] = {
                "last_history_row": rows[-1],
                "step_metadata_reliable_for_checkpoint": False,
            }

    cartesian_candidates = (
        project_root / "annulus_sno" / "data" / "sno_100_flux_samples.mat",
        project_root
        / "annulus_sno"
        / "annulus_sno_annulus_only_v2"
        / "out"
        / "Expand the prior space"
        / "sno_100_flux_samples.mat",
    )
    for path in cartesian_candidates:
        if not path.exists():
            continue
        with path.open("rb") as handle:
            data = loadmat(handle)
        result["cartesian_p_prior_distribution"] = {
            "source": str(path.relative_to(project_root)),
            "samples": int(np.asarray(data["err_sno_pod"]).size),
            "mean_grid_relative_l2": float(np.mean(data["err_sno_pod"])),
            "mean_probe_relative_l2": float(np.mean(data["err_sno_probe"])),
            "not_an_exact_case_coordinate_effect": True,
        }
        break
    return result


def _ablation_conclusion(config: PolarAnnulusConfig, report: dict) -> dict:
    other_variant = "p_rms" if config.pressure_prior_scaling == "raw" else "p_raw"
    smoke_suffix = "_smoke" if config.run_name.endswith("_smoke") else ""
    other_name = f"polar_{other_variant}_seed{config.seed}{smoke_suffix}"
    other_path = Path(config.out_dir) / other_name / "evaluation" / "metrics.json"
    if not other_path.exists():
        return {"status": "pending_other_variant"}
    other = json.loads(other_path.read_text(encoding="utf-8"))
    raw = report if config.pressure_prior_scaling == "raw" else other
    rms = report if config.pressure_prior_scaling == "boundary_rms" else other
    raw_area = np.mean([case["area_weighted_relative_l2"] for case in raw["exact_cases"]])
    rms_area = np.mean([case["area_weighted_relative_l2"] for case in rms["exact_cases"]])
    raw_flux = np.mean([case["boundary_flux_relative_l2"] for case in raw["exact_cases"]])
    rms_flux = np.mean([case["boundary_flux_relative_l2"] for case in rms["exact_cases"]])
    area_gain = (raw_area - rms_area) / max(raw_area, 1.0e-12)
    flux_gain = (raw_flux - rms_flux) / max(raw_flux, 1.0e-12)
    if area_gain >= 0.20 and flux_gain >= 0.20:
        label = "boundary_amplitude_inflation_is_important"
    elif area_gain < 0.10 and flux_gain < 0.10:
        label = "amplitude_calibration_is_not_a_main_factor"
    else:
        label = "uncertain"
    is_smoke = config.run_name.endswith("_smoke")
    return {
        "status": (
            "smoke_only_not_a_scientific_result"
            if is_smoke
            else "complete_single_seed_deterministic_comparison"
        ),
        "label": label if not is_smoke else f"pipeline_only_{label}",
        "area_weighted_relative_l2_improvement": float(area_gain),
        "boundary_flux_relative_l2_improvement": float(flux_gain),
        "statistical_significance_claimed": False,
        "eligible_for_experiment_conclusion": not is_smoke,
    }


def _plot_report(report: dict, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    alphas = [case["alpha"] for case in report["exact_cases"]]
    area = [case["area_weighted_relative_l2"] for case in report["exact_cases"]]
    flux = [case["boundary_flux_relative_l2"] for case in report["exact_cases"]]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)
    axes[0].plot(alphas, area, marker="o")
    axes[0].set(xlabel=r"$\alpha$", ylabel="area-weighted relative L2")
    axes[1].plot(alphas, flux, marker="o", color="#d55e00")
    axes[1].set(xlabel=r"$\alpha$", ylabel="boundary-flux relative L2")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_xscale("log", base=2)
    fig.savefig(output_dir / "exact_alpha_metrics.png", dpi=200)
    plt.close(fig)


def evaluate_variant(
    config: PolarAnnulusConfig,
    include_existing: bool = True,
    samples_per_scale: int = 256,
    chunk_size: int = 32,
    checkpoint_step: int | None = None,
) -> dict:
    fe_state, ol_state, normalizer = load_trained_states(config)
    if checkpoint_step is not None:
        milestone_path = (
            config.output_dir / f"ol_params_step_{checkpoint_step:09d}.msgpack"
        )
        if not milestone_path.exists():
            raise FileNotFoundError(milestone_path)
        milestone_params = serialization.from_bytes(
            ol_state.params, milestone_path.read_bytes()
        )
        ol_state = ol_state.replace(params=milestone_params)
    output_dir = config.output_dir / (
        "evaluation"
        if checkpoint_step is None
        else f"evaluation_step_{checkpoint_step:09d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = _area_weights(config)
    exact_cases = []
    stored_arrays = {}
    for alpha in ALPHAS:
        coords, latent_p, pred = _predict_case(
            config, fe_state, ol_state, normalizer, alpha
        )
        radius = np.asarray(r_from_hat(coords[:, 1], config))
        theta = np.asarray(theta_from_hat(coords[:, 0]))
        exact = exact_annulus_solution(
            radius,
            theta,
            config.k_min,
            config.r_inner,
            config.r_outer,
            flux_amplitude=alpha,
        )
        metrics = _field_metrics(np.asarray(pred), exact, weights)
        metrics.update(
            _physics_metrics(config, fe_state, normalizer, latent_p, alpha)
        )
        exact_cases.append({"alpha": alpha, **metrics})
        stored_arrays[f"pred_alpha_{alpha:g}"] = np.asarray(pred)
        stored_arrays[f"exact_alpha_{alpha:g}"] = exact
    stored_arrays["coords_hat"] = np.asarray(coords)
    np.savez(output_dir / "exact_cases.npz", **stored_arrays)

    report: dict[str, Any] = {
        "variant": config.run_name,
        "seed": config.seed,
        "operator_checkpoint_step": checkpoint_step or config.ol_steps,
        "config_fingerprint": config.fingerprint(),
        "primary_metric": "area_weighted_relative_l2",
        "exact_cases": exact_cases,
        "distribution_in": _distribution_metrics(
            config,
            fe_state,
            ol_state,
            normalizer,
            samples_per_scale,
            chunk_size,
        ),
        "training_cost": _training_cost(config),
        "interpretation_limits": {
            "single_seed": True,
            "confidence_intervals_or_significance_tests": False,
            "coordinate_only_causal_claim_supported": False,
            "q_prior_legacy_checkpoint_progress_match": "approximate_descriptive_only",
            "q_prior_comparison_eligible": checkpoint_step == 10_000,
        },
    }
    if include_existing:
        project_root = Path(__file__).resolve().parent.parent
        report["historical_comparisons"] = collect_historical_comparisons(project_root)
    report["ablation_conclusion"] = (
        _ablation_conclusion(config, report)
        if checkpoint_step is None
        else {
            "status": "milestone_descriptive_only",
            "eligible_for_final_ablation_conclusion": False,
        }
    )
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(_json_ready(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _plot_report(report, output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one polar P-prior variant.")
    parser.add_argument("--variant", choices=("p_raw", "p_rms"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples-per-scale", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--no-existing", action="store_true")
    parser.add_argument("--checkpoint-step", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = make_ablation_config(
        args.variant, args.seed, root / "out_p_prior_ablation"
    )
    report = evaluate_variant(
        config,
        include_existing=not args.no_existing,
        samples_per_scale=args.samples_per_scale,
        chunk_size=args.chunk_size,
        checkpoint_step=args.checkpoint_step,
    )
    print(json.dumps(_json_ready(report["ablation_conclusion"]), indent=2))


if __name__ == "__main__":
    main()
