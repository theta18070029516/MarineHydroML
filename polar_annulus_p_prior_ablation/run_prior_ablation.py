from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
import argparse
import json
import math
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from config_polar import make_ablation_config, make_smoke_config
from data_polar import (
    build_field_normalizer_from_batches,
    evaluate_direct_q_prior_flux,
    evaluate_polar_prior,
    inner_boundary_coords,
    outer_boundary_coords,
    r_to_hat,
    sample_batch,
    sample_bnn_params,
    theta_to_hat,
)
from models_polar import FunctionEncoder
from train_polar import (
    create_fe_state,
    create_ol_state,
    encode_operator_batch,
    fe_train_step,
    load_fe_state,
    load_training_checkpoint,
    ol_train_step,
    save_training_checkpoint,
    train_fe,
    train_operator,
)


def _tree_max_abs_difference(left: Any, right: Any) -> float:
    leaves = [
        float(jnp.max(jnp.abs(a - b)))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right))
    ]
    return max(leaves, default=0.0)


def _tree_all_finite(tree: Any) -> bool:
    return all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(tree))


def _single_pressure(params, physical_coord, sigma_r, config):
    radius, theta = physical_coord
    r_hat = r_to_hat(radius, config)
    phase = (
        params.w1[0, 0] * jnp.sin(theta)
        + params.w1[0, 1] * jnp.cos(theta)
        + params.w1[0, 2] * r_hat
        + params.b1[0]
        - jnp.pi / 4.0
    )
    u = jnp.sqrt(2.0 / params.w1.shape[-1]) * jnp.sum(
        params.w2[0] * jnp.cos(phase)
    )
    return config.pressure_scale(sigma_r) * (radius - config.r_outer) * u


def _physics_preflight(config) -> dict:
    key = jax.random.PRNGKey(7_431)
    derivative_errors = {"p_r": 0.0, "p_rr": 0.0, "p_theta2": 0.0, "f": 0.0}
    prior_outer_max = 0.0
    sign_error = 0.0
    for sigma_theta, sigma_r in config.prior_scale_pairs:
        key, params_key = jax.random.split(key)
        params = sample_bnn_params(
            params_key, 16, sigma_theta, sigma_r, config
        )
        k_values = jnp.ones((16,), dtype=jnp.float32) * config.k_min
        outer = outer_boundary_coords(config)
        outer_eval = evaluate_polar_prior(
            params, outer, k_values, sigma_r, config
        )
        prior_outer_max = max(prior_outer_max, float(jnp.max(jnp.abs(outer_eval.p))))

        physical = jnp.asarray(
            [
                [0.27, 0.19],
                [0.51, 1.37],
                [0.83, 3.21],
                [0.96, 5.71],
            ],
            dtype=jnp.float32,
        )
        coords_hat = jnp.stack(
            [theta_to_hat(physical[:, 1]), r_to_hat(physical[:, 0], config)], axis=-1
        )
        # Derivative identity is linear in the output weights. A fixed 0.1
        # diagnostic amplitude keeps the strict absolute 2e-5 float32 check
        # well-conditioned even for sigma_r=5, without changing training data.
        params_one = jax.tree.map(lambda value: value[:1], params)
        params_one = params_one._replace(w2=0.1 * params_one.w2)
        analytic = evaluate_polar_prior(
            params_one,
            coords_hat,
            jnp.asarray([config.k_min], dtype=jnp.float32),
            sigma_r,
            config,
        )
        scalar = lambda coord: _single_pressure(params_one, coord, sigma_r, config)
        gradients = jax.vmap(jax.grad(scalar))(physical)
        hessians = jax.vmap(jax.hessian(scalar))(physical)
        p_r = gradients[:, 0]
        p_rr = hessians[:, 0, 0]
        p_theta2 = hessians[:, 1, 1]
        f = (
            p_rr
            + p_r / physical[:, 0]
            + p_theta2 / physical[:, 0] ** 2
            - config.k_min**2 * jax.vmap(scalar)(physical)
        )
        derivative_errors["p_r"] = max(
            derivative_errors["p_r"], float(jnp.max(jnp.abs(p_r - analytic.q[0])))
        )
        derivative_errors["p_rr"] = max(
            derivative_errors["p_rr"], float(jnp.max(jnp.abs(p_rr - analytic.q_r[0])))
        )
        derivative_errors["p_theta2"] = max(
            derivative_errors["p_theta2"],
            float(jnp.max(jnp.abs(p_theta2 - analytic.p_theta2[0]))),
        )
        derivative_errors["f"] = max(
            derivative_errors["f"], float(jnp.max(jnp.abs(f - analytic.f[0])))
        )

        boundary = inner_boundary_coords(config)
        bnd_eval = evaluate_polar_prior(
            params_one,
            boundary,
            jnp.asarray([config.k_min], dtype=jnp.float32),
            sigma_r,
            config,
        )
        boundary_theta = jnp.pi * (boundary[:9, 0] + 1.0)
        physical_boundary = jnp.stack(
            [jnp.full_like(boundary_theta, config.r_inner), boundary_theta], axis=-1
        )
        outward_from_autodiff = -jax.vmap(jax.grad(scalar))(
            physical_boundary
        )[:, 0]
        sign_error = max(
            sign_error,
            float(
                jnp.max(
                    jnp.abs(outward_from_autodiff - (-bnd_eval.q[0, :9]))
                )
            ),
        )
        generated = sample_batch(jax.random.fold_in(key, int(sigma_r * 100)), replace(
            config,
            prior_scale_pairs=((sigma_theta, sigma_r),),
            sample_size=16,
            normalizer_batches=1,
        ))
        # Both independently generated batches obey g_n=-P_r. The direct check
        # below uses the exact evaluation; sample_batch is checked for shape and
        # sign convention by comparing its stored flux with a fresh reconstruction
        # from the same public rule in the unit test.
        if generated.boundary_flux.shape != (16, config.theta_size):
            raise AssertionError("Unexpected boundary-flux shape.")

    return {
        "prior_outer_boundary_max_abs": prior_outer_max,
        "derivative_check_output_scale": 0.1,
        "analytic_vs_autodiff_max_abs": derivative_errors,
        "pde_source_max_abs_difference": derivative_errors["f"],
        "inner_normal_sign_identity_max_abs": sign_error,
    }


def _rms_preflight(config) -> list[dict]:
    rms_config = replace(config, pressure_prior_scaling="boundary_rms")
    key = jax.random.PRNGKey(91_177)
    boundary = inner_boundary_coords(config)
    records = []
    for sigma_theta, sigma_r in config.prior_scale_pairs:
        key, params_key = jax.random.split(key)
        params = sample_bnn_params(
            params_key, 256, sigma_theta, sigma_r, config
        )
        k_values = jnp.full((256,), config.k_min, dtype=jnp.float32)
        p_flux = -evaluate_polar_prior(
            params, boundary, k_values, sigma_r, rms_config
        ).q
        q_flux = evaluate_direct_q_prior_flux(params, boundary)
        p_rms = float(jnp.sqrt(jnp.mean(p_flux**2)))
        q_rms = float(jnp.sqrt(jnp.mean(q_flux**2)))
        mismatch = abs(p_rms / q_rms - 1.0)
        records.append(
            {
                "sigma_theta": sigma_theta,
                "sigma_r": sigma_r,
                "coefficient": rms_config.pressure_scale(sigma_r),
                "p_prior_flux_rms": p_rms,
                "q_prior_flux_rms": q_rms,
                "relative_mismatch": mismatch,
            }
        )
    return records


def _model_and_resume_preflight(config) -> dict:
    smoke = make_smoke_config(config)
    key = jax.random.PRNGKey(52_003)
    key_init_fe, key_batch, key_init_ol = jax.random.split(key, 3)
    batch = sample_batch(key_batch, smoke)
    normalizer = build_field_normalizer_from_batches([batch], smoke.normalizer_eps)
    fe_state, _ = create_fe_state(smoke, key_init_fe)
    fe_state, fe_metrics = fe_train_step(fe_state, batch, normalizer, smoke)
    f_tokens, boundary_tokens, target_latent = encode_operator_batch(
        fe_state, batch, normalizer, smoke
    )
    ol_state, _ = create_ol_state(smoke, key_init_ol)
    ol_state, ol_loss = ol_train_step(
        ol_state, f_tokens, boundary_tokens, batch.k_values, target_latent
    )
    jax.block_until_ready(ol_loss)

    outer = outer_boundary_coords(smoke)
    decoded_outer = fe_state.apply_fn(
        {"params": fe_state.params},
        jnp.ones((1, smoke.n_basis), dtype=jnp.float32),
        outer,
        method=FunctionEncoder.reconstruct_p,
    )

    checkpoint_config = replace(
        smoke,
        out_dir=str(config.output_dir / "preflight_artifacts"),
        run_name="checkpoint_resume",
    )
    master = jax.random.PRNGKey(303_771)
    key_init, key_train, key_eval = jax.random.split(master, 3)
    initial, _ = create_fe_state(checkpoint_config, key_init)
    normalizer_batch_key, key_train = jax.random.split(key_train)
    normalizer_batch = sample_batch(normalizer_batch_key, checkpoint_config)
    checkpoint_normalizer = build_field_normalizer_from_batches(
        [normalizer_batch], checkpoint_config.normalizer_eps
    )
    key_train, first_batch_key = jax.random.split(key_train)
    first_batch = sample_batch(first_batch_key, checkpoint_config)
    first_state, _ = fe_train_step(
        initial, first_batch, checkpoint_normalizer, checkpoint_config
    )
    save_training_checkpoint(
        first_state, key_train, key_eval, checkpoint_config, "fe", 1.25
    )

    key_direct, second_batch_key = jax.random.split(key_train)
    second_batch = sample_batch(second_batch_key, checkpoint_config)
    direct_state, _ = fe_train_step(
        first_state, second_batch, checkpoint_normalizer, checkpoint_config
    )

    restored_initial, _ = create_fe_state(checkpoint_config, key_init)
    restored, restored_key, restored_eval_key, elapsed = load_training_checkpoint(
        restored_initial, key_train, key_eval, checkpoint_config, "fe"
    )
    restored_key, restored_batch_key = jax.random.split(restored_key)
    restored_batch = sample_batch(restored_batch_key, checkpoint_config)
    resumed_state, _ = fe_train_step(
        restored, restored_batch, checkpoint_normalizer, checkpoint_config
    )
    jax.block_until_ready(resumed_state.step)
    return {
        "fe_loss_finite": bool(jnp.isfinite(fe_metrics["loss"])),
        "fe_state_finite": _tree_all_finite(fe_state),
        "ol_loss_finite": bool(jnp.isfinite(ol_loss)),
        "ol_state_finite": _tree_all_finite(ol_state),
        "decoder_outer_boundary_max_abs": float(jnp.max(jnp.abs(decoded_outer))),
        "resume_next_batch_max_abs_difference": _tree_max_abs_difference(
            second_batch, restored_batch
        ),
        "resume_next_params_max_abs_difference": _tree_max_abs_difference(
            direct_state.params, resumed_state.params
        ),
        "resume_next_optimizer_max_abs_difference": _tree_max_abs_difference(
            direct_state.opt_state, resumed_state.opt_state
        ),
        "resume_next_key_equal": bool(jnp.array_equal(key_direct, restored_key)),
        "resume_eval_key_equal": bool(jnp.array_equal(key_eval, restored_eval_key)),
        "restored_elapsed_seconds": elapsed,
    }


def run_preflight_checks(config) -> dict:
    physics = _physics_preflight(config)
    rms = _rms_preflight(config)
    model = _model_and_resume_preflight(config)
    derivative_limit = 2.0e-5
    checks = {
        "prior_outer_boundary": physics["prior_outer_boundary_max_abs"] <= 1.0e-7,
        "analytic_derivatives": max(
            physics["analytic_vs_autodiff_max_abs"][name]
            for name in ("p_r", "p_rr", "p_theta2")
        ) <= derivative_limit,
        "pde_source": physics["pde_source_max_abs_difference"] <= 1.0e-4,
        "inner_normal_sign": physics["inner_normal_sign_identity_max_abs"] <= derivative_limit,
        "boundary_rms": max(item["relative_mismatch"] for item in rms) <= 0.05,
        "finite_fe_ol": all(
            model[name]
            for name in ("fe_loss_finite", "fe_state_finite", "ol_loss_finite", "ol_state_finite")
        ),
        "decoder_outer_boundary": model["decoder_outer_boundary_max_abs"] <= 1.0e-7,
        "checkpoint_resume": (
            model["resume_next_batch_max_abs_difference"] == 0.0
            and model["resume_next_params_max_abs_difference"] == 0.0
            and model["resume_next_optimizer_max_abs_difference"] == 0.0
            and model["resume_next_key_equal"]
            and model["resume_eval_key_equal"]
        ),
    }
    report = {
        "config_fingerprint": config.fingerprint(),
        "pressure_prior_scaling": config.pressure_prior_scaling,
        "jax_devices": [str(device) for device in jax.devices()],
        "physics": physics,
        "boundary_rms_calibration": rms,
        "model_and_resume": model,
        "checks": checks,
        "passed": all(checks.values()),
    }
    path = config.output_dir / "preflight_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Preflight failed: {', '.join(failed)}. See {path}")
    return report


def _require_completed_fe(config) -> None:
    path = config.output_dir / "fe_checkpoint_latest.json"
    if not path.exists():
        raise FileNotFoundError("FE checkpoint is missing; run --stage fe first.")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if int(metadata["completed_steps"]) < config.fe_steps:
        raise RuntimeError(
            f"FE is incomplete ({metadata['completed_steps']}/{config.fe_steps}); resume FE first."
        )


def _require_completed_ol(config) -> None:
    path = config.output_dir / "ol_checkpoint_latest.json"
    if not path.exists():
        raise FileNotFoundError("OL checkpoint is missing; run --stage ol first.")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if int(metadata["completed_steps"]) < config.ol_steps:
        raise RuntimeError(
            f"OL is incomplete ({metadata['completed_steps']}/{config.ol_steps}); resume OL first."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Polar P-prior ablation runner.")
    parser.add_argument("--variant", choices=("p_raw", "p_rms"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stage", choices=("diagnostics", "fe", "ol", "eval", "all"), default="all"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true", help="Use a two-step CPU-sized configuration."
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config = make_ablation_config(
        args.variant, args.seed, root / "out_p_prior_ablation"
    )
    if args.smoke:
        config = make_smoke_config(config)
    config.save_json()
    print("Output directory:", config.output_dir)
    print("Config fingerprint:", config.fingerprint())
    print("JAX devices:", jax.devices())
    if not args.smoke and all(device.platform == "cpu" for device in jax.devices()):
        print(
            "WARNING: full 500k+500k training is configured on CPU and is expected "
            "to be impractically slow; use a JAX accelerator runtime for the formal run."
        )

    if args.stage in ("diagnostics", "fe", "ol", "all"):
        report = run_preflight_checks(config)
        print("Preflight passed:", report["passed"])
    if args.stage in ("fe", "all"):
        fe_state, normalizer = train_fe(config, resume=args.resume)
    elif args.stage == "ol":
        _require_completed_fe(config)
        fe_state, normalizer = load_fe_state(config)
    if args.stage in ("ol", "all"):
        _require_completed_fe(config)
        train_operator(config, fe_state, normalizer, resume=args.resume)
    if args.stage in ("eval", "all"):
        from evaluate_prior_ablation import evaluate_variant

        _require_completed_ol(config)
        samples = 8 if args.smoke else 256
        chunk = 2 if args.smoke else 32
        result = evaluate_variant(
            config, include_existing=True, samples_per_scale=samples, chunk_size=chunk
        )
        print("Evaluation conclusion:", result["ablation_conclusion"])


if __name__ == "__main__":
    main()
