from __future__ import annotations

import csv
from dataclasses import replace

import jax
import jax.numpy as jnp

from config_polar import PolarAnnulusConfig, make_smoke_config
from data_polar import (
    build_field_normalizer_from_batches,
    outer_boundary_coords,
    sample_batch,
    sample_operator_batch,
)
from models_polar import FunctionEncoder
from train_polar import (
    build_exact_solution_benchmark,
    create_fe_state,
    create_ol_state,
    encode_operator_batch,
    evaluate_exact_solution_benchmark,
    fe_train_step,
    ol_train_step,
    pressure_relative_l2_from_latent,
    train_operator,
)


def _finite(tree) -> bool:
    return all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(tree))


def test_operator_batch_matches_full_sampler_on_training_fields() -> None:
    config = make_smoke_config(PolarAnnulusConfig())
    key = jax.random.PRNGKey(23)
    full_batch = sample_batch(key, config)
    operator_batch = sample_operator_batch(key, config)
    for name in (
        "boundary_coords",
        "boundary_flux",
        "pod_coords",
        "p_pod",
        "f_pod",
        "k_values",
    ):
        assert bool(
            jnp.array_equal(
                getattr(full_batch, name),
                getattr(operator_batch, name),
            )
        ), name


def test_fe_and_operator_single_step_are_finite() -> None:
    config = make_smoke_config(PolarAnnulusConfig())
    key_fe, key_batch, key_ol = jax.random.split(jax.random.PRNGKey(3), 3)
    batch = sample_batch(key_batch, config)
    normalizer = build_field_normalizer_from_batches([batch], config.normalizer_eps)
    fe_state, _ = create_fe_state(config, key_fe)
    fe_state, metrics = fe_train_step(fe_state, batch, normalizer, config)
    assert bool(jnp.isfinite(metrics["loss"]))
    assert _finite(fe_state)

    f_tokens, boundary_tokens, target_latent = encode_operator_batch(
        fe_state, batch, normalizer, config
    )
    ol_state, _ = create_ol_state(config, key_ol)
    ol_state, loss, prediction = ol_train_step(
        ol_state, f_tokens, boundary_tokens, batch.k_values, target_latent
    )
    assert bool(jnp.isfinite(loss))
    assert prediction.shape == target_latent.shape
    assert _finite(prediction)
    assert _finite(ol_state)

    train_error = pressure_relative_l2_from_latent(
        fe_state.params,
        fe_state.apply_fn,
        prediction,
        batch.pod_coords,
        batch.p_pod,
        normalizer,
    )
    assert bool(jnp.isfinite(train_error))

    benchmark = build_exact_solution_benchmark(config)
    exact_metrics = evaluate_exact_solution_benchmark(
        config, fe_state, ol_state, normalizer, benchmark
    )
    assert bool(jnp.isfinite(exact_metrics["grid_relative_l2"]))
    assert bool(jnp.isfinite(exact_metrics["area_weighted_relative_l2"]))

    outer = outer_boundary_coords(config)
    decoded = fe_state.apply_fn(
        {"params": fe_state.params},
        jnp.ones((1, config.n_basis), dtype=jnp.float32),
        outer,
        method=FunctionEncoder.reconstruct_p,
    )
    assert float(jnp.max(jnp.abs(decoded))) <= 1.0e-7


def test_low_memory_operator_loop_uses_one_batch_per_step(
    tmp_path,
    monkeypatch,
) -> None:
    config = make_smoke_config(PolarAnnulusConfig())
    config = replace(
        config,
        out_dir=str(tmp_path),
        run_name="low_memory_ol",
        ol_steps=2,
        ol_log_interval=1,
        checkpoint_interval=2,
    )
    key_fe, key_norm = jax.random.split(jax.random.PRNGKey(91))
    normalizer_batch = sample_batch(key_norm, config)
    normalizer = build_field_normalizer_from_batches(
        [normalizer_batch],
        config.normalizer_eps,
    )
    fe_state, _ = create_fe_state(config, key_fe)

    calls = 0
    original_sampler = sample_operator_batch

    def counted_sampler(key, sampler_config):
        nonlocal calls
        calls += 1
        return original_sampler(key, sampler_config)

    monkeypatch.setattr("train_polar.sample_operator_batch", counted_sampler)
    train_operator(config, fe_state, normalizer, resume=False)

    assert calls == config.ol_steps
    history_path = config.output_dir / "operator_training_history.csv"
    with history_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == config.ol_steps
    assert set(rows[-1]) == {
        "step",
        "samples_seen",
        "loss",
        "in_distribution_relative_l2",
        "exact_relative_l2",
        "elapsed_seconds",
    }
