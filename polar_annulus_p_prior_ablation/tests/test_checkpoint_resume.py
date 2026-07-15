from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp

from config_polar import PolarAnnulusConfig, make_smoke_config
from data_polar import (
    build_field_normalizer_from_batches,
    sample_batch,
    sample_operator_batch,
)
from train_polar import (
    create_fe_state,
    create_ol_state,
    encode_operator_batch,
    fe_train_step,
    load_fe_inference_state,
    load_training_checkpoint,
    ol_train_step,
    save_normalizer,
    save_training_checkpoint,
)


def _max_difference(left, right) -> float:
    values = [
        float(jnp.max(jnp.abs(a - b)))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right))
    ]
    return max(values, default=0.0)


def test_checkpoint_resume_reproduces_the_next_step(tmp_path) -> None:
    config = make_smoke_config(PolarAnnulusConfig())
    config = replace(config, out_dir=str(tmp_path), run_name="resume")
    key_init, key_train, key_eval = jax.random.split(jax.random.PRNGKey(17), 3)
    state, _ = create_fe_state(config, key_init)

    key_train, norm_key = jax.random.split(key_train)
    norm_batch = sample_batch(norm_key, config)
    normalizer = build_field_normalizer_from_batches(
        [norm_batch], config.normalizer_eps
    )
    key_train, batch1_key = jax.random.split(key_train)
    batch1 = sample_batch(batch1_key, config)
    state1, _ = fe_train_step(state, batch1, normalizer, config)
    save_training_checkpoint(state1, key_train, key_eval, config, "fe", 2.5)
    save_normalizer(normalizer, config.output_dir)

    inference_state, loaded_normalizer = load_fe_inference_state(config)
    assert not hasattr(inference_state, "opt_state")
    assert _max_difference(state1.params, inference_state.params) == 0.0
    assert _max_difference(normalizer, loaded_normalizer) == 0.0

    direct_key, batch2_key = jax.random.split(key_train)
    batch2 = sample_batch(batch2_key, config)
    direct_state, _ = fe_train_step(state1, batch2, normalizer, config)

    template, _ = create_fe_state(config, key_init)
    restored, restored_key, restored_eval_key, elapsed = load_training_checkpoint(
        template, key_train, key_eval, config, "fe"
    )
    restored_key, restored_batch_key = jax.random.split(restored_key)
    restored_batch = sample_batch(restored_batch_key, config)
    resumed_state, _ = fe_train_step(restored, restored_batch, normalizer, config)

    assert elapsed == 2.5
    assert bool(jnp.array_equal(direct_key, restored_key))
    assert bool(jnp.array_equal(key_eval, restored_eval_key))
    assert _max_difference(batch2, restored_batch) == 0.0
    assert _max_difference(direct_state.params, resumed_state.params) == 0.0
    assert _max_difference(direct_state.opt_state, resumed_state.opt_state) == 0.0
    assert int(direct_state.step) == int(resumed_state.step) == 2


def test_operator_checkpoint_resume_reproduces_the_next_step(tmp_path) -> None:
    config = make_smoke_config(PolarAnnulusConfig())
    config = replace(config, out_dir=str(tmp_path), run_name="ol_resume")
    key_fe, key_ol, key_train, key_eval, key_norm = jax.random.split(
        jax.random.PRNGKey(29),
        5,
    )
    normalizer_batch = sample_batch(key_norm, config)
    normalizer = build_field_normalizer_from_batches(
        [normalizer_batch],
        config.normalizer_eps,
    )
    fe_state, _ = create_fe_state(config, key_fe)
    ol_state, _ = create_ol_state(config, key_ol)

    key_train, batch1_key = jax.random.split(key_train)
    batch1 = sample_operator_batch(batch1_key, config)
    tokens1 = encode_operator_batch(fe_state, batch1, normalizer, config)
    state1, _, _ = ol_train_step(
        ol_state,
        tokens1[0],
        tokens1[1],
        batch1.k_values,
        tokens1[2],
    )
    save_training_checkpoint(state1, key_train, key_eval, config, "ol", 3.5)

    direct_key, batch2_key = jax.random.split(key_train)
    batch2 = sample_operator_batch(batch2_key, config)
    tokens2 = encode_operator_batch(fe_state, batch2, normalizer, config)
    direct_state, _, _ = ol_train_step(
        state1,
        tokens2[0],
        tokens2[1],
        batch2.k_values,
        tokens2[2],
    )

    template, _ = create_ol_state(config, key_ol)
    restored, restored_key, restored_eval_key, elapsed = load_training_checkpoint(
        template,
        key_train,
        key_eval,
        config,
        "ol",
    )
    restored_key, restored_batch_key = jax.random.split(restored_key)
    restored_batch = sample_operator_batch(restored_batch_key, config)
    restored_tokens = encode_operator_batch(
        fe_state,
        restored_batch,
        normalizer,
        config,
    )
    resumed_state, _, _ = ol_train_step(
        restored,
        restored_tokens[0],
        restored_tokens[1],
        restored_batch.k_values,
        restored_tokens[2],
    )

    assert elapsed == 3.5
    assert bool(jnp.array_equal(direct_key, restored_key))
    assert bool(jnp.array_equal(key_eval, restored_eval_key))
    assert _max_difference(batch2, restored_batch) == 0.0
    assert _max_difference(direct_state.params, resumed_state.params) == 0.0
    assert _max_difference(direct_state.opt_state, resumed_state.opt_state) == 0.0
    assert int(direct_state.step) == int(resumed_state.step) == 2
