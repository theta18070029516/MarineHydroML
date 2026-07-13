from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp

from config_polar import PolarAnnulusConfig, make_smoke_config
from data_polar import build_field_normalizer_from_batches, sample_batch
from train_polar import (
    create_fe_state,
    fe_train_step,
    load_training_checkpoint,
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
