from __future__ import annotations

import jax
import jax.numpy as jnp

from config_polar import PolarAnnulusConfig, make_smoke_config
from data_polar import (
    build_field_normalizer_from_batches,
    outer_boundary_coords,
    sample_batch,
)
from models_polar import FunctionEncoder
from train_polar import (
    create_fe_state,
    create_ol_state,
    encode_operator_batch,
    fe_train_step,
    ol_train_step,
)


def _finite(tree) -> bool:
    return all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(tree))


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
    ol_state, loss = ol_train_step(
        ol_state, f_tokens, boundary_tokens, batch.k_values, target_latent
    )
    assert bool(jnp.isfinite(loss))
    assert _finite(ol_state)

    outer = outer_boundary_coords(config)
    decoded = fe_state.apply_fn(
        {"params": fe_state.params},
        jnp.ones((1, config.n_basis), dtype=jnp.float32),
        outer,
        method=FunctionEncoder.reconstruct_p,
    )
    assert float(jnp.max(jnp.abs(decoded))) <= 1.0e-7
