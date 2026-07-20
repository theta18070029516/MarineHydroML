from __future__ import annotations

from dataclasses import replace
import unittest

import jax
import jax.numpy as jnp

from config_polar import PolarAnnulusConfig
from data_polar import (
    build_field_normalizer_from_batches,
    make_condition_tokens,
    make_source_tokens,
    normalize_f,
    normalize_p,
    outer_boundary_coords,
    sample_batch,
)
from models_polar import FunctionEncoder
from train_polar import (
    create_fe_state,
    create_ol_state,
    fe_train_step,
    ol_train_step,
)


class ModelSmokeTests(unittest.TestCase):
    def test_one_fe_and_transformer_step(self):
        config = replace(
            PolarAnnulusConfig(),
            sigma_theta_range=(1.0, 2.0),
            sigma_r_range=(1.5, 2.5),
            sample_size=2,
            hidden_bnn=8,
            theta_size=8,
            radial_size=4,
            random_probe_points=4,
            fe_physics_points=2,
            n_basis=8,
            trunk_width=16,
            trunk_depth=2,
            cnn_channels=(4, 4, 4),
            cnn_kernel_size=(3, 3),
            cnn_stride=(1, 2),
            cnn_dense_width=16,
            transformer_dim=8,
            transformer_heads=2,
            transformer_layers=1,
            transformer_mlp_dim=16,
            seq_chunks=2,
            cond_chunks=2,
            fe_phys_weight=0.0,
            fe_steps=2,
            ol_steps=2,
            pool_size=1,
            exact_eval_radial_size=4,
            exact_eval_theta_size=8,
            exact_eval_save_figure=False,
        )

        key_batch, key_fe, key_ol = jax.random.split(jax.random.PRNGKey(0), 3)
        batch = sample_batch(key_batch, config)
        normalizer = build_field_normalizer_from_batches([batch])

        fe_state, _ = create_fe_state(config, key_fe)
        fe_state, metrics = fe_train_step(
            fe_state,
            batch,
            normalizer,
            config,
        )
        self.assertTrue(bool(jnp.isfinite(metrics["loss"])))

        latent_f = fe_state.apply_fn(
            {"params": fe_state.params},
            normalize_f(batch.f_pod, normalizer),
            method=FunctionEncoder.encode_f,
        )
        latent_p = fe_state.apply_fn(
            {"params": fe_state.params},
            normalize_p(batch.p_pod, normalizer),
            method=FunctionEncoder.encode_p,
        )
        f_tokens = make_source_tokens(latent_f, config)
        boundary_tokens = make_condition_tokens(batch, config)

        ol_state, _ = create_ol_state(config, key_ol)
        ol_state, loss = ol_train_step(
            ol_state,
            f_tokens,
            boundary_tokens,
            batch.k_values,
            latent_p,
        )
        self.assertTrue(bool(jnp.isfinite(loss)))

        p_outer_norm = fe_state.apply_fn(
            {"params": fe_state.params},
            latent_p,
            outer_boundary_coords(config),
            method=FunctionEncoder.reconstruct_p,
        )
        self.assertEqual(float(jnp.max(jnp.abs(p_outer_norm))), 0.0)


if __name__ == "__main__":
    unittest.main()
