from __future__ import annotations

from dataclasses import replace
import unittest

import jax
import jax.numpy as jnp

from config_polar import PolarAnnulusConfig
from data_polar import build_field_normalizer_from_batches, sample_batch
from train_polar import create_fe_state, fe_train_step


class PhysicsSmokeTests(unittest.TestCase):
    def test_polar_physics_loss_backward(self):
        config = replace(
            PolarAnnulusConfig(),
            sigma_theta_range=(1.0, 2.0),
            sigma_r_range=(1.5, 2.5),
            sample_size=1,
            hidden_bnn=4,
            theta_size=4,
            radial_size=3,
            random_probe_points=2,
            fe_physics_points=1,
            n_basis=4,
            trunk_width=8,
            trunk_depth=2,
            cnn_channels=(2, 2, 2),
            cnn_kernel_size=(3, 3),
            cnn_stride=(1, 1),
            cnn_dense_width=8,
            transformer_dim=4,
            transformer_heads=1,
            transformer_layers=1,
            transformer_mlp_dim=8,
            seq_chunks=1,
            cond_chunks=1,
            fe_phys_weight=0.1,
            fe_steps=1,
            ol_steps=1,
            pool_size=1,
        )
        key_batch, key_init = jax.random.split(jax.random.PRNGKey(2))
        batch = sample_batch(key_batch, config)
        normalizer = build_field_normalizer_from_batches([batch])
        state, _ = create_fe_state(config, key_init)
        state, metrics = fe_train_step(state, batch, normalizer, config)
        self.assertTrue(bool(jnp.isfinite(metrics["physics_loss"])))
        self.assertGreaterEqual(float(metrics["physics_loss"]), 0.0)


if __name__ == "__main__":
    unittest.main()
