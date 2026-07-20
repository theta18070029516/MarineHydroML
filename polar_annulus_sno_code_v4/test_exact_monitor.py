from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from config_polar import PolarAnnulusConfig
from data_polar import build_field_normalizer_from_batches, sample_batch
from exact_monitor import (
    evaluate_fe_exact,
    evaluate_operator_exact,
    make_exact_benchmark,
    save_exact_monitor_figure,
)
from exact_solution import exact_annulus_fourier_solution
from train_polar import create_fe_state, create_ol_state


class ExactSolutionTests(unittest.TestCase):
    def test_outer_value_and_inner_flux(self):
        theta = np.linspace(0.0, 2.0 * np.pi, 17, endpoint=False)
        outer = exact_annulus_fourier_solution(
            np.ones_like(theta), theta, 1.0
        )
        self.assertLess(float(np.max(np.abs(outer))), 1.0e-12)

        r_inner = 0.2
        step = 1.0e-5
        p_inner = exact_annulus_fourier_solution(
            np.full_like(theta, r_inner), theta, 1.0
        )
        p_next = exact_annulus_fourier_solution(
            np.full_like(theta, r_inner + step), theta, 1.0
        )
        numerical_flux = -(p_next - p_inner) / step
        self.assertLess(
            float(np.max(np.abs(numerical_flux - np.cos(theta)))),
            8.0e-5,
        )


class ExactMonitorTests(unittest.TestCase):
    def test_untrained_monitors_are_finite_and_shape_safe(self):
        config = replace(
            PolarAnnulusConfig(),
            sigma_theta_range=(1.0, 2.0),
            sigma_r_range=(1.0, 2.0),
            sample_size=2,
            prior_generation_chunk_size=1,
            hidden_bnn=8,
            theta_size=8,
            radial_size=4,
            random_probe_points=4,
            normalizer_batches=1,
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
            fe_steps=1,
            ol_steps=1,
            exact_eval_radial_size=4,
            exact_eval_theta_size=8,
            exact_eval_save_figure=False,
        )
        key_batch, key_fe, key_ol = jax.random.split(
            jax.random.PRNGKey(9), 3
        )
        batch = sample_batch(key_batch, config)
        normalizer = build_field_normalizer_from_batches([batch])
        fe_state, _ = create_fe_state(config, key_fe)
        ol_state, _ = create_ol_state(config, key_ol)
        benchmark = make_exact_benchmark(config)

        fe_metrics, fe_pred = evaluate_fe_exact(
            fe_state, normalizer, benchmark, config
        )
        ol_metrics, ol_pred = evaluate_operator_exact(
            ol_state, fe_state, normalizer, benchmark, config
        )

        self.assertEqual(fe_pred.shape, benchmark.p_eval.shape)
        self.assertEqual(ol_pred.shape, benchmark.p_eval.shape)
        self.assertTrue(all(np.isfinite(list(fe_metrics.values()))))
        self.assertTrue(all(np.isfinite(list(ol_metrics.values()))))
        self.assertEqual(fe_metrics["outer_dirichlet_max_abs"], 0.0)
        self.assertEqual(ol_metrics["outer_dirichlet_max_abs"], 0.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            figure_path = save_exact_monitor_figure(
                benchmark,
                fe_pred,
                Path(temp_dir) / "monitor.png",
                "test",
            )
            self.assertGreater(figure_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
