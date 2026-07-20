from __future__ import annotations

from dataclasses import replace
import unittest

import jax
import jax.numpy as jnp

from config_polar import PolarAnnulusConfig
from data_polar import (
    PolarBNNParams,
    build_field_normalizer_from_batches,
    evaluate_polar_prior,
    make_target_cosine_boundary,
    r_to_hat,
    sample_batch,
    sample_bnn_params,
    sample_sigma_pairs,
    theta_from_hat,
    theta_to_hat,
)


class PolarPriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = replace(
            PolarAnnulusConfig(),
            sigma_theta_range=(1.0, 2.0),
            sigma_r_range=(3.0, 4.0),
            sample_size=2,
            hidden_bnn=12,
            theta_size=8,
            radial_size=5,
            random_probe_points=16,
            fe_physics_points=4,
            n_basis=16,
            seq_chunks=4,
            cond_chunks=4,
            transformer_dim=16,
            transformer_heads=4,
        )
        cls.params = sample_bnn_params(
            jax.random.PRNGKey(0),
            batch_size=2,
            sigma_theta=1.0,
            sigma_r=1.0,
            config=cls.config,
        )
        cls.k_values = jnp.asarray([0.7, 1.2])

    def test_coordinate_roundtrip(self):
        theta = jnp.asarray([0.0, 0.3, 2.0 * jnp.pi])
        recovered = theta_from_hat(theta_to_hat(theta))
        self.assertLess(float(jnp.max(jnp.abs(recovered - theta))), 1.0e-6)

    def test_outer_dirichlet_is_exact(self):
        theta = jnp.asarray([0.2, 1.1, 5.5])
        coords = jnp.stack(
            [
                theta_to_hat(theta),
                jnp.ones_like(theta),
            ],
            axis=-1,
        )
        evaluation = evaluate_polar_prior(
            self.params,
            coords,
            self.k_values,
            self.config,
        )
        self.assertEqual(float(jnp.max(jnp.abs(evaluation.p))), 0.0)

    def test_analytic_derivatives_match_autodiff(self):
        theta = jnp.asarray(1.1)
        radius = jnp.asarray(0.63)
        coords = jnp.asarray(
            [[theta_to_hat(theta), r_to_hat(radius, self.config)]]
        )
        evaluation = evaluate_polar_prior(
            self.params,
            coords,
            self.k_values,
            self.config,
        )

        def p_of_r(r):
            point = jnp.asarray(
                [[theta_to_hat(theta), r_to_hat(r, self.config)]]
            )
            return evaluate_polar_prior(
                self.params,
                point,
                self.k_values,
                self.config,
            ).p[0, 0]

        def q_of_r(r):
            point = jnp.asarray(
                [[theta_to_hat(theta), r_to_hat(r, self.config)]]
            )
            return evaluate_polar_prior(
                self.params,
                point,
                self.k_values,
                self.config,
            ).q[0, 0]

        def p_of_theta(t):
            point = jnp.asarray(
                [[theta_to_hat(t), r_to_hat(radius, self.config)]]
            )
            return evaluate_polar_prior(
                self.params,
                point,
                self.k_values,
                self.config,
            ).p[0, 0]

        p_r_ad = jax.grad(p_of_r)(radius)
        q_r_ad = jax.grad(q_of_r)(radius)
        p_theta2_ad = jax.grad(jax.grad(p_of_theta))(theta)

        self.assertLess(float(jnp.abs(p_r_ad - evaluation.q[0, 0])), 2.0e-5)
        self.assertLess(float(jnp.abs(q_r_ad - evaluation.q_r[0, 0])), 2.0e-5)
        self.assertLess(
            float(jnp.abs(p_theta2_ad - evaluation.p_theta2[0, 0])),
            2.0e-5,
        )

    def test_zero_radial_frequency_uses_correct_limit(self):
        w1 = self.params.w1.at[:, 2, :].set(0.0)
        params = PolarBNNParams(w1=w1, b1=self.params.b1, w2=self.params.w2)
        theta = jnp.asarray([0.3, 1.4])
        radius = jnp.asarray([0.2, 0.6])
        coords = jnp.stack(
            [theta_to_hat(theta), r_to_hat(radius, self.config)],
            axis=-1,
        )
        evaluation = evaluate_polar_prior(
            params,
            coords,
            self.k_values,
            self.config,
        )
        expected = (radius[None, :] - self.config.r_outer) * evaluation.q
        self.assertTrue(bool(jnp.all(jnp.isfinite(evaluation.p))))
        self.assertLess(float(jnp.max(jnp.abs(evaluation.p - expected))), 2.0e-5)

    def test_batch_shapes_and_flux_sign(self):
        batch = sample_batch(jax.random.PRNGKey(4), self.config)
        self.assertEqual(
            batch.p_pod.shape,
            (self.config.effective_batch_size, self.config.n_pod),
        )
        self.assertEqual(
            batch.boundary_flux.shape,
            (self.config.effective_batch_size, self.config.theta_size),
        )
        self.assertTrue(bool(jnp.all(jnp.isfinite(batch.f_probe))))
        self.assertEqual(batch.sigma_theta.shape, (self.config.sample_size,))
        self.assertEqual(batch.sigma_r.shape, (self.config.sample_size,))

        target_coords, target_flux = make_target_cosine_boundary(self.config, 1)
        theta = theta_from_hat(target_coords[0, :, 0])
        self.assertLess(
            float(jnp.max(jnp.abs(target_flux[0] - jnp.cos(theta)))),
            1.0e-6,
        )

    def test_sigma_pairs_are_independent_and_inside_ranges(self):
        sigma_theta, sigma_r = sample_sigma_pairs(
            jax.random.PRNGKey(17),
            4096,
            self.config,
        )
        self.assertGreaterEqual(float(jnp.min(sigma_theta)), 1.0)
        self.assertLess(float(jnp.max(sigma_theta)), 2.0)
        self.assertGreaterEqual(float(jnp.min(sigma_r)), 3.0)
        self.assertLess(float(jnp.max(sigma_r)), 4.0)
        correlation = jnp.corrcoef(sigma_theta, sigma_r)[0, 1]
        self.assertLess(abs(float(correlation)), 0.08)

    def test_pressure_normalization_keeps_zero_boundary(self):
        batch = sample_batch(jax.random.PRNGKey(5), self.config)
        normalizer = build_field_normalizer_from_batches([batch])
        self.assertEqual(float(normalizer.mean_p), 0.0)
        self.assertGreater(float(normalizer.std_p), 0.0)


if __name__ == "__main__":
    unittest.main()
