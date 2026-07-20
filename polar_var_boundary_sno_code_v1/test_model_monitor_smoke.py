from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from scipy.io import loadmat

from config_varpolar import VarPolarConfig
from data_varpolar import (
    FieldConditionNormalizer,
    make_condition_tokens,
    sample_batch,
    theta_from_hat,
)
from fem_monitor import (
    evaluate_fe_fem,
    evaluate_operator_fem,
    export_fem_manifest,
    make_synthetic_monitor,
)
from models_varpolar import FunctionEncoder
from train_varpolar import (
    create_fe_state,
    create_ol_state,
    encode_operator_batch,
    fe_train_step,
    ol_train_step,
)


def tiny_model_config(tmp_path=None) -> VarPolarConfig:
    output = str(tmp_path) if tmp_path is not None else "./out_test"
    return replace(
        VarPolarConfig(),
        hidden_geom_bnn=12,
        hidden_bnn=12,
        theta_size=8,
        radial_size=8,
        random_probe_points=8,
        sample_size=2,
        ol_sample_size=2,
        prior_generation_chunk_size=1,
        ol_prior_generation_chunk_size=1,
        prior_point_chunk_size=16,
        n_basis=8,
        seq_chunks=2,
        cond_chunks=2,
        trunk_width=16,
        trunk_depth=2,
        cnn_channels=(4, 4, 4),
        cnn_dense_width=16,
        transformer_dim=8,
        transformer_heads=2,
        transformer_layers=1,
        transformer_mlp_dim=16,
        fem_eval_chunk_size=1,
        fem_monitor_size=2,
        fem_eval_theta_size=8,
        fem_eval_radial_size=8,
        out_dir=output,
        run_name="tiny",
    )


def normalizer_from_batch(batch) -> FieldConditionNormalizer:
    floor = 1.0e-4

    def std(values):
        return jnp.maximum(jnp.std(values), floor)

    return FieldConditionNormalizer(
        mean_p=jnp.asarray(0.0),
        std_p=std(batch.p_pod),
        mean_f=jnp.mean(batch.f_pod),
        std_f=std(batch.f_pod),
        mean_h=jnp.mean(batch.boundary_h),
        std_h=std(batch.boundary_h),
        mean_g=jnp.mean(batch.boundary_load),
        std_g=std(batch.boundary_load),
    )


def test_five_component_condition_features_and_model_updates():
    config = tiny_model_config()
    key_batch, key_fe, key_ol = jax.random.split(jax.random.PRNGKey(10), 3)
    batch = sample_batch(key_batch, config)
    normalizer = normalizer_from_batch(batch)
    tokens = make_condition_tokens(batch, normalizer, config)
    assert tokens.shape == (2, config.cond_chunks, config.cond_chunk_width)
    assert config.cond_chunk_width == 5 * config.boundary_chunk_size

    features = tokens.reshape(2, config.theta_size, 5)
    theta = theta_from_hat(batch.boundary_coords[:, 0])
    np.testing.assert_allclose(features[0, :, 0], jnp.sin(theta), atol=2e-6)
    np.testing.assert_allclose(features[0, :, 1], jnp.cos(theta), atol=2e-6)
    np.testing.assert_allclose(
        features[:, :, 2],
        (batch.boundary_a - config.geom_base) / config.geom_amp,
        atol=2e-6,
    )

    fe_state, _ = create_fe_state(config, key_fe)
    before_fe = jax.tree_util.tree_leaves(fe_state.params)[0]
    fe_state, metrics = fe_train_step(fe_state, batch, normalizer, config)
    jax.block_until_ready(metrics["loss"])
    after_fe = jax.tree_util.tree_leaves(fe_state.params)[0]
    assert np.isfinite(float(metrics["loss"]))
    assert not np.array_equal(np.asarray(before_fe), np.asarray(after_fe))

    latent = fe_state.apply_fn(
        {"params": fe_state.params},
        batch.p_pod / normalizer.std_p,
        method=FunctionEncoder.encode_p,
    )
    outer = fe_state.apply_fn(
        {"params": fe_state.params},
        latent,
        jnp.stack([batch.boundary_coords[:, 0], jnp.ones(config.theta_size)], axis=-1),
        method=FunctionEncoder.reconstruct_p,
    )
    np.testing.assert_allclose(outer, 0.0, atol=1e-7)

    f_tokens, boundary_tokens, target_latent = encode_operator_batch(
        fe_state, batch, normalizer, config
    )
    ol_state, _ = create_ol_state(config, key_ol)
    before_ol = jax.tree_util.tree_leaves(ol_state.params)[0]
    ol_state, loss = ol_train_step(
        ol_state,
        f_tokens,
        boundary_tokens,
        batch.k_values,
        target_latent,
    )
    jax.block_until_ready(loss)
    after_ol = jax.tree_util.tree_leaves(ol_state.params)[0]
    assert np.isfinite(float(loss))
    assert not np.array_equal(np.asarray(before_ol), np.asarray(after_ol))


def test_fem_monitor_chunked_metrics_are_finite():
    config = tiny_model_config()
    key_batch, key_fe, key_ol = jax.random.split(jax.random.PRNGKey(11), 3)
    batch = sample_batch(key_batch, config)
    normalizer = normalizer_from_batch(batch)
    fe_state, _ = create_fe_state(config, key_fe)
    ol_state, _ = create_ol_state(config, key_ol)
    monitor = make_synthetic_monitor(config, batch_size=2)

    fe_aggregate, fe_cases, fe_prediction = evaluate_fe_fem(
        fe_state, normalizer, monitor, config
    )
    ol_aggregate, ol_cases, ol_prediction = evaluate_operator_fem(
        ol_state, fe_state, normalizer, monitor, config
    )
    assert fe_prediction.shape == monitor.p_eval.shape
    assert ol_prediction.shape == monitor.p_eval.shape
    assert all(np.isfinite(value) for value in fe_aggregate.values())
    assert all(np.isfinite(value) for value in ol_aggregate.values())
    assert all(np.asarray(value).shape == (2,) for value in fe_cases.values())
    assert all(np.asarray(value).shape == (2,) for value in ol_cases.values())
    assert "p_area_relative_l2_mean" in fe_aggregate
    assert "latent_relative_l2_p95" in ol_aggregate


def test_fixed_fem_manifest_is_reproducible_and_separate(tmp_path):
    config = tiny_model_config(tmp_path)
    first = export_fem_manifest(config, tmp_path / "first.mat")
    second = export_fem_manifest(config, tmp_path / "second.mat")
    data_first = loadmat(first)
    data_second = loadmat(second)
    for name in ("geometry_w1", "geometry_b1", "geometry_w2", "k_values"):
        np.testing.assert_array_equal(data_first[name], data_second[name])
    k_values = data_first["k_values"].reshape(-1)
    assert k_values.shape == (config.fem_monitor_size,)
    assert np.all(k_values >= config.k_min)
    assert np.all(k_values < config.k_max)
    assert data_first["geometry_w1"].shape == (
        config.fem_monitor_size,
        2,
        config.hidden_geom_bnn,
    )
