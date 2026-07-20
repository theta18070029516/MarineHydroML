from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any
import csv
import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training import train_state
from tqdm.auto import trange

from config_polar import PolarAnnulusConfig
from data_polar import (
    FieldNormalizer,
    SampleBatch,
    build_field_normalizer_online,
    denormalize_f,
    denormalize_p,
    inner_boundary_coords,
    make_condition_tokens,
    make_condition_tokens_from_arrays,
    make_polar_grid,
    make_source_tokens,
    make_target_cosine_boundary,
    normalize_f,
    normalize_p,
    r_from_hat,
    sample_batch,
)
from exact_monitor import (
    evaluate_fe_exact,
    evaluate_operator_exact,
    make_exact_benchmark,
    save_exact_monitor_figure,
)
from models_polar import FunctionEncoder, OperatorTransformer


Array = jax.Array


class TrainState(train_state.TrainState):
    pass


def relative_l2(pred: Array, ref: Array) -> Array:
    denominator = jnp.maximum(jnp.linalg.norm(ref, axis=-1), 1.0e-12)
    return jnp.linalg.norm(pred - ref, axis=-1) / denominator


def save_params(params: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.to_bytes(params))


def save_normalizer(normalizer: FieldNormalizer, out_dir: Path) -> None:
    jnp.save(out_dir / "norm_mean_p.npy", normalizer.mean_p)
    jnp.save(out_dir / "norm_std_p.npy", normalizer.std_p)
    jnp.save(out_dir / "norm_mean_f.npy", normalizer.mean_f)
    jnp.save(out_dir / "norm_std_f.npy", normalizer.std_f)


def load_normalizer(out_dir: Path) -> FieldNormalizer:
    return FieldNormalizer(
        mean_p=jnp.load(out_dir / "norm_mean_p.npy"),
        std_p=jnp.load(out_dir / "norm_std_p.npy"),
        mean_f=jnp.load(out_dir / "norm_mean_f.npy"),
        std_f=jnp.load(out_dir / "norm_std_f.npy"),
    )


FE_HISTORY_COLUMNS = (
    "step",
    "samples_seen",
    "train_loss",
    "train_data_loss",
    "train_physics_loss",
    "eval_p_relative_l2",
    "eval_f_relative_l2",
    "elapsed_seconds",
)


FE_EXACT_HISTORY_COLUMNS = (
    "step",
    "samples_seen",
    "p_grid_relative_l2",
    "p_area_relative_l2",
    "p_rmse",
    "p_relative_linf",
    "f_zero_rmse",
    "f_zero_max_abs",
    "outer_dirichlet_max_abs",
    "elapsed_seconds",
)


OPERATOR_HISTORY_COLUMNS = (
    "step",
    "samples_seen",
    "train_latent_mse",
    "eval_latent_mse",
    "eval_latent_relative_l2",
    "eval_physical_relative_l2",
    "elapsed_seconds",
)


OPERATOR_EXACT_HISTORY_COLUMNS = (
    "step",
    "samples_seen",
    "p_grid_relative_l2",
    "p_area_relative_l2",
    "p_rmse",
    "p_relative_linf",
    "latent_relative_l2",
    "outer_dirichlet_max_abs",
    "inner_flux_relative_l2",
    "elapsed_seconds",
)


def save_metric_history(
    history: list[dict[str, float | int]],
    out_dir: Path,
    stem: str,
    columns: tuple[str, ...],
) -> tuple[Path, Path]:
    """Atomically save a metric history as CSV and NPZ."""
    if not history:
        raise ValueError("history must contain at least one record.")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    csv_temp_path = out_dir / f"{stem}.tmp.csv"
    with csv_temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)
    csv_temp_path.replace(csv_path)

    npz_path = out_dir / f"{stem}.npz"
    npz_temp_path = out_dir / f"{stem}.tmp.npz"
    arrays = {
        name: np.asarray([record[name] for record in history])
        for name in columns
    }
    np.savez(npz_temp_path, **arrays)
    npz_temp_path.replace(npz_path)
    return csv_path, npz_path


def save_operator_history(
    history: list[dict[str, float | int]],
    out_dir: Path,
) -> tuple[Path, Path]:
    return save_metric_history(
        history,
        out_dir,
        "operator_training_history",
        OPERATOR_HISTORY_COLUMNS,
    )


def create_fe_state(
    config: PolarAnnulusConfig,
    key: Array,
) -> tuple[TrainState, FunctionEncoder]:
    model = FunctionEncoder(config)
    dummy_field = jnp.ones((1, config.n_pod), dtype=jnp.float32)
    dummy_coords = jnp.ones((config.n_pod, 2), dtype=jnp.float32)
    variables = model.init(
        key,
        dummy_field,
        dummy_field,
        dummy_coords,
        method=FunctionEncoder.init_all,
    )
    schedule = optax.cosine_decay_schedule(
        config.fe_lr,
        config.fe_steps,
        alpha=1.0e-3,
    )
    optimizer = optax.adamw(
        schedule,
        weight_decay=config.weight_decay,
        b1=config.fe_b1,
        b2=config.fe_b2,
    )
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer,
    )
    return state, model


def create_ol_state(
    config: PolarAnnulusConfig,
    key: Array,
) -> tuple[TrainState, OperatorTransformer]:
    model = OperatorTransformer(config)
    dummy_f = jnp.ones(
        (1, config.seq_chunks, config.seq_chunk_width),
        dtype=jnp.float32,
    )
    dummy_boundary = jnp.ones(
        (1, config.cond_chunks, config.cond_chunk_width),
        dtype=jnp.float32,
    )
    dummy_k = jnp.ones((1,), dtype=jnp.float32)
    variables = model.init(key, dummy_f, dummy_boundary, dummy_k)
    schedule = optax.cosine_decay_schedule(
        config.ol_lr,
        config.ol_steps,
        alpha=1.0e-4,
    )
    optimizer = optax.adamw(
        schedule,
        weight_decay=config.weight_decay,
    )
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer,
    )
    return state, model


def compute_trunk_polar_operator_basis(
    params: Any,
    state: TrainState,
    coords_hat: Array,
    config: PolarAnnulusConfig,
) -> tuple[Array, Array]:
    """Return basis and physical polar Laplacian of every basis function.

    coords_hat columns are [theta_hat, r_hat]. Since both mappings are affine,

        d/dr     = (2/L) d/dr_hat,
        d/dtheta = (1/pi) d/dtheta_hat.
    """

    def trunk_single(coord):
        return state.apply_fn(
            {"params": params},
            coord[None, :],
            method=FunctionEncoder.trunk_p_basis,
        )[0]

    basis = state.apply_fn(
        {"params": params},
        coords_hat,
        method=FunctionEncoder.trunk_p_basis,
    )
    jacobian = jax.vmap(jax.jacfwd(trunk_single))(coords_hat)
    hessian = jax.vmap(jax.jacfwd(jax.jacfwd(trunk_single)))(coords_hat)

    radius = r_from_hat(coords_hat[:, 1], config)
    basis_r = config.drhat_dr * jacobian[..., 1]
    basis_rr = (config.drhat_dr**2) * hessian[..., 1, 1]
    basis_theta2 = (
        config.dthetahat_dtheta**2
    ) * hessian[..., 0, 0]

    laplacian = (
        basis_rr
        + basis_r / radius[:, None]
        + basis_theta2 / (radius[:, None] ** 2)
    )
    return basis, laplacian


def fe_loss_fn(
    params: Any,
    state: TrainState,
    batch: SampleBatch,
    normalizer: FieldNormalizer,
    config: PolarAnnulusConfig,
) -> tuple[Array, dict[str, Array]]:
    p_pod_norm = normalize_p(batch.p_pod, normalizer)
    f_pod_norm = normalize_f(batch.f_pod, normalizer)
    p_probe_norm = normalize_p(batch.p_probe, normalizer)
    f_probe_norm = normalize_f(batch.f_probe, normalizer)

    latent_p = state.apply_fn(
        {"params": params},
        p_pod_norm,
        method=FunctionEncoder.encode_p,
    )
    latent_f = state.apply_fn(
        {"params": params},
        f_pod_norm,
        method=FunctionEncoder.encode_f,
    )

    pred_p_norm = state.apply_fn(
        {"params": params},
        latent_p,
        batch.probe_coords,
        method=FunctionEncoder.reconstruct_p,
    )
    pred_f_norm = state.apply_fn(
        {"params": params},
        latent_f,
        batch.probe_coords,
        method=FunctionEncoder.reconstruct_f,
    )

    loss_p = jnp.mean((pred_p_norm - p_probe_norm) ** 2)
    loss_f = jnp.mean((pred_f_norm - f_probe_norm) ** 2)
    data_loss = loss_p + loss_f

    if config.fe_phys_weight > 0.0:
        physics_coords = batch.probe_coords[: config.fe_physics_points]
        physics_f = batch.f_probe[:, : config.fe_physics_points]
        basis, lap_basis = compute_trunk_polar_operator_basis(
            params,
            state,
            physics_coords,
            config,
        )
        p_pred_norm_for_physics = jnp.einsum(
            "bd,nd->bn",
            latent_p,
            basis,
        ) / jnp.sqrt(config.n_basis)
        lap_p_norm = jnp.einsum(
            "bd,nd->bn",
            latent_p,
            lap_basis,
        ) / jnp.sqrt(config.n_basis)

        p_pred_phys = denormalize_p(p_pred_norm_for_physics, normalizer)
        lap_p_phys = normalizer.std_p * lap_p_norm
        residual = (
            lap_p_phys
            - (batch.k_values[:, None] ** 2) * p_pred_phys
            - physics_f
        )
        residual_norm = residual / normalizer.std_f
        physics_loss = jnp.mean(residual_norm**2)
    else:
        physics_loss = jnp.asarray(0.0, dtype=data_loss.dtype)

    total = data_loss + config.fe_phys_weight * physics_loss
    metrics = {
        "loss": total,
        "data_loss": data_loss,
        "loss_p": loss_p,
        "loss_f": loss_f,
        "physics_loss": physics_loss,
    }
    return total, metrics


@partial(jax.jit, static_argnames=("config",))
def fe_train_step(
    state: TrainState,
    batch: SampleBatch,
    normalizer: FieldNormalizer,
    config: PolarAnnulusConfig,
) -> tuple[TrainState, dict[str, Array]]:
    (_, metrics), gradients = jax.value_and_grad(
        fe_loss_fn,
        has_aux=True,
    )(state.params, state, batch, normalizer, config)
    state = state.apply_gradients(grads=gradients)
    return state, metrics


@partial(jax.jit, static_argnames=("config",))
def fe_eval_step(
    state: TrainState,
    batch: SampleBatch,
    normalizer: FieldNormalizer,
    config: PolarAnnulusConfig,
) -> tuple[Array, Array]:
    p_pod_norm = normalize_p(batch.p_pod, normalizer)
    f_pod_norm = normalize_f(batch.f_pod, normalizer)
    latent_p = state.apply_fn(
        {"params": state.params},
        p_pod_norm,
        method=FunctionEncoder.encode_p,
    )
    latent_f = state.apply_fn(
        {"params": state.params},
        f_pod_norm,
        method=FunctionEncoder.encode_f,
    )
    p_pred_norm = state.apply_fn(
        {"params": state.params},
        latent_p,
        batch.probe_coords,
        method=FunctionEncoder.reconstruct_p,
    )
    f_pred_norm = state.apply_fn(
        {"params": state.params},
        latent_f,
        batch.probe_coords,
        method=FunctionEncoder.reconstruct_f,
    )
    p_pred = denormalize_p(p_pred_norm, normalizer)
    f_pred = denormalize_f(f_pred_norm, normalizer)
    return (
        jnp.mean(relative_l2(p_pred, batch.p_probe)),
        jnp.mean(relative_l2(f_pred, batch.f_probe)),
    )


def ol_loss_fn(
    params: Any,
    state: TrainState,
    f_tokens: Array,
    boundary_tokens: Array,
    k_values: Array,
    target_latent_p: Array,
) -> Array:
    pred_latent_p = state.apply_fn(
        {"params": params},
        f_tokens,
        boundary_tokens,
        k_values,
    )
    return jnp.mean((pred_latent_p - target_latent_p) ** 2)


@jax.jit
def ol_train_step(
    state: TrainState,
    f_tokens: Array,
    boundary_tokens: Array,
    k_values: Array,
    target_latent_p: Array,
) -> tuple[TrainState, Array]:
    loss, gradients = jax.value_and_grad(ol_loss_fn)(
        state.params,
        state,
        f_tokens,
        boundary_tokens,
        k_values,
        target_latent_p,
    )
    state = state.apply_gradients(grads=gradients)
    return state, loss


@partial(jax.jit, static_argnames=("ol_apply_fn", "fe_apply_fn"))
def ol_eval_step(
    ol_params: Any,
    ol_apply_fn: Any,
    fe_params: Any,
    fe_apply_fn: Any,
    f_tokens: Array,
    boundary_tokens: Array,
    k_values: Array,
    target_latent_p: Array,
    probe_coords: Array,
    target_p_probe: Array,
    normalizer: FieldNormalizer,
) -> dict[str, Array]:
    """Evaluate the operator in latent space and decoded physical space.

    The physical metric is evaluated at fresh random probe points rather than
    at the encoder's regular input grid. This measures continuous pressure-field
    reconstruction on unseen coordinates.
    """
    pred_latent_p = ol_apply_fn(
        {"params": ol_params},
        f_tokens,
        boundary_tokens,
        k_values,
    )
    latent_mse = jnp.mean((pred_latent_p - target_latent_p) ** 2)
    latent_relative_l2 = jnp.mean(
        relative_l2(pred_latent_p, target_latent_p)
    )

    pred_p_probe_norm = fe_apply_fn(
        {"params": fe_params},
        pred_latent_p,
        probe_coords,
        method=FunctionEncoder.reconstruct_p,
    )
    pred_p_probe = denormalize_p(pred_p_probe_norm, normalizer)
    physical_relative_l2 = jnp.mean(
        relative_l2(pred_p_probe, target_p_probe)
    )

    return {
        "latent_mse": latent_mse,
        "latent_relative_l2": latent_relative_l2,
        "physical_relative_l2": physical_relative_l2,
    }


def encode_operator_batch(
    fe_state: TrainState,
    batch: SampleBatch,
    normalizer: FieldNormalizer,
    config: PolarAnnulusConfig,
) -> tuple[Array, Array, Array]:
    latent_f = fe_state.apply_fn(
        {"params": fe_state.params},
        normalize_f(batch.f_pod, normalizer),
        method=FunctionEncoder.encode_f,
    )
    target_latent_p = fe_state.apply_fn(
        {"params": fe_state.params},
        normalize_p(batch.p_pod, normalizer),
        method=FunctionEncoder.encode_p,
    )
    return (
        make_source_tokens(latent_f, config),
        make_condition_tokens(batch, config),
        target_latent_p,
    )


def train_fe(
    config: PolarAnnulusConfig,
) -> tuple[TrainState, FieldNormalizer]:
    """Train FE with random-distribution and analytic-solution monitoring."""

    output_dir = config.output_dir
    config.save_json(output_dir / "config_fe.json")
    config.save_json(output_dir / "config.json")

    master_key = jax.random.PRNGKey(config.seed)
    key_norm, key_init, key_train, key_eval = jax.random.split(
        master_key,
        4,
    )

    # 计算并固定归一化统计量
    normalizer, _ = build_field_normalizer_online(
        key_norm,
        config,
    )

    # 确保异步计算已经完成
    jax.block_until_ready(normalizer.std_f)

    print(
        "[Normalizer] 计算完成："
        f" mean_p={float(normalizer.mean_p):.6e},"
        f" std_p={float(normalizer.std_p):.6e},"
        f" mean_f={float(normalizer.mean_f):.6e},"
        f" std_f={float(normalizer.std_f):.6e}"
    )

    save_normalizer(
        normalizer,
        config.output_dir,
    )

    print(f"[Normalizer] 已保存至：{output_dir}")

    state, _ = create_fe_state(
        config,
        key_init,
    )
    exact_benchmark = make_exact_benchmark(config)

    eval_probe_points = min(
        config.random_probe_points,
        config.fe_eval_probe_points,
    )
    eval_sample_size = min(config.sample_size, config.fe_eval_sample_size)
    eval_config = replace(
        config,
        sample_size=eval_sample_size,
        prior_generation_chunk_size=min(
            config.prior_generation_chunk_size,
            eval_sample_size,
        ),
        random_probe_points=eval_probe_points,
        fe_physics_points=min(config.fe_physics_points, eval_probe_points),
    )
    print(
        "[FE eval] "
        f"batch={eval_config.effective_batch_size}, "
        f"probe_points={eval_config.random_probe_points}, "
        f"exact_every={config.fe_exact_eval_interval}"
    )

    history: list[dict[str, float | int]] = []
    exact_history: list[dict[str, float | int]] = []
    start_time = time.perf_counter()
    progress = trange(
        config.fe_steps,
        desc="Function encoder training",
        unit="step",
        dynamic_ncols=True,
    )

    for step_index in progress:
        step = step_index + 1
        key_train, batch_key = jax.random.split(key_train)
        train_batch = sample_batch(batch_key, config)
        state, train_metrics = fe_train_step(
            state, train_batch, normalizer, config
        )

        should_log = (
            step == 1
            or step % config.fe_log_interval == 0
            or step == config.fe_steps
        )
        should_exact = (
            step == 1
            or step % config.fe_exact_eval_interval == 0
            or step == config.fe_steps
        )
        if should_log or should_exact:
            # Complete the train step and release its large sample tensors before
            # allocating either monitoring workload.
            jax.block_until_ready(train_metrics["loss"])

        if should_log:
            train_values = {
                "train_loss": float(train_metrics["loss"]),
                "train_data_loss": float(train_metrics["data_loss"]),
                "train_physics_loss": float(train_metrics["physics_loss"]),
            }

        del train_batch, train_metrics

        if should_log:
            key_eval, eval_key = jax.random.split(key_eval)
            eval_batch = sample_batch(eval_key, eval_config)
            err_p, err_f = fe_eval_step(
                state, eval_batch, normalizer, eval_config
            )
            jax.block_until_ready(err_f)
            record = {
                "step": step,
                "samples_seen": step * config.effective_batch_size,
                **train_values,
                "eval_p_relative_l2": float(err_p),
                "eval_f_relative_l2": float(err_f),
                "elapsed_seconds": float(time.perf_counter() - start_time),
            }
            history.append(record)
            save_metric_history(
                history,
                output_dir,
                "fe_training_history",
                FE_HISTORY_COLUMNS,
            )
            progress.set_postfix(
                loss=f"{record['train_loss']:.3e}",
                p_rl2=f"{record['eval_p_relative_l2']:.3e}",
                refresh=False,
            )
            progress.write(
                f"[FE {step:07d}] "
                f"loss={record['train_loss']:.4e} "
                f"data={record['train_data_loss']:.4e} "
                f"phys={record['train_physics_loss']:.4e} "
                f"random_RL2(P)={record['eval_p_relative_l2']:.4e} "
                f"random_RL2(f)={record['eval_f_relative_l2']:.4e}"
            )
            del eval_batch, err_p, err_f

        if should_exact:
            exact_metrics, exact_pred = evaluate_fe_exact(
                state,
                normalizer,
                exact_benchmark,
                config,
            )
            exact_record = {
                "step": step,
                "samples_seen": step * config.effective_batch_size,
                **exact_metrics,
                "elapsed_seconds": float(time.perf_counter() - start_time),
            }
            exact_history.append(exact_record)
            save_metric_history(
                exact_history,
                output_dir,
                "fe_exact_history",
                FE_EXACT_HISTORY_COLUMNS,
            )
            progress.write(
                f"[FE exact {step:07d}] "
                f"area_RL2(P)={exact_record['p_area_relative_l2']:.4e} "
                f"RMSE(P)={exact_record['p_rmse']:.4e} "
                f"RMSE(f=0)={exact_record['f_zero_rmse']:.4e}"
            )
            if config.exact_eval_save_figure:
                save_exact_monitor_figure(
                    exact_benchmark,
                    exact_pred,
                    output_dir
                    / "exact_monitor"
                    / f"fe_step_{step:07d}.png",
                    f"FE analytic reconstruction at step {step}",
                )
            del exact_pred

        if step % config.fe_checkpoint_interval == 0:
            save_params(
                state.params,
                output_dir / "fe_params_latest.msgpack",
            )

    save_params(state.params, output_dir / "fe_params.msgpack")
    save_params(state.params, output_dir / "fe_params_latest.msgpack")
    if history:
        save_metric_history(
            history,
            output_dir,
            "fe_training_history",
            FE_HISTORY_COLUMNS,
        )
    if exact_history:
        save_metric_history(
            exact_history,
            output_dir,
            "fe_exact_history",
            FE_EXACT_HISTORY_COLUMNS,
        )

    return state, normalizer


def train_operator(
    config: PolarAnnulusConfig,
    fe_state: TrainState,
    normalizer: FieldNormalizer,
) -> TrainState:
    """Train OL with independent random and analytic-solution monitoring."""
    output_dir = config.output_dir
    config.save_json(output_dir / "config_ol.json")

    master_key = jax.random.PRNGKey(config.seed + 10_000)
    key_init, key_train, key_eval = jax.random.split(master_key, 3)
    state, _ = create_ol_state(config, key_init)

    latest_params_path = output_dir / "ol_params_latest.msgpack"
    if latest_params_path.is_file():
        loaded_params = serialization.from_bytes(
            state.params,
            latest_params_path.read_bytes(),
        )
        state = state.replace(params=loaded_params)
        print(
            f"[OL resume] 已加载模型参数：{latest_params_path}；"
            "优化器状态和学习率进度将重新开始。"
        )
    else:
        print(
            f"[OL resume] 未找到 {latest_params_path}，将从随机初始化开始训练。"
        )

    exact_benchmark = make_exact_benchmark(config)

    # Monitoring must not duplicate the full optimization batch in accelerator
    # memory. Keep the same physical distribution while reducing only its batch
    # dimension and the number of continuous pressure probes.
    eval_probe_points = min(
        config.random_probe_points,
        config.ol_eval_probe_points,
    )
    eval_sample_size = min(config.sample_size, config.ol_eval_sample_size)
    eval_config = replace(
        config,
        sample_size=eval_sample_size,
        prior_generation_chunk_size=min(
            config.prior_generation_chunk_size,
            eval_sample_size,
        ),
        random_probe_points=eval_probe_points,
        fe_physics_points=min(config.fe_physics_points, eval_probe_points),
    )
    print(
        "[OL eval] "
        f"batch={eval_config.effective_batch_size}, "
        f"probe_points={eval_config.random_probe_points}, "
        f"exact_every={config.ol_exact_eval_interval}"
    )

    history: list[dict[str, float | int]] = []
    exact_history: list[dict[str, float | int]] = []
    start_time = time.perf_counter()
    progress = trange(
        config.ol_steps,
        desc="Operator training",
        unit="step",
        dynamic_ncols=True,
    )

    for step_index in progress:
        step = step_index + 1

        # Draw a new, physically consistent PI-sampler batch every step.
        key_train, batch_key = jax.random.split(key_train)
        train_batch = sample_batch(batch_key, config)
        f_tokens, boundary_tokens, target_latent_p = encode_operator_batch(
            fe_state,
            train_batch,
            normalizer,
            config,
        )
        state, train_loss = ol_train_step(
            state,
            f_tokens,
            boundary_tokens,
            train_batch.k_values,
            target_latent_p,
        )

        should_log = (
            step == 1
            or step % config.ol_log_interval == 0
            or step == config.ol_steps
        )
        should_exact = (
            step == 1
            or step % config.ol_exact_eval_interval == 0
            or step == config.ol_steps
        )
        if should_log or should_exact:
            # Complete the optimizer step before releasing the full train batch.
            jax.block_until_ready(train_loss)
        if should_log:
            train_loss_value = float(train_loss)
        del train_batch, f_tokens, boundary_tokens, target_latent_p, train_loss

        if should_log:
            # Use a separate random stream so evaluation never reuses the
            # current optimization batch.
            key_eval, eval_batch_key = jax.random.split(key_eval)
            eval_batch = sample_batch(eval_batch_key, eval_config)
            (
                eval_f_tokens,
                eval_boundary_tokens,
                eval_target_latent_p,
            ) = encode_operator_batch(
                fe_state,
                eval_batch,
                normalizer,
                eval_config,
            )
            eval_metrics = ol_eval_step(
                state.params,
                state.apply_fn,
                fe_state.params,
                fe_state.apply_fn,
                eval_f_tokens,
                eval_boundary_tokens,
                eval_batch.k_values,
                eval_target_latent_p,
                eval_batch.probe_coords,
                eval_batch.p_probe,
                normalizer,
            )
            jax.block_until_ready(eval_metrics["physical_relative_l2"])

            record = {
                "step": step,
                "samples_seen": step * config.effective_batch_size,
                "train_latent_mse": train_loss_value,
                "eval_latent_mse": float(eval_metrics["latent_mse"]),
                "eval_latent_relative_l2": float(
                    eval_metrics["latent_relative_l2"]
                ),
                "eval_physical_relative_l2": float(
                    eval_metrics["physical_relative_l2"]
                ),
                "elapsed_seconds": float(time.perf_counter() - start_time),
            }
            history.append(record)
            save_operator_history(history, output_dir)

            progress.set_postfix(
                loss=f"{record['train_latent_mse']:.3e}",
                z_rl2=f"{record['eval_latent_relative_l2']:.3e}",
                p_rl2=f"{record['eval_physical_relative_l2']:.3e}",
                refresh=False,
            )
            progress.write(
                f"[OL {step:07d}] "
                f"train_MSE={record['train_latent_mse']:.4e} "
                f"eval_MSE={record['eval_latent_mse']:.4e} "
                f"RL2(z_P)={record['eval_latent_relative_l2']:.4e} "
                f"RL2(P@probe)={record['eval_physical_relative_l2']:.4e}"
            )
            del (
                eval_batch,
                eval_f_tokens,
                eval_boundary_tokens,
                eval_target_latent_p,
                eval_metrics,
            )

        if should_exact:
            exact_metrics, exact_pred = evaluate_operator_exact(
                state,
                fe_state,
                normalizer,
                exact_benchmark,
                config,
            )
            exact_record = {
                "step": step,
                "samples_seen": step * config.effective_batch_size,
                **exact_metrics,
                "elapsed_seconds": float(time.perf_counter() - start_time),
            }
            exact_history.append(exact_record)
            save_metric_history(
                exact_history,
                output_dir,
                "operator_exact_history",
                OPERATOR_EXACT_HISTORY_COLUMNS,
            )
            progress.write(
                f"[OL exact {step:07d}] "
                f"area_RL2(P)={exact_record['p_area_relative_l2']:.4e} "
                f"RMSE(P)={exact_record['p_rmse']:.4e} "
                f"flux_RL2={exact_record['inner_flux_relative_l2']:.4e}"
            )
            if config.exact_eval_save_figure:
                save_exact_monitor_figure(
                    exact_benchmark,
                    exact_pred,
                    output_dir
                    / "exact_monitor"
                    / f"ol_step_{step:07d}.png",
                    f"OL analytic prediction at step {step}",
                )
            del exact_pred

        if step % config.ol_checkpoint_interval == 0:
            save_params(
                state.params,
                output_dir / "ol_params_latest.msgpack",
            )

    save_params(state.params, output_dir / "ol_params.msgpack")
    save_params(state.params, output_dir / "ol_params_latest.msgpack")
    if history:
        csv_path, npz_path = save_operator_history(history, output_dir)
        print(f"Operator history CSV: {csv_path}")
        print(f"Operator history NPZ: {npz_path}")
    if exact_history:
        csv_path, npz_path = save_metric_history(
            exact_history,
            output_dir,
            "operator_exact_history",
            OPERATOR_EXACT_HISTORY_COLUMNS,
        )
        print(f"Operator exact history CSV: {csv_path}")
        print(f"Operator exact history NPZ: {npz_path}")
    return state


def load_trained_states(
    config: PolarAnnulusConfig,
) -> tuple[TrainState, TrainState, FieldNormalizer]:
    key = jax.random.PRNGKey(config.seed + 20_000)
    key_fe, key_ol = jax.random.split(key)
    fe_state, _ = create_fe_state(config, key_fe)
    ol_state, _ = create_ol_state(config, key_ol)

    fe_path = config.output_dir / "fe_params.msgpack"
    ol_path = config.output_dir / "ol_params.msgpack"
    fe_params = serialization.from_bytes(fe_state.params, fe_path.read_bytes())
    ol_params = serialization.from_bytes(ol_state.params, ol_path.read_bytes())
    return (
        fe_state.replace(params=fe_params),
        ol_state.replace(params=ol_params),
        load_normalizer(config.output_dir),
    )


def predict_zero_source_cosine_flux(
    config: PolarAnnulusConfig,
    fe_state: TrainState,
    ol_state: TrainState,
    normalizer: FieldNormalizer,
    k_value: float,
) -> dict[str, Array]:
    """SNO inference for f=0 and g_n=-P_r=cos(theta)."""
    if not config.k_min <= k_value <= config.k_max:
        raise ValueError("k_value lies outside the trained configuration range.")

    coords = make_polar_grid(config)
    zero_f = jnp.zeros((1, config.n_pod), dtype=jnp.float32)
    latent_f = fe_state.apply_fn(
        {"params": fe_state.params},
        normalize_f(zero_f, normalizer),
        method=FunctionEncoder.encode_f,
    )
    f_tokens = make_source_tokens(latent_f, config)

    boundary_coords, boundary_flux = make_target_cosine_boundary(config, 1)
    boundary_tokens = make_condition_tokens_from_arrays(
        boundary_coords,
        boundary_flux,
        config,
    )
    k_values = jnp.asarray([k_value], dtype=jnp.float32)

    pred_latent_p = ol_state.apply_fn(
        {"params": ol_state.params},
        f_tokens,
        boundary_tokens,
        k_values,
    )
    pred_p_norm = fe_state.apply_fn(
        {"params": fe_state.params},
        pred_latent_p,
        coords,
        method=FunctionEncoder.reconstruct_p,
    )
    pred_p = denormalize_p(pred_p_norm, normalizer)

    return {
        "coords_hat": coords,
        "p_pred": pred_p,
        "boundary_coords": boundary_coords,
        "boundary_flux": boundary_flux,
        "k_values": k_values,
        "pred_latent_p": pred_latent_p,
    }
