from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any
import csv
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training import train_state
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from tqdm.auto import trange

from config_varpolar import VarPolarConfig
from data_varpolar import (
    FieldConditionNormalizer,
    SampleBatch,
    build_normalizer_online,
    denormalize_f,
    denormalize_p,
    make_condition_tokens,
    make_source_tokens,
    normalize_f,
    normalize_p,
    sample_batch,
)
from fem_monitor import (
    FEMMonitorSet,
    evaluate_fe_fem,
    evaluate_operator_fem,
    load_fem_monitor,
)
from models_varpolar import FunctionEncoder, OperatorTransformer


Array = jax.Array


class TrainState(train_state.TrainState):
    pass


def relative_l2(prediction: Array, reference: Array) -> Array:
    return jnp.linalg.norm(prediction - reference, axis=-1) / jnp.maximum(
        jnp.linalg.norm(reference, axis=-1), 1.0e-12
    )


def save_params(params: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(serialization.to_bytes(params))
    temporary.replace(path)


def load_params(template: Any, path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return serialization.from_bytes(template, path.read_bytes())


def save_normalizer(
    normalizer: FieldConditionNormalizer,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **{name: np.asarray(getattr(normalizer, name)) for name in normalizer._fields},
    )


def load_normalizer(path: Path) -> FieldConditionNormalizer:
    if not path.exists():
        raise FileNotFoundError(f"Normalizer not found: {path}")
    with np.load(path) as data:
        return FieldConditionNormalizer(
            *(jnp.asarray(data[name], dtype=jnp.float32)
              for name in FieldConditionNormalizer._fields)
        )


def save_history(records: list[dict[str, float | int]], path: Path) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0].keys())
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)
    arrays = {name: np.asarray([record[name] for record in records]) for name in fields}
    npz_path = path.with_suffix(".npz")
    npz_temp = path.with_suffix(".tmp.npz")
    np.savez(npz_temp, **arrays)
    npz_temp.replace(npz_path)


def save_per_sample_metrics(
    metrics: dict[str, np.ndarray],
    path: Path,
) -> None:
    fields = ["case_index", *metrics.keys()]
    count = len(next(iter(metrics.values())))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(count):
            row = {"case_index": index}
            row.update({name: float(values[index]) for name, values in metrics.items()})
            writer.writerow(row)
    temporary.replace(path)


def save_monitor_figure(
    monitor: FEMMonitorSet,
    prediction: Array,
    config: VarPolarConfig,
    path: Path,
    title: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    indices = sorted({0, monitor.p_eval.shape[0] // 2, monitor.p_eval.shape[0] - 1})
    figure = Figure(figsize=(10.5, 3.0 * len(indices)), constrained_layout=True)
    FigureCanvasAgg(figure)
    for row, index in enumerate(indices):
        reference = np.asarray(monitor.p_eval[index]).reshape(
            config.fem_eval_radial_size, config.fem_eval_theta_size
        )
        predicted = np.asarray(prediction[index]).reshape(reference.shape)
        error = np.abs(predicted - reference)
        vmin = min(reference.min(), predicted.min())
        vmax = max(reference.max(), predicted.max())
        for column, (values, label, limits) in enumerate(
            (
                (reference, "FEM", (vmin, vmax)),
                (predicted, "SNO", (vmin, vmax)),
                (error, "abs error", (0.0, max(error.max(), 1.0e-12))),
            )
        ):
            axis = figure.add_subplot(len(indices), 3, row * 3 + column + 1)
            image = axis.imshow(
                values,
                origin="lower",
                aspect="auto",
                extent=(-1.0, 1.0, -1.0, 1.0),
                vmin=limits[0],
                vmax=limits[1],
                cmap="turbo",
            )
            axis.set_title(f"case {index}: {label}")
            axis.set_xlabel("theta_hat")
            axis.set_ylabel("r_hat")
            figure.colorbar(image, ax=axis, shrink=0.8)
    figure.suptitle(title)
    figure.savefig(path, dpi=160)
    return path


def create_fe_state(
    config: VarPolarConfig,
    key: Array,
) -> tuple[TrainState, FunctionEncoder]:
    model = FunctionEncoder(config)
    variables = model.init(
        key,
        jnp.zeros((1, config.n_pod), dtype=jnp.float32),
        jnp.zeros((1, config.n_pod), dtype=jnp.float32),
        jnp.zeros((2, 2), dtype=jnp.float32),
        method=FunctionEncoder.init_all,
    )
    optimizer = optax.adamw(
        learning_rate=config.fe_lr,
        b1=config.fe_b1,
        b2=config.fe_b2,
        weight_decay=config.weight_decay,
    )
    return (
        TrainState.create(
            apply_fn=model.apply,
            params=variables["params"],
            tx=optimizer,
        ),
        model,
    )


def create_ol_state(
    config: VarPolarConfig,
    key: Array,
) -> tuple[TrainState, OperatorTransformer]:
    model = OperatorTransformer(config)
    variables = model.init(
        key,
        jnp.zeros(
            (1, config.seq_chunks, config.seq_chunk_width), dtype=jnp.float32
        ),
        jnp.zeros(
            (1, config.cond_chunks, config.cond_chunk_width), dtype=jnp.float32
        ),
        jnp.ones((1,), dtype=jnp.float32),
    )
    optimizer = optax.adamw(
        learning_rate=config.ol_lr,
        weight_decay=config.weight_decay,
    )
    return (
        TrainState.create(
            apply_fn=model.apply,
            params=variables["params"],
            tx=optimizer,
        ),
        model,
    )


def fe_loss_fn(
    params: Any,
    state: TrainState,
    batch: SampleBatch,
    normalizer: FieldConditionNormalizer,
    config: VarPolarConfig,
) -> tuple[Array, dict[str, Array]]:
    del config
    latent_p = state.apply_fn(
        {"params": params},
        normalize_p(batch.p_pod, normalizer),
        method=FunctionEncoder.encode_p,
    )
    latent_f = state.apply_fn(
        {"params": params},
        normalize_f(batch.f_pod, normalizer),
        method=FunctionEncoder.encode_f,
    )
    pred_p = state.apply_fn(
        {"params": params},
        latent_p,
        batch.probe_coords,
        method=FunctionEncoder.reconstruct_p,
    )
    pred_f = state.apply_fn(
        {"params": params},
        latent_f,
        batch.probe_coords,
        method=FunctionEncoder.reconstruct_f,
    )
    loss_p = jnp.mean(
        (pred_p - normalize_p(batch.p_probe, normalizer)) ** 2
    )
    loss_f = jnp.mean(
        (pred_f - normalize_f(batch.f_probe, normalizer)) ** 2
    )
    total = loss_p + loss_f
    return total, {"loss": total, "loss_p": loss_p, "loss_f": loss_f}


@partial(jax.jit, static_argnames=("config",))
def fe_train_step(
    state: TrainState,
    batch: SampleBatch,
    normalizer: FieldConditionNormalizer,
    config: VarPolarConfig,
) -> tuple[TrainState, dict[str, Array]]:
    (_, metrics), gradients = jax.value_and_grad(
        fe_loss_fn, has_aux=True
    )(state.params, state, batch, normalizer, config)
    return state.apply_gradients(grads=gradients), metrics


@jax.jit
def fe_eval_step(
    state: TrainState,
    batch: SampleBatch,
    normalizer: FieldConditionNormalizer,
) -> tuple[Array, Array]:
    latent_p = state.apply_fn(
        {"params": state.params},
        normalize_p(batch.p_pod, normalizer),
        method=FunctionEncoder.encode_p,
    )
    latent_f = state.apply_fn(
        {"params": state.params},
        normalize_f(batch.f_pod, normalizer),
        method=FunctionEncoder.encode_f,
    )
    pred_p = denormalize_p(
        state.apply_fn(
            {"params": state.params},
            latent_p,
            batch.probe_coords,
            method=FunctionEncoder.reconstruct_p,
        ),
        normalizer,
    )
    pred_f = denormalize_f(
        state.apply_fn(
            {"params": state.params},
            latent_f,
            batch.probe_coords,
            method=FunctionEncoder.reconstruct_f,
        ),
        normalizer,
    )
    return (
        jnp.mean(relative_l2(pred_p, batch.p_probe)),
        jnp.mean(relative_l2(pred_f, batch.f_probe)),
    )


def encode_operator_batch(
    fe_state: TrainState,
    batch: SampleBatch,
    normalizer: FieldConditionNormalizer,
    config: VarPolarConfig,
) -> tuple[Array, Array, Array]:
    latent_f = fe_state.apply_fn(
        {"params": fe_state.params},
        normalize_f(batch.f_pod, normalizer),
        method=FunctionEncoder.encode_f,
    )
    target_latent = fe_state.apply_fn(
        {"params": fe_state.params},
        normalize_p(batch.p_pod, normalizer),
        method=FunctionEncoder.encode_p,
    )
    return (
        make_source_tokens(latent_f, config),
        make_condition_tokens(batch, normalizer, config),
        target_latent,
    )


def ol_loss_fn(
    params: Any,
    state: TrainState,
    f_tokens: Array,
    boundary_tokens: Array,
    k_values: Array,
    target_latent: Array,
) -> Array:
    prediction = state.apply_fn(
        {"params": params}, f_tokens, boundary_tokens, k_values
    )
    return jnp.mean((prediction - target_latent) ** 2)


@jax.jit
def ol_train_step(
    state: TrainState,
    f_tokens: Array,
    boundary_tokens: Array,
    k_values: Array,
    target_latent: Array,
) -> tuple[TrainState, Array]:
    loss, gradients = jax.value_and_grad(ol_loss_fn)(
        state.params,
        state,
        f_tokens,
        boundary_tokens,
        k_values,
        target_latent,
    )
    return state.apply_gradients(grads=gradients), loss


@partial(jax.jit, static_argnames=("ol_apply_fn", "fe_apply_fn"))
def ol_eval_step(
    ol_params: Any,
    ol_apply_fn: Any,
    fe_params: Any,
    fe_apply_fn: Any,
    f_tokens: Array,
    boundary_tokens: Array,
    k_values: Array,
    target_latent: Array,
    probe_coords: Array,
    target_p: Array,
    normalizer: FieldConditionNormalizer,
) -> tuple[Array, Array]:
    pred_latent = ol_apply_fn(
        {"params": ol_params}, f_tokens, boundary_tokens, k_values
    )
    pred_p = denormalize_p(
        fe_apply_fn(
            {"params": fe_params},
            pred_latent,
            probe_coords,
            method=FunctionEncoder.reconstruct_p,
        ),
        normalizer,
    )
    return (
        jnp.mean((pred_latent - target_latent) ** 2),
        jnp.mean(relative_l2(pred_p, target_p)),
    )


def _load_required_monitor(config: VarPolarConfig) -> FEMMonitorSet:
    monitor = load_fem_monitor(config.fem_monitor_path)
    if monitor.p_pod.shape[0] != config.fem_monitor_size:
        raise ValueError(
            f"Expected {config.fem_monitor_size} FEM cases, got {monitor.p_pod.shape[0]}."
        )
    return monitor


def train_fe(
    config: VarPolarConfig,
) -> tuple[TrainState, FieldConditionNormalizer]:
    output = config.output_dir
    config.save_json(output / "config_fe.json")
    key = jax.random.PRNGKey(config.seed)
    key_norm, key_init, key_train, key_eval = jax.random.split(key, 4)
    normalizer, _ = build_normalizer_online(key_norm, config)
    jax.block_until_ready(normalizer.std_g)
    save_normalizer(normalizer, output / "normalizer.npz")
    state, _ = create_fe_state(config, key_init)
    monitor = _load_required_monitor(config)

    eval_config = replace(
        config,
        sample_size=min(config.sample_size, config.eval_sample_size),
        prior_generation_chunk_size=min(
            config.prior_generation_chunk_size, config.eval_sample_size
        ),
        random_probe_points=min(
            config.random_probe_points, config.eval_probe_points
        ),
    )
    random_history: list[dict[str, float | int]] = []
    fem_history: list[dict[str, float | int]] = []
    best_fem = np.inf
    start_time = time.perf_counter()
    progress = trange(config.fe_steps, desc="FE training", unit="step")
    for index in progress:
        step = index + 1
        key_train, batch_key = jax.random.split(key_train)
        batch = sample_batch(batch_key, config)
        state, train_metrics = fe_train_step(state, batch, normalizer, config)
        should_log = step == 1 or step % config.log_interval == 0 or step == config.fe_steps
        should_fem = step == 1 or step % config.fem_eval_interval == 0 or step == config.fe_steps
        if should_log or should_fem:
            jax.block_until_ready(train_metrics["loss"])
        if should_log:
            key_eval, eval_key = jax.random.split(key_eval)
            eval_batch = sample_batch(eval_key, eval_config)
            error_p, error_f = fe_eval_step(state, eval_batch, normalizer)
            record = {
                "step": step,
                "samples_seen": step * config.sample_size,
                "train_loss": float(train_metrics["loss"]),
                "train_loss_p": float(train_metrics["loss_p"]),
                "train_loss_f": float(train_metrics["loss_f"]),
                "eval_p_relative_l2": float(error_p),
                "eval_f_relative_l2": float(error_f),
                "elapsed_seconds": time.perf_counter() - start_time,
            }
            random_history.append(record)
            save_history(random_history, output / "fe_training_history.csv")
            progress.set_postfix(loss=f"{record['train_loss']:.3e}", p=f"{record['eval_p_relative_l2']:.3e}")
        if should_fem:
            aggregate, per_sample, prediction = evaluate_fe_fem(
                state, normalizer, monitor, config
            )
            record = {
                "step": step,
                "samples_seen": step * config.sample_size,
                **aggregate,
                "elapsed_seconds": time.perf_counter() - start_time,
            }
            fem_history.append(record)
            save_history(fem_history, output / "fe_fem_history.csv")
            save_per_sample_metrics(
                per_sample,
                output / "fem_monitor" / f"fe_step_{step:07d}_cases.csv",
            )
            save_monitor_figure(
                monitor,
                prediction,
                config,
                output / "fem_monitor" / f"fe_step_{step:07d}.png",
                f"FE FEM monitor at step {step}",
            )
            score = aggregate["p_area_relative_l2_mean"]
            progress.write(
                f"[FE FEM {step:07d}] area_RL2_mean={score:.4e} "
                f"p95={aggregate['p_area_relative_l2_p95']:.4e} "
                f"load_RL2_mean={aggregate['inner_load_relative_l2_mean']:.4e}"
            )
            if score < best_fem:
                best_fem = score
                save_params(state.params, output / "fe_params_best_fem.msgpack")
        if step % config.checkpoint_interval == 0:
            save_params(state.params, output / "fe_params_latest.msgpack")
        del batch, train_metrics

    save_params(state.params, output / "fe_params.msgpack")
    save_params(state.params, output / "fe_params_latest.msgpack")
    return state, normalizer


def train_operator(
    config: VarPolarConfig,
    fe_state: TrainState | None = None,
    normalizer: FieldConditionNormalizer | None = None,
) -> TrainState:
    output = config.output_dir
    config.save_json(output / "config_ol.json")
    if normalizer is None:
        normalizer = load_normalizer(output / "normalizer.npz")
    key = jax.random.PRNGKey(config.seed + 20_260_525)
    key_fe, key_ol, key_train, key_eval = jax.random.split(key, 4)
    if fe_state is None:
        fe_state, _ = create_fe_state(config, key_fe)
        fe_state = fe_state.replace(
            params=load_params(fe_state.params, output / "fe_params.msgpack")
        )
    state, _ = create_ol_state(config, key_ol)
    monitor = _load_required_monitor(config)
    data_config = replace(
        config,
        sample_size=config.ol_sample_size,
        prior_generation_chunk_size=config.ol_prior_generation_chunk_size,
    )
    eval_config = replace(
        data_config,
        sample_size=min(data_config.sample_size, config.eval_sample_size),
        prior_generation_chunk_size=min(
            data_config.prior_generation_chunk_size, config.eval_sample_size
        ),
        random_probe_points=min(
            data_config.random_probe_points, config.eval_probe_points
        ),
    )
    random_history: list[dict[str, float | int]] = []
    fem_history: list[dict[str, float | int]] = []
    best_fem = np.inf
    start_time = time.perf_counter()
    progress = trange(config.ol_steps, desc="Operator training", unit="step")
    for index in progress:
        step = index + 1
        key_train, batch_key = jax.random.split(key_train)
        batch = sample_batch(batch_key, data_config)
        f_tokens, boundary_tokens, target_latent = encode_operator_batch(
            fe_state, batch, normalizer, config
        )
        state, loss = ol_train_step(
            state,
            f_tokens,
            boundary_tokens,
            batch.k_values,
            target_latent,
        )
        should_log = step == 1 or step % config.log_interval == 0 or step == config.ol_steps
        should_fem = step == 1 or step % config.fem_eval_interval == 0 or step == config.ol_steps
        if should_log or should_fem:
            jax.block_until_ready(loss)
        if should_log:
            key_eval, eval_key = jax.random.split(key_eval)
            eval_batch = sample_batch(eval_key, eval_config)
            eval_f_tokens, eval_boundary_tokens, eval_target_latent = encode_operator_batch(
                fe_state, eval_batch, normalizer, config
            )
            latent_mse, physical_error = ol_eval_step(
                state.params,
                state.apply_fn,
                fe_state.params,
                fe_state.apply_fn,
                eval_f_tokens,
                eval_boundary_tokens,
                eval_batch.k_values,
                eval_target_latent,
                eval_batch.probe_coords,
                eval_batch.p_probe,
                normalizer,
            )
            record = {
                "step": step,
                "samples_seen": step * data_config.sample_size,
                "train_latent_mse": float(loss),
                "eval_latent_mse": float(latent_mse),
                "eval_physical_relative_l2": float(physical_error),
                "elapsed_seconds": time.perf_counter() - start_time,
            }
            random_history.append(record)
            save_history(random_history, output / "operator_training_history.csv")
            progress.set_postfix(loss=f"{record['train_latent_mse']:.3e}", p=f"{record['eval_physical_relative_l2']:.3e}")
        if should_fem:
            aggregate, per_sample, prediction = evaluate_operator_fem(
                state, fe_state, normalizer, monitor, config
            )
            record = {
                "step": step,
                "samples_seen": step * data_config.sample_size,
                **aggregate,
                "elapsed_seconds": time.perf_counter() - start_time,
            }
            fem_history.append(record)
            save_history(fem_history, output / "operator_fem_history.csv")
            save_per_sample_metrics(
                per_sample,
                output / "fem_monitor" / f"ol_step_{step:07d}_cases.csv",
            )
            save_monitor_figure(
                monitor,
                prediction,
                config,
                output / "fem_monitor" / f"ol_step_{step:07d}.png",
                f"Operator FEM monitor at step {step}",
            )
            score = aggregate["p_area_relative_l2_mean"]
            progress.write(
                f"[OL FEM {step:07d}] area_RL2_mean={score:.4e} "
                f"p95={aggregate['p_area_relative_l2_p95']:.4e} "
                f"load_RL2_mean={aggregate['inner_load_relative_l2_mean']:.4e}"
            )
            if score < best_fem:
                best_fem = score
                save_params(state.params, output / "ol_params_best_fem.msgpack")
        if step % config.checkpoint_interval == 0:
            save_params(state.params, output / "ol_params_latest.msgpack")
        del batch, f_tokens, boundary_tokens, target_latent, loss

    save_params(state.params, output / "ol_params.msgpack")
    save_params(state.params, output / "ol_params_latest.msgpack")
    return state
