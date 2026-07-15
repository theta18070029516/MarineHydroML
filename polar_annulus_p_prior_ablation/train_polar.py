from __future__ import annotations

from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import csv
import gc
import json
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
    OperatorBatch,
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
    sample_operator_batch,
)
from exact_solution import exact_annulus_solution
from models_polar import FunctionEncoder, OperatorTransformer


Array = jax.Array


class TrainState(train_state.TrainState):
    pass


CHECKPOINT_FORMAT_VERSION = 1
MILESTONE_STEPS = frozenset((10_000, 50_000, 100_000, 200_000, 300_000, 400_000, 500_000))


def relative_l2(pred: Array, ref: Array) -> Array:
    denominator = jnp.maximum(jnp.linalg.norm(ref, axis=-1), 1.0e-12)
    return jnp.linalg.norm(pred - ref, axis=-1) / denominator


def save_params(params: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(serialization.to_bytes(params))
    temporary.replace(path)


def save_training_checkpoint(
    state: TrainState,
    key_train: Array,
    key_eval: Array,
    config: PolarAnnulusConfig,
    stage: str,
    elapsed_seconds: float,
) -> tuple[Path, Path]:
    """Atomically save all state needed for bitwise-continuable training."""
    if stage not in ("fe", "ol"):
        raise ValueError("stage must be 'fe' or 'ol'.")
    output_dir = config.output_dir
    binary_path = output_dir / f"{stage}_checkpoint_latest.msgpack"
    metadata_path = output_dir / f"{stage}_checkpoint_latest.json"
    fingerprint = config.fingerprint()
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "stage": stage,
        "config_fingerprint": fingerprint,
        "state": {
            "step": state.step,
            "params": state.params,
            "opt_state": state.opt_state,
        },
        "key_train": key_train,
        "key_eval": key_eval,
        "elapsed_seconds": np.asarray(elapsed_seconds, dtype=np.float64),
    }
    temporary = binary_path.with_suffix(binary_path.suffix + ".tmp")
    temporary.write_bytes(serialization.to_bytes(payload))
    temporary.replace(binary_path)

    metadata = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "stage": stage,
        "completed_steps": int(jax.device_get(state.step)),
        "config_fingerprint": fingerprint,
        "elapsed_seconds": float(elapsed_seconds),
        "checkpoint_file": binary_path.name,
    }
    metadata_tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata_tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata_tmp.replace(metadata_path)
    return binary_path, metadata_path


def load_training_checkpoint(
    state: TrainState,
    key_train: Array,
    key_eval: Array,
    config: PolarAnnulusConfig,
    stage: str,
) -> tuple[TrainState, Array, Array, float]:
    """Restore params, optimizer, step, both PRNG streams and elapsed time."""
    path = config.output_dir / f"{stage}_checkpoint_latest.msgpack"
    if not path.exists():
        raise FileNotFoundError(path)
    template = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "stage": stage,
        "config_fingerprint": config.fingerprint(),
        "state": {
            "step": state.step,
            "params": state.params,
            "opt_state": state.opt_state,
        },
        "key_train": key_train,
        "key_eval": key_eval,
        "elapsed_seconds": np.asarray(0.0, dtype=np.float64),
    }
    restored = serialization.from_bytes(template, path.read_bytes())
    if restored["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint format version.")
    if restored["stage"] != stage:
        raise ValueError(f"Checkpoint stage is {restored['stage']!r}, expected {stage!r}.")
    expected = config.fingerprint()
    if restored["config_fingerprint"] != expected:
        raise ValueError(
            "Checkpoint/config fingerprint mismatch; refusing an unsafe resume."
        )
    restored_state = state.replace(
        step=restored["state"]["step"],
        params=restored["state"]["params"],
        opt_state=restored["state"]["opt_state"],
    )
    return (
        restored_state,
        restored["key_train"],
        restored["key_eval"],
        float(restored["elapsed_seconds"]),
    )


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


def load_fe_inference_state(
    config: PolarAnnulusConfig,
) -> tuple[SimpleNamespace, FieldNormalizer]:
    """Load FE parameters without constructing its 534 MiB Adam state."""
    final_path = config.output_dir / "fe_params.msgpack"
    checkpoint_path = config.output_dir / "fe_checkpoint_latest.msgpack"

    if checkpoint_path.exists():
        payload = serialization.msgpack_restore(checkpoint_path.read_bytes())
        if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("Unsupported FE checkpoint format version.")
        if payload["stage"] != "fe":
            raise ValueError("Expected an FE checkpoint.")
        if payload["config_fingerprint"] != config.fingerprint():
            raise ValueError(
                "FE checkpoint/config fingerprint mismatch; refusing to load."
            )
        raw_params = payload["state"]["params"]
        del payload
    elif final_path.exists():
        raw_params = serialization.msgpack_restore(final_path.read_bytes())
    else:
        raise FileNotFoundError(
            f"Neither {final_path.name} nor {checkpoint_path.name} exists in "
            f"{config.output_dir}."
        )

    params = jax.device_put(raw_params)
    jax.tree_util.tree_map(lambda value: value.block_until_ready(), params)
    del raw_params
    gc.collect()

    model = FunctionEncoder(config)
    inference_state = SimpleNamespace(params=params, apply_fn=model.apply)
    return inference_state, load_normalizer(config.output_dir)


FE_HISTORY_COLUMNS = (
    "step",
    "samples_seen",
    "loss",
    "data_loss",
    "loss_p",
    "loss_f",
    "physics_loss",
    "eval_p_relative_l2",
    "eval_f_relative_l2",
    "elapsed_seconds",
)


OPERATOR_HISTORY_COLUMNS = (
    "step",
    "samples_seen",
    "loss",
    "in_distribution_relative_l2",
    "exact_relative_l2",
    "elapsed_seconds",
)


def _save_history(
    history: list[dict[str, float | int]],
    out_dir: Path,
    stem: str,
    columns: tuple[str, ...],
) -> tuple[Path, Path]:
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
    arrays = {name: np.asarray([record[name] for record in history]) for name in columns}
    np.savez(npz_temp_path, **arrays)
    npz_temp_path.replace(npz_path)
    return csv_path, npz_path


def _load_history(path: Path) -> list[dict[str, float | int]]:
    if not path.exists():
        return []
    records: list[dict[str, float | int]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            record: dict[str, float | int] = {}
            for name, value in row.items():
                record[name] = int(value) if name in ("step", "samples_seen") else float(value)
            records.append(record)
    return records


def save_fe_history(
    history: list[dict[str, float | int]], out_dir: Path
) -> tuple[Path, Path]:
    return _save_history(history, out_dir, "fe_training_history", FE_HISTORY_COLUMNS)


def save_operator_history(
    history: list[dict[str, float | int]],
    out_dir: Path,
) -> tuple[Path, Path]:
    """Atomically save operator-training metrics as CSV and NPZ."""
    return _save_history(
        history, out_dir, "operator_training_history", OPERATOR_HISTORY_COLUMNS
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


@partial(jax.jit, donate_argnums=(0,))
def ol_train_step(
    state: TrainState,
    f_tokens: Array,
    boundary_tokens: Array,
    k_values: Array,
    target_latent_p: Array,
) -> tuple[TrainState, Array, Array]:
    """Update OL once and return the prediction already used by the loss."""

    def loss_with_prediction(params):
        pred_latent_p = state.apply_fn(
            {"params": params},
            f_tokens,
            boundary_tokens,
            k_values,
        )
        loss = jnp.mean((pred_latent_p - target_latent_p) ** 2)
        return loss, pred_latent_p

    (loss, pred_latent_p), gradients = jax.value_and_grad(
        loss_with_prediction,
        has_aux=True,
    )(
        state.params
    )
    state = state.apply_gradients(grads=gradients)
    return state, loss, pred_latent_p


@partial(jax.jit, static_argnames=("fe_apply_fn",))
def pressure_relative_l2_from_latent(
    fe_params: Any,
    fe_apply_fn: Any,
    pred_latent_p: Array,
    pod_coords: Array,
    target_p_pod: Array,
    normalizer: FieldNormalizer,
) -> Array:
    """Decode a prediction and score it on the current OL training batch."""
    pred_p_norm = fe_apply_fn(
        {"params": fe_params},
        pred_latent_p,
        pod_coords,
        method=FunctionEncoder.reconstruct_p,
    )
    pred_p = denormalize_p(pred_p_norm, normalizer)
    return jnp.mean(relative_l2(pred_p, target_p_pod))


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
    fe_state: Any,
    batch: SampleBatch | OperatorBatch,
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


def _legacy_train_fe_parameter_only(
    config: PolarAnnulusConfig,
) -> tuple[TrainState, FieldNormalizer]:
    """Train FE using fresh prior samples at every optimization step."""

    config.save_json()

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

    print(
        f"[Normalizer] 已保存至：{config.output_dir}"
    )

    state, _ = create_fe_state(
        config,
        key_init,
    )

    for step in range(config.fe_steps):
        key_train, batch_key = jax.random.split(key_train)

        batch = sample_batch(
            batch_key,
            config,
        )

        state, metrics = fe_train_step(
            state,
            batch,
            normalizer,
            config,
        )

        if step % 500 == 0 or step == config.fe_steps - 1:
            key_eval, eval_key = jax.random.split(key_eval)

            eval_batch = sample_batch(
                eval_key,
                config,
            )

            err_p, err_f = fe_eval_step(
                state,
                eval_batch,
                normalizer,
                config,
            )

            print(
                f"[FE {step:07d}] "
                f"loss={float(metrics['loss']):.4e} "
                f"data={float(metrics['data_loss']):.4e} "
                f"phys={float(metrics['physics_loss']):.4e} "
                f"RL2(P)={float(err_p):.4e} "
                f"RL2(f)={float(err_f):.4e}"
            )

            if step % 10000 == 0:
                save_params(
                    state.params,
                    config.output_dir / "fe_params.msgpack",
                )

    save_params(
        state.params,
        config.output_dir / "fe_params.msgpack",
    )

    return state, normalizer


def _legacy_train_operator_parameter_only(
    config: PolarAnnulusConfig,
    fe_state: TrainState,
    normalizer: FieldNormalizer,
) -> TrainState:
    """Train the latent operator with a newly sampled batch at every step.

    Training metrics are periodically evaluated on an independent fresh batch
    and saved to both CSV and NPZ. The reported physical-space relative L2 is
    computed after decoding the predicted pressure latent at random probe points.
    """
    config.save_json()
    output_dir = config.output_dir

    master_key = jax.random.PRNGKey(config.seed + 10_000)
    key_init, key_train, key_eval = jax.random.split(master_key, 3)
    state, _ = create_ol_state(config, key_init)

    # Monitoring must not duplicate the full optimization batch in accelerator
    # memory. Keep the same physical distribution while reducing only its batch
    # dimension and the number of continuous pressure probes.
    eval_probe_points = min(
        config.random_probe_points,
        config.ol_eval_probe_points,
    )
    eval_config = replace(
        config,
        sample_size=min(config.sample_size, config.ol_eval_sample_size),
        random_probe_points=eval_probe_points,
        fe_physics_points=min(config.fe_physics_points, eval_probe_points),
    )
    print(
        "[OL eval] "
        f"batch={eval_config.effective_batch_size}, "
        f"probe_points={eval_config.random_probe_points}"
    )

    history: list[dict[str, float | int]] = []
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
        state, train_loss, _ = ol_train_step(
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
        if should_log:
            # Materialize the scalar loss and drop references to the training
            # batch before allocating the independent evaluation batch.
            train_loss_value = float(train_loss)
            del train_batch, f_tokens, boundary_tokens, target_latent_p, train_loss

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
                "loss": train_loss_value,
                "in_distribution_relative_l2": float(
                    eval_metrics["physical_relative_l2"]
                ),
                "exact_relative_l2": float("nan"),
                "elapsed_seconds": float(time.perf_counter() - start_time),
            }
            history.append(record)
            save_operator_history(history, output_dir)

            progress.set_postfix(
                loss=f"{record['loss']:.3e}",
                p_rl2=f"{record['in_distribution_relative_l2']:.3e}",
                refresh=False,
            )
            progress.write(
                f"[OL {step:07d}] "
                f"loss={record['loss']:.4e} "
                f"RL2(P)={record['in_distribution_relative_l2']:.4e}"
            )
            del (
                eval_batch,
                eval_f_tokens,
                eval_boundary_tokens,
                eval_target_latent_p,
                eval_metrics,
            )

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
    return state


def _legacy_load_trained_states_parameter_only(
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


def build_exact_solution_benchmark(
    config: PolarAnnulusConfig,
) -> dict[str, Array | float]:
    """Precompute the fixed f=0, g_n=cos(theta), k=1 exact benchmark."""
    k_value = 1.0
    if not config.k_min <= k_value <= config.k_max:
        raise ValueError("The exact online benchmark requires k=1 in the training range.")

    coords_hat = make_polar_grid(config)
    radius = r_from_hat(coords_hat[:, 1], config)
    theta = jnp.pi * (coords_hat[:, 0] + 1.0)
    p_exact_np = exact_annulus_solution(
        np.asarray(jax.device_get(radius)),
        np.asarray(jax.device_get(theta)),
        k_value,
        config.r_inner,
        config.r_outer,
        flux_amplitude=1.0,
    )
    p_exact = jnp.asarray(p_exact_np[None, :], dtype=jnp.float32)

    # The grid is flattened in [Nr, Nt] order. The periodic theta weights are
    # constant and cancel from relative L2. Use trapezoidal radial weights and
    # the polar area Jacobian r.
    radial_endpoint_weight = jnp.where(
        jnp.isclose(jnp.abs(coords_hat[:, 1]), 1.0),
        0.5,
        1.0,
    )
    area_weights = radius * radial_endpoint_weight
    return {
        "coords_hat": coords_hat,
        "p_exact": p_exact,
        "area_weights": area_weights,
        "k_value": k_value,
    }


def evaluate_exact_solution_benchmark(
    config: PolarAnnulusConfig,
    fe_state: TrainState,
    ol_state: TrainState,
    normalizer: FieldNormalizer,
    benchmark: dict[str, Array | float],
) -> dict[str, Array]:
    """Evaluate deterministic SNO pressure errors against the analytic solution."""
    prediction = predict_zero_source_cosine_flux(
        config,
        fe_state,
        ol_state,
        normalizer,
        float(benchmark["k_value"]),
    )
    p_pred = prediction["p_pred"]
    p_exact = benchmark["p_exact"]
    area_weights = benchmark["area_weights"][None, :]
    error = p_pred - p_exact
    grid_relative_l2 = jnp.linalg.norm(error) / jnp.maximum(
        jnp.linalg.norm(p_exact), 1.0e-12
    )
    area_relative_l2 = jnp.sqrt(
        jnp.sum(area_weights * error**2)
        / jnp.maximum(jnp.sum(area_weights * p_exact**2), 1.0e-12)
    )
    return {
        "grid_relative_l2": grid_relative_l2,
        "area_weighted_relative_l2": area_relative_l2,
    }


# The definitions below are the resumable ablation entry points.  They are kept
# at the end of the snapshot so importing this isolated repository cannot fall
# back to the original parameter-only training loops above.
def train_fe(
    config: PolarAnnulusConfig,
    resume: bool = False,
) -> tuple[TrainState, FieldNormalizer]:
    """Train FE with a complete checkpoint and two independent PRNG streams."""
    config.save_json()
    master_key = jax.random.PRNGKey(config.seed)
    key_norm, key_init, key_train, key_eval = jax.random.split(master_key, 4)
    state, _ = create_fe_state(config, key_init)
    checkpoint_path = config.output_dir / "fe_checkpoint_latest.msgpack"

    if resume and checkpoint_path.exists():
        normalizer = load_normalizer(config.output_dir)
        state, key_train, key_eval, elapsed_offset = load_training_checkpoint(
            state, key_train, key_eval, config, "fe"
        )
        start_step = int(jax.device_get(state.step))
        history = [
            record
            for record in _load_history(config.output_dir / "fe_training_history.csv")
            if int(record["step"]) <= start_step
        ]
        print(f"[FE resume] completed_steps={start_step}")
    else:
        normalizer, _ = build_field_normalizer_online(key_norm, config)
        jax.block_until_ready(normalizer.std_f)
        save_normalizer(normalizer, config.output_dir)
        start_step = 0
        elapsed_offset = 0.0
        history: list[dict[str, float | int]] = []
        print(
            "[Normalizer]"
            f" mean_p={float(normalizer.mean_p):.6e},"
            f" std_p={float(normalizer.std_p):.6e},"
            f" mean_f={float(normalizer.mean_f):.6e},"
            f" std_f={float(normalizer.std_f):.6e}"
        )

    if start_step > config.fe_steps:
        raise ValueError("FE checkpoint step exceeds configured fe_steps.")

    start_time = time.perf_counter()
    progress = trange(
        start_step,
        config.fe_steps,
        desc="Function-encoder training",
        unit="step",
        dynamic_ncols=True,
    )
    for step_index in progress:
        key_train, batch_key = jax.random.split(key_train)
        batch = sample_batch(batch_key, config)
        state, metrics = fe_train_step(state, batch, normalizer, config)
        completed_step = step_index + 1

        should_log = (
            completed_step == 1
            or completed_step % config.fe_log_interval == 0
            or completed_step == config.fe_steps
        )
        if should_log:
            key_eval, eval_key = jax.random.split(key_eval)
            eval_batch = sample_batch(eval_key, config)
            err_p, err_f = fe_eval_step(state, eval_batch, normalizer, config)
            jax.block_until_ready(err_f)
            elapsed = elapsed_offset + time.perf_counter() - start_time
            record = {
                "step": completed_step,
                "samples_seen": completed_step * config.effective_batch_size,
                "loss": float(metrics["loss"]),
                "data_loss": float(metrics["data_loss"]),
                "loss_p": float(metrics["loss_p"]),
                "loss_f": float(metrics["loss_f"]),
                "physics_loss": float(metrics["physics_loss"]),
                "eval_p_relative_l2": float(err_p),
                "eval_f_relative_l2": float(err_f),
                "elapsed_seconds": float(elapsed),
            }
            history.append(record)
            save_fe_history(history, config.output_dir)
            progress.set_postfix(
                loss=f"{record['loss']:.3e}",
                p_rl2=f"{record['eval_p_relative_l2']:.3e}",
                f_rl2=f"{record['eval_f_relative_l2']:.3e}",
                refresh=False,
            )

        if (
            completed_step % config.checkpoint_interval == 0
            or completed_step == config.fe_steps
        ):
            elapsed = elapsed_offset + time.perf_counter() - start_time
            save_training_checkpoint(
                state, key_train, key_eval, config, "fe", elapsed
            )
        # if completed_step in MILESTONE_STEPS:
        #     save_params(
        #         state.params,
        #         config.output_dir / f"fe_params_step_{completed_step:09d}.msgpack",
        #     )

    save_params(state.params, config.output_dir / "fe_params.msgpack")
    return state, normalizer


def train_operator(
    config: PolarAnnulusConfig,
    fe_state: Any,
    normalizer: FieldNormalizer,
    resume: bool = False,
) -> TrainState:
    """Low-memory OL training with current-batch and exact-solution monitoring."""
    config.save_json()
    output_dir = config.output_dir
    master_key = jax.random.PRNGKey(config.seed + 10_000)
    key_init, key_train, key_eval = jax.random.split(master_key, 3)
    state, _ = create_ol_state(config, key_init)
    checkpoint_path = output_dir / "ol_checkpoint_latest.msgpack"

    if resume and checkpoint_path.exists():
        state, key_train, key_eval, elapsed_offset = load_training_checkpoint(
            state, key_train, key_eval, config, "ol"
        )
        start_step = int(jax.device_get(state.step))
        history = [
            record
            for record in _load_history(output_dir / "operator_training_history.csv")
            if int(record["step"]) <= start_step
        ]
        if history and set(history[-1]) != set(OPERATOR_HISTORY_COLUMNS):
            raise ValueError(
                "The existing OL history uses the older evaluation schema. "
                "Use a new run_name for the low-memory OL run."
            )
        print(f"[OL resume] completed_steps={start_step}")
    else:
        start_step = 0
        elapsed_offset = 0.0
        history: list[dict[str, float | int]] = []

    if start_step > config.ol_steps:
        raise ValueError("OL checkpoint step exceeds configured ol_steps.")

    exact_benchmark = build_exact_solution_benchmark(config)
    print(
        "[OL monitor] every "
        f"{config.ol_log_interval} steps: loss, current-batch RL2(P), "
        "exact RL2(P)."
    )
    start_time = time.perf_counter()
    progress = trange(
        start_step,
        config.ol_steps,
        desc="Operator training",
        unit="step",
        dynamic_ncols=True,
    )
    for step_index in progress:
        completed_step = step_index + 1
        key_train, batch_key = jax.random.split(key_train)
        train_batch = sample_operator_batch(batch_key, config)
        f_tokens, boundary_tokens, target_latent_p = encode_operator_batch(
            fe_state, train_batch, normalizer, config
        )
        state, train_loss, train_pred_latent_p = ol_train_step(
            state,
            f_tokens,
            boundary_tokens,
            train_batch.k_values,
            target_latent_p,
        )

        should_log = (
            completed_step % config.ol_log_interval == 0
            or completed_step == config.ol_steps
        )
        if should_log:
            in_distribution_error = pressure_relative_l2_from_latent(
                fe_state.params,
                fe_state.apply_fn,
                train_pred_latent_p,
                train_batch.pod_coords,
                train_batch.p_pod,
                normalizer,
            )
            exact_metrics = evaluate_exact_solution_benchmark(
                config,
                fe_state,
                state,
                normalizer,
                exact_benchmark,
            )
            train_loss_value = float(train_loss)
            in_distribution_value = float(in_distribution_error)
            exact_value = float(exact_metrics["area_weighted_relative_l2"])
            elapsed = elapsed_offset + time.perf_counter() - start_time
            record = {
                "step": completed_step,
                "samples_seen": completed_step * config.effective_batch_size,
                "loss": train_loss_value,
                "in_distribution_relative_l2": in_distribution_value,
                "exact_relative_l2": exact_value,
                "elapsed_seconds": float(elapsed),
            }
            history.append(record)
            save_operator_history(history, output_dir)
            progress.set_postfix(
                loss=f"{record['loss']:.3e}",
                in_l2=f"{record['in_distribution_relative_l2']:.3e}",
                exact_l2=f"{record['exact_relative_l2']:.3e}",
                refresh=True,
            )
            progress.write(
                f"[OL {completed_step:07d}] "
                f"loss={record['loss']:.4e} "
                f"in_dist_RL2(P)={record['in_distribution_relative_l2']:.4e} "
                f"exact_RL2(P)={record['exact_relative_l2']:.4e} "
                f"elapsed={record['elapsed_seconds']:.1f}s"
            )

        del (
            train_batch,
            f_tokens,
            boundary_tokens,
            target_latent_p,
            train_pred_latent_p,
            train_loss,
        )

        if (
            completed_step % config.checkpoint_interval == 0
            or completed_step == config.ol_steps
        ):
            elapsed = elapsed_offset + time.perf_counter() - start_time
            save_training_checkpoint(
                state, key_train, key_eval, config, "ol", elapsed
            )
        # if completed_step in MILESTONE_STEPS:
        #     save_params(
        #         state.params,
        #         output_dir / f"ol_params_step_{completed_step:09d}.msgpack",
        #     )

    save_params(state.params, output_dir / "ol_params.msgpack")
    return state


def load_fe_state(config: PolarAnnulusConfig) -> tuple[TrainState, FieldNormalizer]:
    """Load the completed FE stage without importing either reference project."""
    key_norm, key_init, key_train, key_eval = jax.random.split(
        jax.random.PRNGKey(config.seed), 4
    )
    del key_norm
    state, _ = create_fe_state(config, key_init)
    final_path = config.output_dir / "fe_params.msgpack"
    if final_path.exists():
        params = serialization.from_bytes(state.params, final_path.read_bytes())
        state = state.replace(params=params)
    else:
        state, _, _, _ = load_training_checkpoint(
            state, key_train, key_eval, config, "fe"
        )
    return state, load_normalizer(config.output_dir)


def load_trained_states(
    config: PolarAnnulusConfig,
) -> tuple[TrainState, TrainState, FieldNormalizer]:
    fe_state, normalizer = load_fe_state(config)
    key_init, key_train, key_eval = jax.random.split(
        jax.random.PRNGKey(config.seed + 10_000), 3
    )
    ol_state, _ = create_ol_state(config, key_init)
    final_path = config.output_dir / "ol_params.msgpack"
    if final_path.exists():
        params = serialization.from_bytes(ol_state.params, final_path.read_bytes())
        ol_state = ol_state.replace(params=params)
    else:
        ol_state, _, _, _ = load_training_checkpoint(
            ol_state, key_train, key_eval, config, "ol"
        )
    return fe_state, ol_state, normalizer
