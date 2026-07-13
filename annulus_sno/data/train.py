from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scipy.stats import qmc

import jax
import jax.numpy as jnp
import optax
from flax import serialization
from flax.training import train_state

from config import AnnulusConfig
from data import (
    PCAStats,
    build_pca_stats,
    make_condition_tokens,
    make_source_tokens,
    project_with_pca,
    reconstruct_with_pca,
    sample_batch,
    build_batch_pool_list,
    FieldNormalizer,
    build_field_normalizer,
    normalize_u,
    denormalize_u,
    normalize_f,
    denormalize_f,
)
from models import FunctionEncoder, SingleFieldFunctionEncoder, OperatorTransformer


Array = jax.Array


class TrainState(train_state.TrainState):
    pass


def rl2_error(pred: Array, ref: Array) -> Array:
    denom = jnp.linalg.norm(ref, axis=-1)
    denom = jnp.clip(denom, a_min=1e-12)
    return jnp.linalg.norm(pred - ref, axis=-1) / denom


# def create_fe_state(config: AnnulusConfig, key: Array) -> tuple[TrainState, FunctionEncoder]:
#     model = FunctionEncoder(config)
#     dummy_coeff = jnp.ones((1, config.n_basis), dtype=jnp.float32)
#     dummy_coords = jnp.ones((config.radial_size * config.theta_size, 2), dtype=jnp.float32)
#     variables = model.init(key, dummy_coeff, dummy_coeff, dummy_coords, method=FunctionEncoder.init_all)
#     schedule = optax.cosine_decay_schedule(config.fe_lr, config.fe_steps, alpha=1e-3)
#     tx = optax.adamw(schedule, weight_decay=config.weight_decay, b1=config.fe_b1, b2=config.fe_b2)
#     state = TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)
#     return state, model

def create_fe_state(config: AnnulusConfig, key: Array) -> tuple[TrainState, FunctionEncoder]:
    model = FunctionEncoder(config)

    dummy_field = jnp.ones(
        (1, config.radial_size * config.theta_size),
        dtype=jnp.float32,
    )
    dummy_coords = jnp.ones(
        (config.radial_size * config.theta_size, 2),
        dtype=jnp.float32,
    )

    variables = model.init(
        key,
        dummy_field,
        dummy_field,
        dummy_coords,
        method=FunctionEncoder.init_all,
    )

    schedule = optax.cosine_decay_schedule(config.fe_lr, config.fe_steps, alpha=1e-3)
    tx = optax.adamw(schedule, weight_decay=config.weight_decay, b1=config.fe_b1, b2=config.fe_b2) #, b1=config.fe_b1, b2=config.fe_b2
    state = TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)
    return state, model

def create_single_fe_state(
    config: AnnulusConfig,
    key: Array,
    field_name: str,
) -> tuple[TrainState, SingleFieldFunctionEncoder]:
    model = SingleFieldFunctionEncoder(
        config=config,
        field_name=field_name,
    )

    dummy_field = jnp.ones(
        (1, config.radial_size * config.theta_size),
        dtype=jnp.float32,
    )
    dummy_coords = jnp.ones(
        (config.random_probe_points, 2),
        dtype=jnp.float32,
    )

    variables = model.init(
        key,
        dummy_field,
        dummy_coords,
        method=SingleFieldFunctionEncoder.init_all,
    )

    schedule = optax.cosine_decay_schedule(
        config.fe_lr,
        config.fe_steps,
        alpha=1e-3,
    )
    tx = optax.adamw(
        schedule,
        weight_decay=config.weight_decay,
        b1=config.fe_b1,
        b2=config.fe_b2,
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=tx,
    )
    return state, model


def create_ol_state(config: AnnulusConfig, key: Array) -> tuple[TrainState, OperatorTransformer]:
    model = OperatorTransformer(config)
    dummy_f = jnp.ones((1, config.seq_chunks, config.seq_chunk_width), dtype=jnp.float32)
    dummy_bc = jnp.ones((1, config.cond_chunks, config.cond_chunk_width), dtype=jnp.float32)
    dummy_k = jnp.ones((1,), dtype=jnp.float32)
    variables = model.init(key, dummy_f, dummy_bc, dummy_k)
    schedule = optax.cosine_decay_schedule(config.ol_lr, config.ol_steps, alpha=1e-4)
    tx = optax.adamw(schedule, weight_decay=config.weight_decay)
    return TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx), model


def save_params(params: dict[str, Any], path: Path) -> None:
    path.write_bytes(serialization.to_bytes(params))


def save_pca_stats(stats: PCAStats, out_dir: Path) -> None:
    jnp.save(out_dir / 'mean_u.npy', stats.mean_u)
    jnp.save(out_dir / 'modes_u.npy', stats.modes_u)
    jnp.save(out_dir / 'eigvals_u.npy', stats.eigvals_u)
    jnp.save(out_dir / 'mean_f.npy', stats.mean_f)
    jnp.save(out_dir / 'modes_f.npy', stats.modes_f)
    jnp.save(out_dir / 'eigvals_f.npy', stats.eigvals_f)


def load_pca_stats(out_dir: Path) -> PCAStats:
    return PCAStats(
        mean_u=jnp.load(out_dir / 'mean_u.npy'),
        modes_u=jnp.load(out_dir / 'modes_u.npy'),
        eigvals_u=jnp.load(out_dir / 'eigvals_u.npy'),
        mean_f=jnp.load(out_dir / 'mean_f.npy'),
        modes_f=jnp.load(out_dir / 'modes_f.npy'),
        eigvals_f=jnp.load(out_dir / 'eigvals_f.npy'),
    )

def save_field_normalizer(normalizer: FieldNormalizer, out_dir: Path) -> None:
    jnp.save(out_dir / "norm_mean_u.npy", normalizer.mean_u)
    jnp.save(out_dir / "norm_std_u.npy", normalizer.std_u)
    jnp.save(out_dir / "norm_mean_f.npy", normalizer.mean_f)
    jnp.save(out_dir / "norm_std_f.npy", normalizer.std_f)


def load_field_normalizer(out_dir: Path) -> FieldNormalizer:
    return FieldNormalizer(
        mean_u=jnp.load(out_dir / "norm_mean_u.npy"),
        std_u=jnp.load(out_dir / "norm_std_u.npy"),
        mean_f=jnp.load(out_dir / "norm_mean_f.npy"),
        std_f=jnp.load(out_dir / "norm_std_f.npy"),
    )


# def fe_loss_fn(params: dict, state: TrainState, coeffs_u: Array, coeffs_f: Array, out_u: Array, out_f: Array, coords: Array) -> Array:
#     latent_u = state.apply_fn({'params': params}, coeffs_u, method=FunctionEncoder.encode_u)
#     latent_f = state.apply_fn({'params': params}, coeffs_f, method=FunctionEncoder.encode_f)
#     pred_u = state.apply_fn({'params': params}, latent_u, coords, method=FunctionEncoder.reconstruct)
#     pred_f = state.apply_fn({'params': params}, latent_f, coords, method=FunctionEncoder.reconstruct)
#     return jnp.mean((pred_u - 0.05*out_u) ** 2) # jnp.mean((pred_u - out_u) ** 2) + jnp.mean((pred_f - out_f) ** 2)

# def fe_loss_fn(
#     params: dict,
#     state: TrainState,
#     u_in: Array,
#     f_in: Array,
#     out_u: Array,
#     out_f: Array,
#     coords: Array,
# ) -> Array:
#     latent_u = state.apply_fn(
#         {'params': params},
#         u_in,
#         method=FunctionEncoder.encode_u,
#     )
#     latent_f = state.apply_fn(
#         {'params': params},
#         f_in,
#         method=FunctionEncoder.encode_f,
#     )

#     pred_u = state.apply_fn(
#         {'params': params},
#         latent_u,
#         coords,
#         method=FunctionEncoder.reconstruct,
#     )
#     pred_f = state.apply_fn(
#         {'params': params},
#         latent_f,
#         coords,
#         method=FunctionEncoder.reconstruct,
#     )

#     loss_u = jnp.mean((pred_u - out_u) ** 2)
#     loss_f = jnp.mean((pred_f - out_f) ** 2)

#     return loss_f #loss_u #loss_f #loss_u + loss_f

def fe_loss_fn(
    params: dict,
    state: TrainState,
    u_in_norm: Array,
    f_in_norm: Array,
    u_out_norm: Array,
    f_out_norm: Array,
    coords: Array,
) -> Array:
    latent_u = state.apply_fn(
        {"params": params},
        u_in_norm,
        method=FunctionEncoder.encode_u,
    )
    latent_f = state.apply_fn(
        {"params": params},
        f_in_norm,
        method=FunctionEncoder.encode_f,
    )

    pred_u_norm = state.apply_fn(
        {"params": params},
        latent_u,
        coords,
        method=FunctionEncoder.reconstruct,
    )
    pred_f_norm = state.apply_fn(
        {"params": params},
        latent_f,
        coords,
        method=FunctionEncoder.reconstruct,
    )

    loss_u = jnp.mean((pred_u_norm - u_out_norm) ** 2)
    loss_f = jnp.mean((pred_f_norm - f_out_norm) ** 2)

    return loss_u + loss_f

def single_fe_loss_fn(
    params: dict,
    state: TrainState,
    field_in_norm: Array,
    field_out_norm: Array,
    coords: Array,
) -> Array:
    latent = state.apply_fn(
        {"params": params},
        field_in_norm,
        method=SingleFieldFunctionEncoder.encode,
    )

    pred_norm = state.apply_fn(
        {"params": params},
        latent,
        coords,
        method=SingleFieldFunctionEncoder.reconstruct,
    )

    return jnp.mean((pred_norm - field_out_norm) ** 2)


# @jax.jit
# def fe_train_step(state: TrainState, coeffs_u: Array, coeffs_f: Array, out_u: Array, out_f: Array, coords: Array) -> tuple[TrainState, Array]:
#     loss, grads = jax.value_and_grad(fe_loss_fn)(state.params, state, coeffs_u, coeffs_f, out_u, out_f, coords)
#     state = state.apply_gradients(grads=grads)
#     return state, loss

# @jax.jit
# def fe_train_step(
#     state: TrainState,
#     u_in: Array,
#     f_in: Array,
#     out_u: Array,
#     out_f: Array,
#     coords: Array,
# ) -> tuple[TrainState, Array]:
#     loss, grads = jax.value_and_grad(fe_loss_fn)(
#         state.params,
#         state,
#         u_in,
#         f_in,
#         out_u,
#         out_f,
#         coords,
#     )
#     state = state.apply_gradients(grads=grads)
#     return state, loss

@jax.jit
def fe_train_step(
    state: TrainState,
    u_in_norm: Array,
    f_in_norm: Array,
    u_out_norm: Array,
    f_out_norm: Array,
    coords: Array,
) -> tuple[TrainState, Array]:
    loss, grads = jax.value_and_grad(fe_loss_fn)(
        state.params,
        state,
        u_in_norm,
        f_in_norm,
        u_out_norm,
        f_out_norm,
        coords,
    )
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def single_fe_train_step(
    state: TrainState,
    field_in_norm: Array,
    field_out_norm: Array,
    coords: Array,
) -> tuple[TrainState, Array]:
    loss, grads = jax.value_and_grad(single_fe_loss_fn)(
        state.params,
        state,
        field_in_norm,
        field_out_norm,
        coords,
    )
    state = state.apply_gradients(grads=grads)
    return state, loss


# @jax.jit
# def fe_eval_step(state: TrainState, coeffs_u: Array, coeffs_f: Array, out_u: Array, out_f: Array, coords: Array) -> tuple[Array, Array]:
#     latent_u = state.apply_fn({'params': state.params}, coeffs_u, method=FunctionEncoder.encode_u)
#     latent_f = state.apply_fn({'params': state.params}, coeffs_f, method=FunctionEncoder.encode_f)
#     pred_u = state.apply_fn({'params': state.params}, latent_u, coords, method=FunctionEncoder.reconstruct)
#     pred_f = state.apply_fn({'params': state.params}, latent_f, coords, method=FunctionEncoder.reconstruct)
#     return rl2_error(pred_u, 0.05*out_u).mean(), rl2_error(pred_f, out_f).mean()

# @jax.jit
# def fe_eval_step(
#     state: TrainState,
#     u_in: Array,
#     f_in: Array,
#     out_u: Array,
#     out_f: Array,
#     coords: Array,
# ) -> tuple[Array, Array]:
#     latent_u = state.apply_fn({'params': state.params}, u_in, method=FunctionEncoder.encode_u)
#     latent_f = state.apply_fn({'params': state.params}, f_in, method=FunctionEncoder.encode_f)

#     pred_u = state.apply_fn({'params': state.params}, latent_u, coords, method=FunctionEncoder.reconstruct)
#     pred_f = state.apply_fn({'params': state.params}, latent_f, coords, method=FunctionEncoder.reconstruct)

#     return rl2_error(pred_u, out_u).mean(), rl2_error(pred_f, out_f).mean()

@jax.jit
def fe_eval_step(
    state: TrainState,
    u_in_norm: Array,
    f_in_norm: Array,
    u_out: Array,
    f_out: Array,
    coords: Array,
    normalizer: FieldNormalizer,
) -> tuple[Array, Array]:
    latent_u = state.apply_fn(
        {"params": state.params},
        u_in_norm,
        method=FunctionEncoder.encode_u,
    )
    latent_f = state.apply_fn(
        {"params": state.params},
        f_in_norm,
        method=FunctionEncoder.encode_f,
    )

    pred_u_norm = state.apply_fn(
        {"params": state.params},
        latent_u,
        coords,
        method=FunctionEncoder.reconstruct,
    )
    pred_f_norm = state.apply_fn(
        {"params": state.params},
        latent_f,
        coords,
        method=FunctionEncoder.reconstruct,
    )

    pred_u = denormalize_u(pred_u_norm, normalizer)
    pred_f = denormalize_f(pred_f_norm, normalizer)

    return rl2_error(pred_u, u_out).mean(), rl2_error(pred_f, f_out).mean()

@jax.jit
def single_fe_eval_step_u(
    state: TrainState,
    field_in_norm: Array,
    field_out: Array,
    coords: Array,
    normalizer: FieldNormalizer,
) -> Array:
    latent = state.apply_fn(
        {"params": state.params},
        field_in_norm,
        method=SingleFieldFunctionEncoder.encode,
    )

    pred_norm = state.apply_fn(
        {"params": state.params},
        latent,
        coords,
        method=SingleFieldFunctionEncoder.reconstruct,
    )

    pred = denormalize_u(pred_norm, normalizer)
    return rl2_error(pred, field_out).mean()

@jax.jit
def single_fe_eval_step_f(
    state: TrainState,
    field_in_norm: Array,
    field_out: Array,
    coords: Array,
    normalizer: FieldNormalizer,
) -> Array:
    latent = state.apply_fn(
        {"params": state.params},
        field_in_norm,
        method=SingleFieldFunctionEncoder.encode,
    )

    pred_norm = state.apply_fn(
        {"params": state.params},
        latent,
        coords,
        method=SingleFieldFunctionEncoder.reconstruct,
    )

    pred = denormalize_f(pred_norm, normalizer)
    return rl2_error(pred, field_out).mean()

def ol_loss_fn(params: dict, state: TrainState, f_tokens: Array, cond_tokens: Array, k_values: Array, target_u_latent: Array) -> Array:
    pred_latent_u = state.apply_fn({'params': params}, f_tokens, cond_tokens, k_values)
    return jnp.mean((pred_latent_u - target_u_latent) ** 2)


@jax.jit
def ol_train_step(state: TrainState, f_tokens: Array, cond_tokens: Array, k_values: Array, target_u_latent: Array) -> tuple[TrainState, Array]:
    loss, grads = jax.value_and_grad(ol_loss_fn)(state.params, state, f_tokens, cond_tokens, k_values, target_u_latent)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def ol_eval_step(state: TrainState, f_tokens: Array, cond_tokens: Array, k_values: Array, target_u_latent: Array) -> Array:
    pred = state.apply_fn({'params': state.params}, f_tokens, cond_tokens, k_values)
    return rl2_error(pred, target_u_latent).mean()


def compute_pca_diagnostics(config: AnnulusConfig, stats: PCAStats, key: Array) -> dict[str, float]:
    batch = sample_batch(key, config)
    coeff_u = project_with_pca(batch.u_pod, stats.mean_u, stats.modes_u)
    coeff_f = project_with_pca(batch.f_pod, stats.mean_f, stats.modes_f)
    rec_u = reconstruct_with_pca(coeff_u, stats.mean_u, stats.modes_u)
    rec_f = reconstruct_with_pca(coeff_f, stats.mean_f, stats.modes_f)
    return {
        'rl2_u': float(rl2_error(rec_u, batch.u_pod).mean()),
        'rl2_f': float(rl2_error(rec_f, batch.f_pod).mean()),
    }


# two branch net, one trunk net
def train_fe(config: AnnulusConfig) -> tuple[TrainState, FieldNormalizer]:
    key = jax.random.PRNGKey(config.seed)
    config.save_json()

    key, key_norm, key_init = jax.random.split(key, 3)

    print("[Normalizer] building field normalizer ...")
    normalizer = build_field_normalizer(config, key_norm)
    save_field_normalizer(normalizer, config.output_dir)

    print(
        "[Normalizer]",
        {
            "mean_u": float(normalizer.mean_u),
            "std_u": float(normalizer.std_u),
            "mean_f": float(normalizer.mean_f),
            "std_f": float(normalizer.std_f),
        },
    )

    state, model = create_fe_state(config, key_init)

    for step in range(config.fe_steps):
        key, key_batch = jax.random.split(key)
        batch = sample_batch(key_batch, config)

        u_in_norm = normalize_u(batch.u_pod, normalizer)
        f_in_norm = normalize_f(batch.f_pod, normalizer)

        u_out_norm = normalize_u(batch.u_probe, normalizer)
        f_out_norm = normalize_f(batch.f_probe, normalizer)

        state, loss = fe_train_step(
            state,
            u_in_norm,
            f_in_norm,
            u_out_norm,
            f_out_norm,
            batch.probe_coords,
        )

        if step % 500 == 0:
            ru, rf = fe_eval_step(
                state,
                u_in_norm,
                f_in_norm,
                batch.u_probe,
                batch.f_probe,
                batch.probe_coords,
                normalizer,
            )
            print(
                f"[FE] step={step:07d} "
                f"loss={float(loss):.4e} "
                f"rl2_u={float(ru):.4e} "
                f"rl2_f={float(rf):.4e}"
            )

    save_params(state.params, config.output_dir / "fe_params.msgpack")
    return state, normalizer

# u-deeponet and f-deeponet are independent
def train_single_fe(
    config: AnnulusConfig,
    field: str,
    normalizer: FieldNormalizer | None = None,
) -> TrainState:
    assert field in ("u", "f")

    key = jax.random.PRNGKey(config.seed)
    config.save_json()

    key, key_init_u, key_init_f = jax.random.split(key, 3)

    if normalizer is None:
        normalizer = load_field_normalizer(config.output_dir)

    print(
        "[Normalizer]",
        {
            "mean_u": float(normalizer.mean_u),
            "std_u": float(normalizer.std_u),
            "mean_f": float(normalizer.mean_f),
            "std_f": float(normalizer.std_f),
        },
    )

    if field == "u":
        key_init = key_init_u #jax.random.PRNGKey(config.seed + 101)
        save_name = "fe_u_params.msgpack"
    else:
        key_init = key_init_f #jax.random.PRNGKey(config.seed + 102)
        save_name = "fe_f_params.msgpack"

    # base_data_key = jax.random.PRNGKey(config.seed + 2026)

    state, model = create_single_fe_state(
        config=config,
        key=key_init,
        field_name=field,
    )

    for step in range(config.fe_steps):
        # key_batch = jax.random.fold_in(base_data_key, step)
        key, key_batch = jax.random.split(key)
        batch = sample_batch(key_batch, config)

        if field == "u":
            field_in_norm = normalize_u(batch.u_pod, normalizer)
            field_out_norm = normalize_u(batch.u_probe, normalizer)
            field_out_phys = batch.u_probe
        else:
            field_in_norm = normalize_f(batch.f_pod, normalizer)
            field_out_norm = normalize_f(batch.f_probe, normalizer)
            field_out_phys = batch.f_probe

        state, loss = single_fe_train_step(
            state,
            field_in_norm,
            field_out_norm,
            batch.probe_coords,
        )

        if step % 500 == 0:
            if field == "u":
                rl2 = single_fe_eval_step_u(
                    state,
                    field_in_norm,
                    field_out_phys,
                    batch.probe_coords,
                    normalizer,
                )
            else:
                rl2 = single_fe_eval_step_f(
                    state,
                    field_in_norm,
                    field_out_phys,
                    batch.probe_coords,
                    normalizer,
                )

            print(
                f"[FE-{field}] step={step:07d} "
                f"loss={float(loss):.4e} "
                f"rl2_{field}={float(rl2):.4e}"
            )

        if step % 10000 == 0:
            save_params(state.params, config.output_dir / save_name)

    save_params(state.params, config.output_dir / save_name)
    return state

# def train_fe(config: AnnulusConfig) -> TrainState:
#     key = jax.random.PRNGKey(config.seed)
#     config.save_json()

#     key, key_pool, key_init = jax.random.split(key, 3)

#     state, model = create_fe_state(config, key_init)

#     pool_size = config.pool_size
#     print(f"[Pool] building {pool_size} fixed batches as Python list ...")
#     batch_pool = build_batch_pool_list(config, key_pool, pool_size=pool_size)
#     print("[Pool] done.")

#     u_scale = 0.1
#     f_scale = 0.001

#     for step in range(config.fe_steps):
#         key, key_select = jax.random.split(key)

#         idx = int(jax.random.randint(
#             key_select,
#             shape=(),
#             minval=0,
#             maxval=pool_size,
#         ))
        
#         # if step % 1000 == 0:
#         #     key, key_select = jax.random.split(key)

#         #     idx = int(jax.random.randint(
#         #         key_select,
#         #         shape=(),
#         #         minval=0,
#         #         maxval=pool_size,
#         #     ))

#         #     batch = batch_pool[idx]

#         batch = batch_pool[idx]

#         state, loss = fe_train_step(
#             state,
#             u_scale * batch.u_pod,
#             f_scale * batch.f_pod,
#             u_scale * batch.u_probe,
#             f_scale * batch.f_probe,
#             batch.probe_coords,
#         )

#         if step % 500 == 0:
#             ru, rf = fe_eval_step(
#                 state,
#                 u_scale * batch.u_pod,
#                 f_scale * batch.f_pod,
#                 u_scale * batch.u_probe,
#                 f_scale * batch.f_probe,
#                 batch.probe_coords,
#             )

#             print(
#                 f'[FE] step={step:07d} '
#                 f'batch_idx={idx:03d} '
#                 f'loss={float(loss):.4e} '
#                 f'rl2_u={float(ru):.4e} '
#                 f'rl2_f={float(rf):.4e}'
#             )

#     save_params(state.params, config.output_dir / 'fe_params.msgpack')
#     return state


def train_ol(config: AnnulusConfig, fe_state: TrainState | None = None, pca_stats: PCAStats | None = None) -> TrainState:
    key = jax.random.PRNGKey(config.seed)
    config.save_json()

    key, key_fe_init, key_ol_init = jax.random.split(key, 3)
    if pca_stats is None:
        pca_stats = load_pca_stats(config.output_dir)

    if fe_state is None:
        fe_state, _ = create_fe_state(config, key_fe_init)
        fe_bytes = (config.output_dir / 'fe_params.msgpack').read_bytes()
        fe_state = fe_state.replace(params=serialization.from_bytes(fe_state.params, fe_bytes))

    ol_state, _ = create_ol_state(config, key_ol_init)

    for step in range(config.ol_steps):
        key, key_batch = jax.random.split(key)
        batch = sample_batch(key_batch, config)

        coeff_u = project_with_pca(batch.u_pod, pca_stats.mean_u, pca_stats.modes_u)
        coeff_f = project_with_pca(batch.f_pod, pca_stats.mean_f, pca_stats.modes_f)

        target_u_latent = fe_state.apply_fn({'params': fe_state.params}, coeff_u, method=FunctionEncoder.encode_u)
        latent_f = fe_state.apply_fn({'params': fe_state.params}, coeff_f, method=FunctionEncoder.encode_f)

        f_tokens = make_source_tokens(latent_f, config)
        cond_tokens = make_condition_tokens(batch, config)

        ol_state, loss = ol_train_step(ol_state, f_tokens, cond_tokens, batch.k_values, target_u_latent)

        if step % 500 == 0:
            r = ol_eval_step(ol_state, f_tokens, cond_tokens, batch.k_values, target_u_latent)
            print(f'[OL] step={step:07d} loss={float(loss):.4e} rl2_latent={float(r):.4e}')

    save_params(ol_state.params, config.output_dir / 'transformer_params.msgpack')
    return ol_state


def run_inference(config: AnnulusConfig, k_value: float, force_zero: bool = True, fe_state: TrainState | None = None, ol_state: TrainState | None = None, pca_stats: PCAStats | None = None) -> dict[str, Array]:
    key = jax.random.PRNGKey(config.seed + 123)
    key, key_fe_init, key_ol_init, key_batch = jax.random.split(key, 4)

    if pca_stats is None:
        pca_stats = load_pca_stats(config.output_dir)

    if fe_state is None:
        fe_state, _ = create_fe_state(config, key_fe_init)
        fe_state = fe_state.replace(params=serialization.from_bytes(fe_state.params, (config.output_dir / 'fe_params.msgpack').read_bytes()))

    if ol_state is None:
        ol_state, _ = create_ol_state(config, key_ol_init)
        ol_state = ol_state.replace(params=serialization.from_bytes(ol_state.params, (config.output_dir / 'transformer_params.msgpack').read_bytes()))

    batch = sample_batch(key_batch, config)
    if force_zero:
        f_pod = jnp.zeros_like(batch.f_pod)
    else:
        f_pod = batch.f_pod

    coeff_f = project_with_pca(f_pod, pca_stats.mean_f, pca_stats.modes_f)
    latent_f = fe_state.apply_fn({'params': fe_state.params}, coeff_f, method=FunctionEncoder.encode_f)
    f_tokens = make_source_tokens(latent_f, config)
    cond_tokens = make_condition_tokens(batch, config)
    k_values = jnp.full((batch.k_values.shape[0],), k_value)

    pred_u_latent = ol_state.apply_fn({'params': ol_state.params}, f_tokens, cond_tokens, k_values)
    pred_u = fe_state.apply_fn({'params': fe_state.params}, pred_u_latent, batch.probe_coords, method=FunctionEncoder.reconstruct)

    out = {
        'u_pred': pred_u,
        'coords': batch.probe_coords,
        'boundary_coords': batch.boundary_coords,
        'boundary_flux': batch.boundary_flux,
    }
    jnp.save(config.output_dir / 'inference_pred.npy', pred_u)
    jnp.save(config.output_dir / 'inference_coords.npy', batch.probe_coords)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='Fixed-annulus SNO pipeline.')
    parser.add_argument('--stage', choices=['fe', 'ol', 'infer'], required=True)
    parser.add_argument('--k_value', type=float, default=1.0)
    parser.add_argument('--run_name', type=str, default='annulus_sno_annulus_only_v1')
    args = parser.parse_args()

    config = AnnulusConfig(run_name=args.run_name)
    if args.stage == 'fe':
        train_fe(config)
    elif args.stage == 'ol':
        train_ol(config)
    else:
        run_inference(config, args.k_value)


if __name__ == '__main__':
    main()
