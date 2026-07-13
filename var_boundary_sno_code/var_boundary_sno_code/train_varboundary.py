from __future__ import annotations

import pickle
import gzip
from pathlib import Path
from typing import Any
from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import serialization
from flax.training import train_state

from config_varboundary import VarBoundaryConfig
from data_varboundary import (
    FieldNormalizer,
    SampleBatch,
    build_field_normalizer,
    build_field_normalizer_from_pool,
    save_batch_pool,
    load_batch_pool,
    build_batch_pool,
    denormalize_f,
    denormalize_u,
    make_condition_tokens,
    make_source_tokens,
    normalize_f,
    normalize_u,
    physical_to_canonical_single,
    sample_batch,
)
from models_varboundary import FunctionEncoder, OperatorTransformer

Array = jax.Array


class TrainState(train_state.TrainState):
    pass


def rl2_error(pred: Array, ref: Array) -> Array:
    denom = jnp.linalg.norm(ref, axis=-1)
    denom = jnp.clip(denom, a_min=1e-12)
    return jnp.linalg.norm(pred - ref, axis=-1) / denom


def save_params(params: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(serialization.to_bytes(params))


def save_field_normalizer(normalizer: FieldNormalizer, out_dir: Path) -> None:
    jnp.save(out_dir / "norm_mean_u.npy", normalizer.mean_u)
    jnp.save(out_dir / "norm_std_u.npy", normalizer.std_u)
    jnp.save(out_dir / "norm_mean_f.npy", normalizer.mean_f)
    jnp.save(out_dir / "norm_std_f.npy", normalizer.std_f)


def field_normalizer_exists(out_dir: Path) -> bool:
    required_files = [
        out_dir / "norm_mean_u.npy",
        out_dir / "norm_std_u.npy",
        out_dir / "norm_mean_f.npy",
        out_dir / "norm_std_f.npy",
    ]
    return all(path.exists() for path in required_files)


def load_field_normalizer(out_dir: Path) -> FieldNormalizer:
    return FieldNormalizer(
        mean_u=jnp.load(out_dir / "norm_mean_u.npy"),
        std_u=jnp.load(out_dir / "norm_std_u.npy"),
        mean_f=jnp.load(out_dir / "norm_mean_f.npy"),
        std_f=jnp.load(out_dir / "norm_std_f.npy"),
    )


def create_fe_state(config: VarBoundaryConfig, key: Array) -> tuple[TrainState, FunctionEncoder]:
    model = FunctionEncoder(config)
    dummy_field = jnp.ones((1, config.n_pod), dtype=jnp.float32)
    dummy_coords = jnp.ones((config.n_pod, 2), dtype=jnp.float32)
    variables = model.init(key, dummy_field, dummy_field, dummy_coords, method=FunctionEncoder.init_all)
    schedule = optax.cosine_decay_schedule(config.fe_lr, config.fe_steps, alpha=1e-3)
    tx = optax.adamw(
        schedule,
        weight_decay=config.weight_decay,
        b1=config.fe_b1,
        b2=config.fe_b2,
    )
    state = TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)
    return state, model


def create_ol_state(config: VarBoundaryConfig, key: Array) -> tuple[TrainState, OperatorTransformer]:
    model = OperatorTransformer(config)
    dummy_f = jnp.ones((1, config.seq_chunks, config.seq_chunk_width), dtype=jnp.float32)
    dummy_bc = jnp.ones((1, config.cond_chunks, config.cond_chunk_width), dtype=jnp.float32)
    dummy_k = jnp.ones((1,), dtype=jnp.float32)
    variables = model.init(key, dummy_f, dummy_bc, dummy_k)
    schedule = optax.cosine_decay_schedule(config.ol_lr, config.ol_steps, alpha=1e-4)
    tx = optax.adamw(schedule, weight_decay=config.weight_decay)
    state = TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)
    return state, model


def _u_norm_from_latent_at_physical_point(
    params: dict,
    state: TrainState,
    latent_u: Array,
    x_phys: Array,
    geom,
    config: VarBoundaryConfig,
) -> Array:
    """u_hat(canonical(x_phys)) in normalized units.

    This is differentiated with respect to x_phys for the physical Laplacian.
    """
    x_hat = physical_to_canonical_single(x_phys, geom, config)
    basis = state.apply_fn(
        {"params": params},
        x_hat[None, :],
        method=FunctionEncoder.trunk_basis,
    )[0]
    return jnp.dot(latent_u, basis) / jnp.sqrt(config.n_basis)


def physical_residual_loss(
    params: dict,
    state: TrainState,
    latent_u: Array,
    phys_coords: Array,
    f_out_phys: Array,
    k_values: Array,
    geom_params,
    normalizer: FieldNormalizer,
    config: VarBoundaryConfig,
) -> Array:
    """Compute FE PDE loss in the physical domain.

    This is the key difference from the fixed-annulus code: the trunk lives on the
    canonical annulus, but the PDE is evaluated with respect to physical x,y by
    differentiating u_hat(Phi^{-1}(x,y)) with JAX AD.
    """

    def one_sample(z_u, x_phys, f_phys, k, geom):
        def u_norm_of_x(x):
            return _u_norm_from_latent_at_physical_point(
                params=params,
                state=state,
                latent_u=z_u,
                x_phys=x,
                geom=geom,
                config=config,
            )

        u_norm = jax.vmap(u_norm_of_x)(x_phys)
        hess = jax.vmap(jax.hessian(u_norm_of_x))(x_phys)
        lap_u_norm_phys = jnp.trace(hess, axis1=-2, axis2=-1)

        u_phys = denormalize_u(u_norm, normalizer)
        lap_u_phys = normalizer.std_u * lap_u_norm_phys
        residual = lap_u_phys - (k**2) * u_phys - f_phys
        return jnp.mean((residual / normalizer.std_f) ** 2)

    losses = jax.vmap(one_sample)(latent_u, phys_coords, f_out_phys, k_values, geom_params)
    return jnp.mean(losses)


def fe_loss_fn(
    params: dict,
    state: TrainState,
    batch: SampleBatch,
    normalizer: FieldNormalizer,
    config: VarBoundaryConfig,
) -> Array:
    # Inputs to branch networks: normalized field values on canonical POD grid.
    u_in_norm = normalize_u(batch.u_pod, normalizer)
    f_in_norm = normalize_f(batch.f_pod, normalizer)

    # Targets on canonical probe points.
    u_probe_norm = normalize_u(batch.u_probe, normalizer)
    f_probe_norm = normalize_f(batch.f_probe, normalizer)

    latent_u = state.apply_fn({"params": params}, u_in_norm, method=FunctionEncoder.encode_u)
    latent_f = state.apply_fn({"params": params}, f_in_norm, method=FunctionEncoder.encode_f)

    pred_u_norm = state.apply_fn(
        {"params": params},
        latent_u,
        batch.probe_coords,
        method=FunctionEncoder.reconstruct,
    )
    pred_f_norm = state.apply_fn(
        {"params": params},
        latent_f,
        batch.probe_coords,
        method=FunctionEncoder.reconstruct,
    )

    data_loss = jnp.mean((pred_u_norm - u_probe_norm) ** 2) + jnp.mean(
        (pred_f_norm - f_probe_norm) ** 2
    )

    # phys_loss = physical_residual_loss(
    #     params=params,
    #     state=state,
    #     latent_u=latent_u,
    #     phys_coords=batch.probe_phys_coords,
    #     f_out_phys=batch.f_probe,
    #     k_values=batch.k_values,
    #     geom_params=batch.geom_params,
    #     normalizer=normalizer,
    #     config=config,
    # )

    return data_loss #+ config.fe_phys_weight * phys_loss


@partial(jax.jit, static_argnums=(3,))
def fe_train_step(
    state: TrainState,
    batch: SampleBatch,
    normalizer: FieldNormalizer,
    config: VarBoundaryConfig,
) -> tuple[TrainState, Array]:
    loss, grads = jax.value_and_grad(fe_loss_fn)(state.params, state, batch, normalizer, config)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jax.jit
def fe_eval_step(
    state: TrainState,
    batch: SampleBatch,
    normalizer: FieldNormalizer,
) -> tuple[Array, Array]:
    u_in_norm = normalize_u(batch.u_pod, normalizer)
    f_in_norm = normalize_f(batch.f_pod, normalizer)
    latent_u = state.apply_fn({"params": state.params}, u_in_norm, method=FunctionEncoder.encode_u)
    latent_f = state.apply_fn({"params": state.params}, f_in_norm, method=FunctionEncoder.encode_f)
    pred_u_norm = state.apply_fn(
        {"params": state.params}, latent_u, batch.probe_coords, method=FunctionEncoder.reconstruct
    )
    pred_f_norm = state.apply_fn(
        {"params": state.params}, latent_f, batch.probe_coords, method=FunctionEncoder.reconstruct
    )
    pred_u = denormalize_u(pred_u_norm, normalizer)
    pred_f = denormalize_f(pred_f_norm, normalizer)
    return rl2_error(pred_u, batch.u_probe).mean(), rl2_error(pred_f, batch.f_probe).mean()


@jax.jit
def ol_eval_u_error(
    ol_state: TrainState,
    fe_state: TrainState,
    f_tokens: jnp.ndarray,
    bc_tokens: jnp.ndarray,
    k_values: jnp.ndarray,
    batch: SampleBatch,
    normalizer: FieldNormalizer,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate OL prediction error in physical u-field space.

    Transformer predicts latent_u_pred.
    FE trunk reconstructs normalized u_pred on canonical pod grid.
    Then denormalize and compare with batch.u_pod.
    """

    # Transformer latent prediction
    pred_u_latent = ol_state.apply_fn(
        {"params": ol_state.params},
        f_tokens,
        bc_tokens,
        k_values,
    )

    # Reconstruct normalized u on canonical pod coordinates
    u_pred_norm = fe_state.apply_fn(
        {"params": fe_state.params},
        pred_u_latent,
        batch.pod_coords,
        method=FunctionEncoder.reconstruct,
    )

    # Denormalize to physical scale
    u_pred = denormalize_u(u_pred_norm, normalizer)

    # Per-sample relative L2 error
    err_each = rl2_error(u_pred, batch.u_pod)

    # Mean batch error
    err_mean = jnp.mean(err_each)

    return err_mean, err_each


def ol_loss_fn(params: dict, state: TrainState, f_tokens: Array, bc_tokens: Array, k_values: Array, target_u_latent: Array) -> Array:
    pred_u_latent = state.apply_fn({"params": params}, f_tokens, bc_tokens, k_values)
    return jnp.mean((pred_u_latent - target_u_latent) ** 2)


@jax.jit
def ol_train_step(
    state: TrainState,
    f_tokens: Array,
    bc_tokens: Array,
    k_values: Array,
    target_u_latent: Array,
) -> tuple[TrainState, Array]:
    loss, grads = jax.value_and_grad(ol_loss_fn)(state.params, state, f_tokens, bc_tokens, k_values, target_u_latent)
    state = state.apply_gradients(grads=grads)
    return state, loss


def train_fe(config: VarBoundaryConfig) -> tuple[TrainState, FieldNormalizer]:
    out_dir = config.output_dir
    config.save_json(out_dir / "config.json")

    key = jax.random.PRNGKey(config.seed)
    key, key_norm, key_init = jax.random.split(key, 3)

    normalizer = build_field_normalizer(config, key_norm)
    save_field_normalizer(normalizer, out_dir)
    print(
        "[Normalizer]",
        {
            "mean_u": float(normalizer.mean_u),
            "std_u": float(normalizer.std_u),
            "mean_f": float(normalizer.mean_f),
            "std_f": float(normalizer.std_f),
        },
    )

    fe_state, _ = create_fe_state(config, key_init)

    for step in range(1, config.fe_steps+1):
        key, subkey = jax.random.split(key)
        batch = sample_batch(subkey, config)
        fe_state, loss = fe_train_step(fe_state, batch, normalizer, config)

        if step % 500 == 0:
            err_u, err_f = fe_eval_step(fe_state, batch, normalizer)
            print(
                f"[FE {step:07d}] loss={float(loss):.3e}, "
                f"RL2_u={float(err_u):.3e}, RL2_f={float(err_f):.3e}"
            )

        if step % 10000 == 0:
            save_params(fe_state.params, out_dir / "fe_params.msgpack")

    save_params(fe_state.params, out_dir / "fe_params.msgpack")
    return fe_state, normalizer


def train_fe_with_pool(config: VarBoundaryConfig) -> tuple[TrainState, FieldNormalizer]:
    out_dir = config.output_dir
    config.save_json(out_dir / "config.json")

    key = jax.random.PRNGKey(config.seed)
    key, key_pool, key_init, key_train = jax.random.split(key, 4)

    pool_path = out_dir / config.fe_pool_filename

    # ------------------------------------------------------------
    # 1. Load or build pool
    # ------------------------------------------------------------
    if config.reuse_fe_pool and pool_path.exists():
        print(f"[FE] Reusing existing training pool: {pool_path}")
        pool = load_batch_pool(pool_path)
    else:
        print(f"[FE] Building training pool: {config.fe_pool_size} batches")
        pool = build_batch_pool(
            config=config,
            key=key_pool,
            pool_size=config.fe_pool_size,
        )

        if config.save_fe_pool:
            save_batch_pool(pool, pool_path)

    # ------------------------------------------------------------
    # 2. Load or build normalizer
    # ------------------------------------------------------------
    if field_normalizer_exists(out_dir):
        print(f"[FE] Reusing existing field normalizer from: {out_dir}")
        normalizer = load_field_normalizer(out_dir)
    else:
        print("[FE] Building field normalizer from training pool")
        normalizer = build_field_normalizer_from_pool(pool)
        save_field_normalizer(normalizer, out_dir)

    print(
        "[Normalizer]",
        {
            "mean_u": float(normalizer.mean_u),
            "std_u": float(normalizer.std_u),
            "mean_f": float(normalizer.mean_f),
            "std_f": float(normalizer.std_f),
        },
    )

    # ------------------------------------------------------------
    # 3. Create FE state
    # ------------------------------------------------------------
    fe_state, _ = create_fe_state(config, key_init)

    pool_size = len(pool)

    # ------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------
    for step in range(config.fe_steps+1):
        key_train, key_idx = jax.random.split(key_train)
        idx = int(jax.random.randint(key_idx, (), 0, pool_size))

        # Move only selected batch to device.
        batch = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x),
            pool[idx],
        )

        fe_state, loss = fe_train_step(
            fe_state,
            batch,
            normalizer,
            config,
        )

        if step % 500 == 0:
            err_u, err_f = fe_eval_step(fe_state, batch, normalizer)
            print(
                f"[FE {step:07d}] pool_idx={idx:04d}, "
                f"loss={float(loss):.3e}, "
                f"RL2_u={float(err_u):.3e}, RL2_f={float(err_f):.3e}"
            )

        if step > 0 and step % 10000 == 0:
            save_params(fe_state.params, out_dir / "fe_params.msgpack")

    save_params(fe_state.params, out_dir / "fe_params.msgpack")
    return fe_state, normalizer


def train_ol(config: VarBoundaryConfig, fe_state: TrainState | None = None, normalizer: FieldNormalizer | None = None) -> TrainState:
    out_dir = config.output_dir
    key = jax.random.PRNGKey(config.seed + 20260525)
    key, key_fe, key_ol = jax.random.split(key, 3)

    if normalizer is None:
        normalizer = load_field_normalizer(out_dir)

    if fe_state is None:
        fe_state, _ = create_fe_state(config, key_fe)
        fe_path = out_dir / "fe_params.msgpack"
        if not fe_path.exists():
            raise FileNotFoundError(f"Cannot find trained FE params: {fe_path}")
        fe_state = fe_state.replace(params=serialization.from_bytes(fe_state.params, fe_path.read_bytes()))

    ol_state, _ = create_ol_state(config, key_ol)

    for step in range(config.ol_steps):
        key, subkey = jax.random.split(key)
        batch = sample_batch(subkey, config)

        u_pod_norm = normalize_u(batch.u_pod, normalizer)
        f_pod_norm = normalize_f(batch.f_pod, normalizer)

        latent_f = fe_state.apply_fn({"params": fe_state.params}, f_pod_norm, method=FunctionEncoder.encode_f)
        target_u_latent = fe_state.apply_fn({"params": fe_state.params}, u_pod_norm, method=FunctionEncoder.encode_u)

        f_tokens = make_source_tokens(latent_f, config)
        bc_tokens = make_condition_tokens(batch, config)

        ol_state, loss = ol_train_step(ol_state, f_tokens, bc_tokens, batch.k_values, target_u_latent)

        if step % 500 == 0:
            print(f"[OL {step:07d}] loss={float(loss):.3e}")

    save_params(ol_state.params, out_dir / "ol_params.msgpack")
    return ol_state


def train_ol_with_pool(
    config: VarBoundaryConfig,
    fe_state: TrainState | None = None,
    normalizer: FieldNormalizer | None = None,
) -> TrainState:
    out_dir = config.output_dir

    key = jax.random.PRNGKey(config.seed + 20260525)
    key, key_fe, key_ol, key_train = jax.random.split(key, 4)

    # ------------------------------------------------------------
    # 1. Load normalizer
    # ------------------------------------------------------------
    if normalizer is None:
        normalizer = load_field_normalizer(out_dir)

    print(
        "[Normalizer]",
        {
            "mean_u": float(normalizer.mean_u),
            "std_u": float(normalizer.std_u),
            "mean_f": float(normalizer.mean_f),
            "std_f": float(normalizer.std_f),
        },
    )

    # ------------------------------------------------------------
    # 2. Load trained FE
    # ------------------------------------------------------------
    if fe_state is None:
        fe_state, _ = create_fe_state(config, key_fe)

        fe_path = out_dir / "fe_params.msgpack"
        if not fe_path.exists():
            raise FileNotFoundError(f"Cannot find trained FE params: {fe_path}")

        fe_state = fe_state.replace(
            params=serialization.from_bytes(
                fe_state.params,
                fe_path.read_bytes(),
            )
        )

    print(f"[OL] Loaded FE params from: {fe_path}")

    # ------------------------------------------------------------
    # 3. Load training pool
    # ------------------------------------------------------------
    pool_path = out_dir / config.fe_pool_filename

    if not pool_path.exists():
        raise FileNotFoundError(
            f"Cannot find FE training pool: {pool_path}\n"
            f"Please generate and save the pool first."
        )

    print(f"[OL] Loading training pool from: {pool_path}")
    pool = load_batch_pool(pool_path)
    pool_size = len(pool)
    print(f"[OL] Loaded {pool_size} batches.")

    if pool_size == 0:
        raise ValueError("Loaded pool is empty.")

    # ------------------------------------------------------------
    # 4. Create Transformer / OL state
    # ------------------------------------------------------------
    ol_state, _ = create_ol_state(config, key_ol)

    # ------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------
    for step in range(config.ol_steps+1):
        key_train, key_idx = jax.random.split(key_train)

        idx = int(
            jax.random.randint(
                key_idx,
                shape=(),
                minval=0,
                maxval=pool_size,
            )
        )

        # Move only the selected batch to device.
        batch = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x),
            pool[idx],
        )

        # --------------------------------------------------------
        # FE encoding
        # --------------------------------------------------------
        u_pod_norm = normalize_u(batch.u_pod, normalizer)
        f_pod_norm = normalize_f(batch.f_pod, normalizer)

        latent_f = fe_state.apply_fn(
            {"params": fe_state.params},
            f_pod_norm,
            method=FunctionEncoder.encode_f,
        )

        target_u_latent = fe_state.apply_fn(
            {"params": fe_state.params},
            u_pod_norm,
            method=FunctionEncoder.encode_u,
        )

        # FE is frozen during OL training.
        latent_f = jax.lax.stop_gradient(latent_f)
        target_u_latent = jax.lax.stop_gradient(target_u_latent)

        # --------------------------------------------------------
        # Build Transformer tokens
        # --------------------------------------------------------
        f_tokens = make_source_tokens(latent_f, config)
        bc_tokens = make_condition_tokens(batch, config)

        # --------------------------------------------------------
        # One OL training step
        # --------------------------------------------------------
        ol_state, loss = ol_train_step(
            ol_state,
            f_tokens,
            bc_tokens,
            batch.k_values,
            target_u_latent,
        )

        if step % 500 == 0:
            err_u_mean, err_u_each = ol_eval_u_error(
                ol_state=ol_state,
                fe_state=fe_state,
                f_tokens=f_tokens,
                bc_tokens=bc_tokens,
                k_values=batch.k_values,
                batch=batch,
                normalizer=normalizer,
            )
        
            print(
                f"[OL {step:07d}] "
                f"pool_idx={idx:04d}, "
                f"loss={float(loss):.3e}, "
                f"RL2_u={float(err_u_mean):.3e}, "
                f"RL2_u_max={float(jnp.max(err_u_each)):.3e}"
            )

        if step % 10000 == 0:
            save_params(
                ol_state.params,
                out_dir / "ol_params.msgpack",
            )

    save_params(
        ol_state.params,
        out_dir / "ol_params.msgpack",
    )

    return ol_state
