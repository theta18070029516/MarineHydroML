# -*- coding: utf-8 -*-
"""
Low-sigma prior diagnostic for fixed-annulus SNO.

Purpose
-------
For sigma in {0.5, 1.0, 2.0}:
  1. Generate PI-sampler samples from that single sigma.
  2. Evaluate FE oracle reconstruction accuracy for u and f.
  3. Evaluate SNO zero-shot prediction accuracy for u.
  4. Export all fields, latents, and RL2 diagnostics to a MATLAB .mat file.

Usage example
-------------
python diagnose_low_sigma_annulus_sno.py \
  --project-dir /home/user/data/Hollon/海洋工程水动力/annulus_sno_annulus_only_v2/annulus_sno_annulus_only_v2 \
  --out-dir /home/user/data/Hollon/海洋工程水动力/annulus_sno_annulus_only_v2/out \
  --run-name test \
  --n-per-sigma 128 \
  --k-min 1.0 --k-max 1.0 \
  --r-inner 0.2 --r-outer 1.0 \
  --fe-param-name fe_params_physv2.msgpack \
  --ol-param-name ol_params_physv2.msgpack

Notes
-----
- Keep all architecture-related cfg values identical to the trained FE/OL models.
- Only cfg.sigma_list, cfg.num_repeats, and cfg.sample_size are changed for diagnostics.
- If your saved parameter filenames are different, use --fe-param-name / --ol-param-name.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import savemat


def add_project_dir(project_dir: str | None) -> None:
    if project_dir:
        project_path = Path(project_dir).expanduser().resolve()
        if not project_path.exists():
            raise FileNotFoundError(f"project_dir does not exist: {project_path}")
        sys.path.insert(0, str(project_path))


def import_annulus_modules():
    """Import project modules.

    Preferred names in the user's project are config.py, data.py, models.py, train.py.
    The fallback names match the uploaded files in this ChatGPT workspace.
    """
    try:
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
        from flax import serialization  # noqa: F401

        from config import AnnulusConfig
        from data import (
            make_condition_tokens,
            make_source_tokens,
            normalize_f,
            normalize_u,
            denormalize_f,
            denormalize_u,
            sample_batch,
        )
        from models import FunctionEncoder
        from train import (
            create_fe_state,
            create_ol_state,
            load_field_normalizer,
            rl2_error,
        )
        return {
            "AnnulusConfig": AnnulusConfig,
            "make_condition_tokens": make_condition_tokens,
            "make_source_tokens": make_source_tokens,
            "normalize_f": normalize_f,
            "normalize_u": normalize_u,
            "denormalize_f": denormalize_f,
            "denormalize_u": denormalize_u,
            "sample_batch": sample_batch,
            "FunctionEncoder": FunctionEncoder,
            "create_fe_state": create_fe_state,
            "create_ol_state": create_ol_state,
            "load_field_normalizer": load_field_normalizer,
            "rl2_error": rl2_error,
            "serialization": serialization,
            "jax": jax,
            "jnp": jnp,
        }
    except ImportError:
        # Fallback for the uploaded-file naming convention.
        import jax  # noqa: F401
        import jax.numpy as jnp  # noqa: F401
        from flax import serialization  # noqa: F401

        import config_annulus as config_mod

        # data_annulus.py imports from "config"; train_annulus.py imports from
        # "config", "data", and "models". Register aliases before importing.
        sys.modules.setdefault("config", config_mod)
        import data_annulus as data_mod
        sys.modules.setdefault("data", data_mod)
        import models_annulus as models_mod
        sys.modules.setdefault("models", models_mod)
        import train_annulus as train_mod

        return {
            "AnnulusConfig": config_mod.AnnulusConfig,
            "make_condition_tokens": data_mod.make_condition_tokens,
            "make_source_tokens": data_mod.make_source_tokens,
            "normalize_f": data_mod.normalize_f,
            "normalize_u": data_mod.normalize_u,
            "denormalize_f": data_mod.denormalize_f,
            "denormalize_u": data_mod.denormalize_u,
            "sample_batch": data_mod.sample_batch,
            "FunctionEncoder": models_mod.FunctionEncoder,
            "create_fe_state": train_mod.create_fe_state,
            "create_ol_state": train_mod.create_ol_state,
            "load_field_normalizer": train_mod.load_field_normalizer,
            "rl2_error": train_mod.rl2_error,
            "serialization": serialization,
            "jax": jax,
            "jnp": jnp,
        }


def to_numpy(x):
    # jax.device_get is injected through global MODS after import.
    return np.asarray(MODS["jax"].device_get(x))


def find_param_file(output_dir: Path, explicit_name: str | None, candidates: Iterable[str]) -> Path:
    if explicit_name and explicit_name.lower() != "auto":
        path = output_dir / explicit_name
        if not path.exists():
            raise FileNotFoundError(f"Cannot find parameter file: {path}")
        return path

    for name in candidates:
        path = output_dir / name
        if path.exists():
            return path

    raise FileNotFoundError(
        "Cannot find parameter file. Tried:\n"
        + "\n".join(str(output_dir / name) for name in candidates)
    )


def build_config(args):
    AnnulusConfig = MODS["AnnulusConfig"]
    cfg = AnnulusConfig()

    # Runtime path must match where trained params and normalizer are saved.
    cfg.out_dir = args.out_dir
    cfg.run_name = args.run_name

    # Geometry / PDE. These must match the model/data convention used during training.
    cfg.r_inner = args.r_inner
    cfg.r_outer = args.r_outer
    cfg.k_min = args.k_min
    cfg.k_max = args.k_max

    # Sampling / discretization. These affect model input shapes and must match training.
    cfg.n_basis = args.n_basis
    cfg.theta_size = args.theta_size
    cfg.radial_size = args.radial_size
    cfg.random_probe_points = args.random_probe_points
    cfg.pod_snapshots = args.pod_snapshots

    # FE architecture. Must match training.
    cfg.trunk_width = args.trunk_width
    cfg.trunk_depth = args.trunk_depth
    cfg.cnn_dense_width = args.cnn_dense_width

    # Transformer architecture. Must match training.
    cfg.transformer_dim = args.transformer_dim
    cfg.transformer_heads = args.transformer_heads
    cfg.transformer_layers = args.transformer_layers
    cfg.transformer_mlp_dim = args.transformer_mlp_dim
    cfg.seq_chunks = args.seq_chunks
    cfg.cond_chunks = args.cond_chunks

    # Training metadata used to initialize TrainState optimizers; not used for training here.
    cfg.fe_steps = args.fe_steps
    cfg.ol_steps = args.ol_steps
    cfg.fe_lr = args.fe_lr
    cfg.ol_lr = args.ol_lr
    cfg.weight_decay = args.weight_decay
    cfg.seed = args.seed

    # This script evaluates one sigma at a time, so num_repeats must be 1.
    cfg.num_repeats = 1
    cfg.sample_size = args.n_per_sigma

    return cfg


def load_states(cfg, args):
    jax = MODS["jax"]
    serialization = MODS["serialization"]
    create_fe_state = MODS["create_fe_state"]
    create_ol_state = MODS["create_ol_state"]
    load_field_normalizer = MODS["load_field_normalizer"]

    output_dir = cfg.output_dir
    normalizer = load_field_normalizer(output_dir)

    key = jax.random.PRNGKey(args.seed + 20260623)
    key_fe, key_ol = jax.random.split(key, 2)

    fe_state, _ = create_fe_state(cfg, key_fe)
    ol_state, _ = create_ol_state(cfg, key_ol)

    fe_path = find_param_file(
        output_dir,
        args.fe_param_name,
        candidates=(
            "fe_params_physv2.msgpack",
            "fe_params_phys.msgpack",
            "fe_params.msgpack",
        ),
    )
    ol_path = find_param_file(
        output_dir,
        args.ol_param_name,
        candidates=(
            "ol_params_physv2.msgpack",
            "ol_params.msgpack",
            "transformer_params.msgpack",
        ),
    )

    fe_state = fe_state.replace(
        params=serialization.from_bytes(fe_state.params, fe_path.read_bytes())
    )
    ol_state = ol_state.replace(
        params=serialization.from_bytes(ol_state.params, ol_path.read_bytes())
    )

    print(f"[Loaded] FE params: {fe_path}")
    print(f"[Loaded] OL params: {ol_path}")
    print(
        "[Loaded normalizer] "
        f"mean_u={float(normalizer.mean_u):.6e}, std_u={float(normalizer.std_u):.6e}, "
        f"mean_f={float(normalizer.mean_f):.6e}, std_f={float(normalizer.std_f):.6e}"
    )
    return fe_state, ol_state, normalizer, fe_path, ol_path


def evaluate_one_sigma(cfg_base, sigma_value: float, key, fe_state, ol_state, normalizer):
    jnp = MODS["jnp"]
    sample_batch = MODS["sample_batch"]
    normalize_u = MODS["normalize_u"]
    normalize_f = MODS["normalize_f"]
    denormalize_u = MODS["denormalize_u"]
    denormalize_f = MODS["denormalize_f"]
    make_source_tokens = MODS["make_source_tokens"]
    make_condition_tokens = MODS["make_condition_tokens"]
    FunctionEncoder = MODS["FunctionEncoder"]
    rl2_error = MODS["rl2_error"]

    # Make a private config for this sigma. This keeps labels exact.
    cfg_eval = copy.deepcopy(cfg_base)
    cfg_eval.sigma_list = (float(sigma_value),)
    cfg_eval.num_repeats = 1

    batch = sample_batch(key, cfg_eval)

    # ------------------------------------------------------------
    # 1. FE encode true u and true f from regular POD grid.
    # ------------------------------------------------------------
    u_pod_norm = normalize_u(batch.u_pod, normalizer)
    f_pod_norm = normalize_f(batch.f_pod, normalizer)

    target_u_latent = fe_state.apply_fn(
        {"params": fe_state.params},
        u_pod_norm,
        method=FunctionEncoder.encode_u,
    )
    latent_f = fe_state.apply_fn(
        {"params": fe_state.params},
        f_pod_norm,
        method=FunctionEncoder.encode_f,
    )

    # ------------------------------------------------------------
    # 2. FE oracle reconstruction of u and f on both POD and probe points.
    # ------------------------------------------------------------
    u_fe_pod_norm = fe_state.apply_fn(
        {"params": fe_state.params},
        target_u_latent,
        batch.pod_coords,
        method=FunctionEncoder.reconstruct,
    )
    u_fe_probe_norm = fe_state.apply_fn(
        {"params": fe_state.params},
        target_u_latent,
        batch.probe_coords,
        method=FunctionEncoder.reconstruct,
    )
    f_fe_pod_norm = fe_state.apply_fn(
        {"params": fe_state.params},
        latent_f,
        batch.pod_coords,
        method=FunctionEncoder.reconstruct,
    )
    f_fe_probe_norm = fe_state.apply_fn(
        {"params": fe_state.params},
        latent_f,
        batch.probe_coords,
        method=FunctionEncoder.reconstruct,
    )

    u_fe_pod = denormalize_u(u_fe_pod_norm, normalizer)
    u_fe_probe = denormalize_u(u_fe_probe_norm, normalizer)
    f_fe_pod = denormalize_f(f_fe_pod_norm, normalizer)
    f_fe_probe = denormalize_f(f_fe_probe_norm, normalizer)

    # ------------------------------------------------------------
    # 3. SNO prediction: f latent + boundary condition tokens + k -> u latent.
    # ------------------------------------------------------------
    f_tokens = make_source_tokens(latent_f, cfg_eval)
    cond_tokens = make_condition_tokens(batch, cfg_eval)

    pred_u_latent = ol_state.apply_fn(
        {"params": ol_state.params},
        f_tokens,
        cond_tokens,
        batch.k_values,
    )

    u_sno_pod_norm = fe_state.apply_fn(
        {"params": fe_state.params},
        pred_u_latent,
        batch.pod_coords,
        method=FunctionEncoder.reconstruct,
    )
    u_sno_probe_norm = fe_state.apply_fn(
        {"params": fe_state.params},
        pred_u_latent,
        batch.probe_coords,
        method=FunctionEncoder.reconstruct,
    )

    u_sno_pod = denormalize_u(u_sno_pod_norm, normalizer)
    u_sno_probe = denormalize_u(u_sno_probe_norm, normalizer)

    # ------------------------------------------------------------
    # 4. Per-sample diagnostics.
    # ------------------------------------------------------------
    err_fe_u_pod = rl2_error(u_fe_pod, batch.u_pod)
    err_fe_u_probe = rl2_error(u_fe_probe, batch.u_probe)
    err_fe_f_pod = rl2_error(f_fe_pod, batch.f_pod)
    err_fe_f_probe = rl2_error(f_fe_probe, batch.f_probe)

    err_sno_u_pod = rl2_error(u_sno_pod, batch.u_pod)
    err_sno_u_probe = rl2_error(u_sno_probe, batch.u_probe)
    err_latent = rl2_error(pred_u_latent, target_u_latent)

    # Basic source / boundary magnitudes help diagnose distribution shift.
    norm_u_pod = jnp.linalg.norm(batch.u_pod, axis=-1)
    norm_f_pod = jnp.linalg.norm(batch.f_pod, axis=-1)
    norm_flux = jnp.linalg.norm(batch.boundary_flux, axis=-1)

    print(
        f"[sigma={sigma_value:.3g}] "
        f"FE_u_probe={float(err_fe_u_probe.mean()):.4e}, "
        f"FE_f_probe={float(err_fe_f_probe.mean()):.4e}, "
        f"SNO_u_probe={float(err_sno_u_probe.mean()):.4e}, "
        f"latent={float(err_latent.mean()):.4e}"
    )

    return {
        "sigma": sigma_value,
        "batch": batch,
        "latent_f": latent_f,
        "target_u_latent": target_u_latent,
        "pred_u_latent": pred_u_latent,
        "u_fe_pod": u_fe_pod,
        "u_fe_probe": u_fe_probe,
        "f_fe_pod": f_fe_pod,
        "f_fe_probe": f_fe_probe,
        "u_sno_pod": u_sno_pod,
        "u_sno_probe": u_sno_probe,
        "err_fe_u_pod": err_fe_u_pod,
        "err_fe_u_probe": err_fe_u_probe,
        "err_fe_f_pod": err_fe_f_pod,
        "err_fe_f_probe": err_fe_f_probe,
        "err_sno_u_pod": err_sno_u_pod,
        "err_sno_u_probe": err_sno_u_probe,
        "err_latent": err_latent,
        "norm_u_pod": norm_u_pod,
        "norm_f_pod": norm_f_pod,
        "norm_flux": norm_flux,
    }


def stack_result(results, key: str):
    return np.stack([to_numpy(r[key]) for r in results], axis=0)


def stack_batch_field(results, attr: str):
    return np.stack([to_numpy(getattr(r["batch"], attr)) for r in results], axis=0)


def export_mat(cfg, args, results, fe_path: Path, ol_path: Path):
    out_dir = cfg.output_dir
    mat_path = Path(args.mat_path) if args.mat_path else out_dir / "low_sigma_fe_sno_diagnostic.mat"
    mat_path.parent.mkdir(parents=True, exist_ok=True)

    Nr = cfg.radial_size
    Nt = cfg.theta_size
    n_sigma = len(results)
    B = args.n_per_sigma

    pod_coords = to_numpy(results[0]["batch"].pod_coords)
    probe_coords = to_numpy(results[0]["batch"].probe_coords)
    x_grid = pod_coords[:, 0].reshape(Nr, Nt)
    y_grid = pod_coords[:, 1].reshape(Nr, Nt)

    sigma_values = np.asarray([r["sigma"] for r in results], dtype=np.float64)

    # Summary columns are fixed and documented in summary_columns.
    summary = np.zeros((n_sigma, 15), dtype=np.float64)
    for i, r in enumerate(results):
        summary[i, :] = np.array(
            [
                r["sigma"],
                float(to_numpy(r["err_fe_u_pod"]).mean()),
                float(to_numpy(r["err_fe_u_probe"]).mean()),
                float(to_numpy(r["err_fe_f_pod"]).mean()),
                float(to_numpy(r["err_fe_f_probe"]).mean()),
                float(to_numpy(r["err_sno_u_pod"]).mean()),
                float(to_numpy(r["err_sno_u_probe"]).mean()),
                float(to_numpy(r["err_latent"]).mean()),
                float(to_numpy(r["err_fe_u_probe"]).max()),
                float(to_numpy(r["err_fe_f_probe"]).max()),
                float(to_numpy(r["err_sno_u_probe"]).max()),
                float(to_numpy(r["err_latent"]).max()),
                float(to_numpy(r["norm_u_pod"]).mean()),
                float(to_numpy(r["norm_f_pod"]).mean()),
                float(to_numpy(r["norm_flux"]).mean()),
            ],
            dtype=np.float64,
        )

    summary_columns = np.array(
        [
            "sigma",
            "mean_err_fe_u_pod",
            "mean_err_fe_u_probe",
            "mean_err_fe_f_pod",
            "mean_err_fe_f_probe",
            "mean_err_sno_u_pod",
            "mean_err_sno_u_probe",
            "mean_err_latent",
            "max_err_fe_u_probe",
            "max_err_fe_f_probe",
            "max_err_sno_u_probe",
            "max_err_latent",
            "mean_norm_u_pod",
            "mean_norm_f_pod",
            "mean_norm_boundary_flux",
        ],
        dtype=object,
    )

    # MATLAB shape convention used here:
    #   first dimension  = sigma index
    #   second dimension = sample index
    #   remaining dims   = points / grid / latent dimension
    mat_data = {
        # Metadata
        "sigma_values": sigma_values[:, None],
        "n_sigma": np.array([[n_sigma]], dtype=np.int32),
        "n_per_sigma": np.array([[B]], dtype=np.int32),
        "summary": summary,
        "summary_columns": summary_columns,
        "fe_param_path": np.array(str(fe_path), dtype=object),
        "ol_param_path": np.array(str(ol_path), dtype=object),

        # Coordinates
        "pod_coords": pod_coords,
        "probe_coords": probe_coords,
        "x_grid": x_grid,
        "y_grid": y_grid,

        # True fields: [n_sigma, B, N]
        "u_true_pod": stack_batch_field(results, "u_pod"),
        "f_true_pod": stack_batch_field(results, "f_pod"),
        "u_true_probe": stack_batch_field(results, "u_probe"),
        "f_true_probe": stack_batch_field(results, "f_probe"),
        "boundary_coords": stack_batch_field(results, "boundary_coords"),
        "boundary_flux": stack_batch_field(results, "boundary_flux"),
        "k_values": stack_batch_field(results, "k_values"),

        # FE oracle reconstructions
        "u_fe_pod": stack_result(results, "u_fe_pod"),
        "u_fe_probe": stack_result(results, "u_fe_probe"),
        "f_fe_pod": stack_result(results, "f_fe_pod"),
        "f_fe_probe": stack_result(results, "f_fe_probe"),

        # SNO predictions
        "u_sno_pod": stack_result(results, "u_sno_pod"),
        "u_sno_probe": stack_result(results, "u_sno_probe"),

        # Latents
        "latent_f": stack_result(results, "latent_f"),
        "target_u_latent": stack_result(results, "target_u_latent"),
        "pred_u_latent": stack_result(results, "pred_u_latent"),

        # Per-sample errors: [n_sigma, B]
        "err_fe_u_pod": stack_result(results, "err_fe_u_pod"),
        "err_fe_u_probe": stack_result(results, "err_fe_u_probe"),
        "err_fe_f_pod": stack_result(results, "err_fe_f_pod"),
        "err_fe_f_probe": stack_result(results, "err_fe_f_probe"),
        "err_sno_u_pod": stack_result(results, "err_sno_u_pod"),
        "err_sno_u_probe": stack_result(results, "err_sno_u_probe"),
        "err_latent": stack_result(results, "err_latent"),

        # Distribution-shift helper statistics
        "norm_u_pod": stack_result(results, "norm_u_pod"),
        "norm_f_pod": stack_result(results, "norm_f_pod"),
        "norm_boundary_flux": stack_result(results, "norm_flux"),

        # Config snapshot
        "r_inner": np.array([[cfg.r_inner]], dtype=np.float64),
        "r_outer": np.array([[cfg.r_outer]], dtype=np.float64),
        "k_min": np.array([[cfg.k_min]], dtype=np.float64),
        "k_max": np.array([[cfg.k_max]], dtype=np.float64),
        "radial_size": np.array([[cfg.radial_size]], dtype=np.int32),
        "theta_size": np.array([[cfg.theta_size]], dtype=np.int32),
        "random_probe_points": np.array([[cfg.random_probe_points]], dtype=np.int32),
        "n_basis": np.array([[cfg.n_basis]], dtype=np.int32),
    }

    # Optional grid-shaped fields for convenient MATLAB plotting.
    if args.export_grid_fields:
        mat_data.update(
            {
                "u_true_grid": mat_data["u_true_pod"].reshape(n_sigma, B, Nr, Nt),
                "f_true_grid": mat_data["f_true_pod"].reshape(n_sigma, B, Nr, Nt),
                "u_fe_grid": mat_data["u_fe_pod"].reshape(n_sigma, B, Nr, Nt),
                "f_fe_grid": mat_data["f_fe_pod"].reshape(n_sigma, B, Nr, Nt),
                "u_sno_grid": mat_data["u_sno_pod"].reshape(n_sigma, B, Nr, Nt),
                "u_fe_abs_error_grid": np.abs(
                    mat_data["u_fe_pod"] - mat_data["u_true_pod"]
                ).reshape(n_sigma, B, Nr, Nt),
                "f_fe_abs_error_grid": np.abs(
                    mat_data["f_fe_pod"] - mat_data["f_true_pod"]
                ).reshape(n_sigma, B, Nr, Nt),
                "u_sno_abs_error_grid": np.abs(
                    mat_data["u_sno_pod"] - mat_data["u_true_pod"]
                ).reshape(n_sigma, B, Nr, Nt),
            }
        )

    savemat(mat_path, mat_data, do_compression=True)
    print(f"[Saved] {mat_path}")
    print("\n========== Summary ==========")
    print(summary_columns.tolist())
    print(summary)
    return mat_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose FE and SNO accuracy on low-sigma fixed-annulus PI-sampler samples."
    )
    parser.add_argument("--project-dir", type=str, default=None,
                        help="Directory containing config.py, data.py, models.py, train.py.")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Base output directory containing run_name and trained params.")
    parser.add_argument("--run-name", type=str, required=True,
                        help="Run name subdirectory containing trained params and normalizer.")
    parser.add_argument("--mat-path", type=str, default=None,
                        help="Optional full output .mat path. Default: cfg.output_dir/low_sigma_fe_sno_diagnostic.mat")

    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--n-per-sigma", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)

    # Match your current fixed-annulus experiment by default.
    parser.add_argument("--r-inner", type=float, default=0.2)
    parser.add_argument("--r-outer", type=float, default=1.0)
    parser.add_argument("--k-min", type=float, default=1.0)
    parser.add_argument("--k-max", type=float, default=1.0)

    parser.add_argument("--n-basis", type=int, default=512)
    parser.add_argument("--theta-size", type=int, default=128)
    parser.add_argument("--radial-size", type=int, default=32)
    parser.add_argument("--random-probe-points", type=int, default=1024)
    parser.add_argument("--pod-snapshots", type=int, default=100)

    parser.add_argument("--trunk-width", type=int, default=512)
    parser.add_argument("--trunk-depth", type=int, default=5)
    parser.add_argument("--cnn-dense-width", type=int, default=1024)

    parser.add_argument("--transformer-dim", type=int, default=512)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-layers", type=int, default=4)
    parser.add_argument("--transformer-mlp-dim", type=int, default=1024)
    parser.add_argument("--seq-chunks", type=int, default=32)
    parser.add_argument("--cond-chunks", type=int, default=32)

    parser.add_argument("--fe-steps", type=int, default=300_000)
    parser.add_argument("--ol-steps", type=int, default=200_000)
    parser.add_argument("--fe-lr", type=float, default=1e-3)
    parser.add_argument("--ol-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)

    parser.add_argument("--fe-param-name", type=str, default="auto",
                        help="FE parameter filename, or 'auto'.")
    parser.add_argument("--ol-param-name", type=str, default="auto",
                        help="OL/Transformer parameter filename, or 'auto'.")
    parser.add_argument("--export-grid-fields", action="store_true",
                        help="Also export [n_sigma, B, radial_size, theta_size] grid-shaped fields.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    add_project_dir(args.project_dir)

    # Global module bundle used by helper functions.
    global MODS
    MODS = import_annulus_modules()

    cfg = build_config(args)
    print(f"[Output dir] {cfg.output_dir}")
    print(f"[Sigmas] {args.sigmas}")
    print(f"[n_per_sigma] {args.n_per_sigma}")

    fe_state, ol_state, normalizer, fe_path, ol_path = load_states(cfg, args)

    jax = MODS["jax"]
    key = jax.random.PRNGKey(args.seed + 20260623)
    sigma_keys = jax.random.split(key, len(args.sigmas))

    results = []
    for sigma_value, key_sigma in zip(args.sigmas, sigma_keys):
        result = evaluate_one_sigma(
            cfg_base=cfg,
            sigma_value=float(sigma_value),
            key=key_sigma,
            fe_state=fe_state,
            ol_state=ol_state,
            normalizer=normalizer,
        )
        results.append(result)

    export_mat(cfg, args, results, fe_path, ol_path)
