from __future__ import annotations

import argparse
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from config_varpolar import VarPolarConfig
from fem_monitor import (
    export_fem_manifest,
    load_fem_monitor,
    run_matlab_fem_builder,
)
from train_varpolar import train_fe, train_operator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Variable-boundary polar SNO v1 pipeline"
    )
    parser.add_argument(
        "stage",
        choices=("manifest", "fem", "fe", "ol", "all", "test"),
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON previously written by VarPolarConfig.save_json().",
    )
    parser.add_argument("--out-dir", help="Override the configured output root.")
    parser.add_argument("--run-name", help="Override the configured run name.")
    parser.add_argument(
        "--reuse-manifest",
        action="store_true",
        help="For the fem/all stage, reuse an existing fixed manifest.",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> VarPolarConfig:
    config = (
        VarPolarConfig.from_json(args.config)
        if args.config is not None
        else VarPolarConfig()
    )
    overrides = {}
    if args.out_dir is not None:
        overrides["out_dir"] = args.out_dir
    if args.run_name is not None:
        overrides["run_name"] = args.run_name
    return replace(config, **overrides) if overrides else config


def ensure_manifest(config: VarPolarConfig, reuse: bool) -> Path:
    if reuse and config.fem_manifest_path.exists():
        return config.fem_manifest_path
    return export_fem_manifest(config)


def build_fem(config: VarPolarConfig, reuse_manifest: bool) -> Path:
    manifest = ensure_manifest(config, reuse_manifest)
    output = run_matlab_fem_builder(config, manifest, config.fem_monitor_path)
    monitor = load_fem_monitor(output)
    print(
        "FEM monitor:",
        output,
        "cases=",
        monitor.p_pod.shape[0],
        "max_relres=",
        float(monitor.pcg_relres.max()),
        "max_convergence=",
        float(monitor.convergence_error.max()),
    )
    return output


def main() -> None:
    args = parse_args()
    if args.stage == "test":
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return

    config = load_config(args)
    config.save_json()
    print("Output directory:", config.output_dir)
    print("FEM monitor seed:", config.fem_monitor_seed)

    if args.stage == "manifest":
        print("Manifest:", export_fem_manifest(config))
        return
    if args.stage == "fem":
        build_fem(config, args.reuse_manifest)
        return
    if args.stage == "fe":
        train_fe(config)
        return
    if args.stage == "ol":
        train_operator(config)
        return

    build_fem(config, args.reuse_manifest)
    fe_state, normalizer = train_fe(config)
    train_operator(config, fe_state, normalizer)


if __name__ == "__main__":
    main()
