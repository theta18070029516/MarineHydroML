from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Literal
import json
import math


PriorScaling = Literal["raw", "boundary_rms"]
Variant = Literal["p_raw", "p_rms"]


@dataclass(frozen=True)
class PolarAnnulusConfig:
    """Fixed-annulus SNO configuration for the polar pressure-prior ablation.

    Coordinates are stored as ``[theta_hat, r_hat]`` with
    ``theta_hat = theta / pi - 1`` and
    ``r_hat = 2 (r-r_inner)/(r_outer-r_inner) - 1``.  The BNN itself sees the
    periodic features ``[sin(theta), cos(theta), r_hat]``.
    """

    # Experiment switch. This repository deliberately supports pressure only.
    prior_field: Literal["pressure"] = "pressure"
    pressure_prior_scaling: PriorScaling = "raw"

    # Geometry and PDE: Delta P - k^2 P = f.
    r_inner: float = 0.2
    r_outer: float = 1.0
    k_min: float = 1.0
    k_max: float = 1.0

    # polar_v2 prior controls. Each entry is (sigma_theta, sigma_r).
    prior_scale_pairs: tuple[tuple[float, float], ...] = (
        (0.5, 0.5),
        (1.0, 1.0),
        (1.5, 2.0),
    )
    repeats_per_scale: int = 1
    sample_size: int = 256
    hidden_bnn: int = 256
    bnn_bias_sigma: float = 1.0
    bnn_output_sigma: float = 1.0

    # Sampling and discretization.
    theta_size: int = 128
    radial_size: int = 32
    random_probe_points: int = 1024
    normalizer_batches: int = 100
    normalizer_eps: float = 1.0e-6
    n_basis: int = 512

    # Function encoder.
    trunk_width: int = 512
    trunk_depth: int = 5
    fe_phys_weight: float = 0.0
    fe_physics_points: int = 128

    # Periodic CNN branch.
    cnn_channels: tuple[int, int, int] = (32, 64, 128)
    cnn_kernel_size: tuple[int, int] = (3, 5)
    cnn_stride: tuple[int, int] = (1, 2)
    cnn_dense_width: int = 1024

    # Transformer.
    transformer_dim: int = 512
    transformer_heads: int = 8
    transformer_layers: int = 4
    transformer_mlp_dim: int = 2048
    seq_chunks: int = 32
    cond_chunks: int = 32

    # Training. These are the requested complete-run settings.
    fe_steps: int = 300_000
    ol_steps: int = 300_000
    fe_lr: float = 1.0e-3
    ol_lr: float = 1.0e-3
    weight_decay: float = 1.0e-6
    fe_b1: float = 0.5
    fe_b2: float = 0.9
    pool_size: int = 10
    fe_log_interval: int = 500
    ol_log_interval: int = 500
    checkpoint_interval: int = 10_000
    ol_eval_sample_size: int = 32
    ol_eval_probe_points: int = 1024
    seed: int = 0

    # Runtime.
    dtype: str = "float32"
    out_dir: str = "./out_p_prior_ablation"
    run_name: str = "polar_p_raw_seed0"

    def __post_init__(self) -> None:
        if self.prior_field != "pressure":
            raise ValueError("This ablation repository supports prior_field='pressure' only.")
        if self.pressure_prior_scaling not in ("raw", "boundary_rms"):
            raise ValueError("pressure_prior_scaling must be 'raw' or 'boundary_rms'.")
        if not self.r_outer > self.r_inner > 0.0:
            raise ValueError("Require 0 < r_inner < r_outer.")
        if self.k_max < self.k_min:
            raise ValueError("Require k_max >= k_min.")
        if self.sample_size <= 0 or self.hidden_bnn <= 0:
            raise ValueError("sample_size and hidden_bnn must be positive.")
        if not self.prior_scale_pairs:
            raise ValueError("prior_scale_pairs must not be empty.")
        if self.repeats_per_scale <= 0:
            raise ValueError("repeats_per_scale must be positive.")
        if any(st <= 0.0 or sr <= 0.0 for st, sr in self.prior_scale_pairs):
            raise ValueError("All BNN prior scales must be positive.")
        if not 0 < self.fe_physics_points <= self.random_probe_points:
            raise ValueError("Require 0 < fe_physics_points <= random_probe_points.")
        if self.theta_size % self.cond_chunks != 0:
            raise ValueError("theta_size must be divisible by cond_chunks.")
        if self.n_basis % self.seq_chunks != 0:
            raise ValueError("n_basis must be divisible by seq_chunks.")
        if len(self.cnn_channels) != 3:
            raise ValueError("cnn_channels must contain exactly three entries.")
        kr, kt = self.cnn_kernel_size
        if kr <= 0 or kt <= 0 or kr % 2 == 0 or kt % 2 == 0:
            raise ValueError("cnn_kernel_size entries must be positive odd integers.")
        if self.transformer_dim % self.transformer_heads != 0:
            raise ValueError("transformer_dim must be divisible by transformer_heads.")
        if min(
            self.fe_steps,
            self.ol_steps,
            self.fe_log_interval,
            self.ol_log_interval,
            self.checkpoint_interval,
            self.ol_eval_sample_size,
            self.ol_eval_probe_points,
        ) <= 0:
            raise ValueError("Training, logging, checkpoint and evaluation sizes must be positive.")

    @property
    def radial_length(self) -> float:
        return self.r_outer - self.r_inner

    @property
    def drhat_dr(self) -> float:
        return 2.0 / self.radial_length

    @property
    def dthetahat_dtheta(self) -> float:
        return 1.0 / math.pi

    @property
    def effective_batch_size(self) -> int:
        return len(self.prior_scale_pairs) * self.repeats_per_scale * self.sample_size

    @property
    def n_pod(self) -> int:
        return self.radial_size * self.theta_size

    @property
    def seq_chunk_width(self) -> int:
        return self.n_basis // self.seq_chunks

    @property
    def boundary_chunk_size(self) -> int:
        return self.theta_size // self.cond_chunks

    @property
    def boundary_feature_dim(self) -> int:
        return 4  # [sin(theta), cos(theta), r_hat, g_n]

    @property
    def cond_chunk_width(self) -> int:
        return self.boundary_chunk_size * self.boundary_feature_dim

    @property
    def output_dir(self) -> Path:
        path = Path(self.out_dir) / self.run_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def pressure_scale(self, sigma_r: float) -> float:
        """Return c_sigma_r without looking at any target or sampled field."""
        if self.pressure_prior_scaling == "raw":
            return 1.0
        return 1.0 / math.sqrt(1.0 + 4.0 * sigma_r**2)

    def canonical_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return sha256(payload).hexdigest()

    def save_json(self, path: str | Path | None = None) -> Path:
        out = Path(path) if path is not None else self.output_dir / "config.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = self.canonical_dict() | {"config_fingerprint": self.fingerprint()}
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out


def make_ablation_config(
    variant: Variant,
    seed: int = 0,
    out_dir: str | Path = "./out_p_prior_ablation",
) -> PolarAnnulusConfig:
    if variant not in ("p_raw", "p_rms"):
        raise ValueError("variant must be 'p_raw' or 'p_rms'.")
    scaling: PriorScaling = "raw" if variant == "p_raw" else "boundary_rms"
    return PolarAnnulusConfig(
        pressure_prior_scaling=scaling,
        seed=seed,
        out_dir=str(out_dir),
        run_name=f"polar_{variant}_seed{seed}",
    )


def make_smoke_config(config: PolarAnnulusConfig) -> PolarAnnulusConfig:
    """Small, architecture-compatible configuration for CI and CPU checks."""
    return replace(
        config,
        prior_scale_pairs=((1.0, 1.0),),
        sample_size=2,
        hidden_bnn=16,
        theta_size=8,
        radial_size=8,
        random_probe_points=8,
        normalizer_batches=1,
        n_basis=8,
        trunk_width=16,
        trunk_depth=2,
        fe_physics_points=4,
        cnn_channels=(4, 8, 8),
        cnn_dense_width=16,
        transformer_dim=8,
        transformer_heads=2,
        transformer_layers=1,
        transformer_mlp_dim=16,
        seq_chunks=2,
        cond_chunks=2,
        fe_steps=2,
        ol_steps=2,
        fe_log_interval=1,
        ol_log_interval=1,
        checkpoint_interval=1,
        ol_eval_sample_size=1,
        ol_eval_probe_points=8,
        run_name=f"{config.run_name}_smoke",
    )
