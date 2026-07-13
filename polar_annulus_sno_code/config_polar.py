from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class PolarAnnulusConfig:
    """Configuration for the fixed-annulus SNO in normalized polar coordinates.

    Physical domain:
        theta in [0, 2*pi), r in [r_inner, r_outer]

    Stored DeepONet coordinates:
        theta_hat = theta/pi - 1       in [-1, 1)
        r_hat     = 2(r-r_inner)/L - 1 in [-1, 1]
        L = r_outer-r_inner

    The BNN does not use theta_hat directly. It uses the periodic features
        [sin(theta), cos(theta), r_hat]
    and produces q = partial P / partial r in physical units.
    """

    # Geometry
    r_inner: float = 0.2
    r_outer: float = 1.0

    # PDE: Delta P - k^2 P = f
    k_min: float = 1.0
    k_max: float = 1.0

    # q = P_r prior. Each pair is (sigma_theta, sigma_r).
    # sigma_theta is used for both sin(theta) and cos(theta) features.
    prior_scale_pairs: tuple[tuple[float, float], ...] = (
        (3.0, 3.0),
        (5.0, 5.0),
        (7.0, 7.0),
    )
    repeats_per_scale: int = 1
    sample_size: int = 256
    hidden_bnn: int = 256
    bnn_bias_sigma: float = 1.0
    bnn_output_sigma: float = 1.0

    # Sampling and discretization
    theta_size: int = 128
    radial_size: int = 32
    random_probe_points: int = 1024
    normalizer_batches: int = 100
    normalizer_eps: float = 1.0e-6
    n_basis: int = 512

    # Function encoder
    trunk_width: int = 512
    trunk_depth: int = 5
    fe_phys_weight: float = 0.0
    # Computing second derivatives of all trunk bases is expensive. Use only a
    # fixed prefix of probe points for the FE physics term.
    fe_physics_points: int = 128

    # Periodic CNN branch
    cnn_channels: tuple[int, int, int] = (32, 64, 128)
    cnn_kernel_size: tuple[int, int] = (3, 5)
    cnn_stride: tuple[int, int] = (1, 2)
    cnn_dense_width: int = 1024

    # Transformer
    transformer_dim: int = 512
    transformer_heads: int = 8
    transformer_layers: int = 4
    transformer_mlp_dim: int = 2048
    seq_chunks: int = 32
    cond_chunks: int = 32

    # Training
    fe_steps: int = 500_000
    ol_steps: int = 200_000
    fe_lr: float = 1.0e-3
    ol_lr: float = 1.0e-3
    weight_decay: float = 1.0e-6
    fe_b1: float = 0.5
    fe_b2: float = 0.9
    pool_size: int = 10
    ol_log_interval: int = 500
    ol_checkpoint_interval: int = 10_000
    # Independent monitoring uses a smaller batch and fewer probe coordinates
    # than optimization to avoid doubling the peak accelerator memory.
    ol_eval_sample_size: int = 16
    ol_eval_probe_points: int = 256
    seed: int = 0

    # Runtime
    dtype: str = "float32"
    out_dir: str = "./out_polar_annulus_sno"
    run_name: str = "polar_annulus_sno_v1"

    def __post_init__(self) -> None:
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
            raise ValueError(
                "Require 0 < fe_physics_points <= random_probe_points."
            )
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
        if self.ol_log_interval <= 0:
            raise ValueError("ol_log_interval must be positive.")
        if self.ol_checkpoint_interval <= 0:
            raise ValueError("ol_checkpoint_interval must be positive.")
        if self.ol_eval_sample_size <= 0:
            raise ValueError("ol_eval_sample_size must be positive.")
        if self.ol_eval_probe_points <= 0:
            raise ValueError("ol_eval_probe_points must be positive.")

    @property
    def radial_length(self) -> float:
        return self.r_outer - self.r_inner

    @property
    def drhat_dr(self) -> float:
        """d r_hat / d r."""
        return 2.0 / self.radial_length

    @property
    def dthetahat_dtheta(self) -> float:
        """d theta_hat / d theta."""
        return 1.0 / 3.141592653589793

    @property
    def effective_batch_size(self) -> int:
        return (
            len(self.prior_scale_pairs)
            * self.repeats_per_scale
            * self.sample_size
        )

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
        # [sin(theta), cos(theta), r_hat, g_n]
        return 4

    @property
    def cond_chunk_width(self) -> int:
        return self.boundary_chunk_size * self.boundary_feature_dim

    @property
    def output_dir(self) -> Path:
        path = Path(self.out_dir) / self.run_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_json(self, path: str | Path | None = None) -> Path:
        out = Path(path) if path is not None else self.output_dir / "config.json"
        out.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return out
