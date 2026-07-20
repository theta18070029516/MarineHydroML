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

    # q = P_r prior. Every sample independently draws
    # sigma_theta ~ U[sigma_theta_range] and sigma_r ~ U[sigma_r_range].
    # sigma_theta is used for both sin(theta) and cos(theta) features.
    sigma_theta_range: tuple[float, float] = (3.0, 7.0)
    sigma_r_range: tuple[float, float] = (3.0, 7.0)
    # In v4 sample_size is the total batch size, not a per-scale group size.
    sample_size: int = 768
    # Limit the BNN evaluation working set without changing the batch law.
    prior_generation_chunk_size: int = 256
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
    ol_steps: int = 300_000
    fe_lr: float = 1.0e-3
    ol_lr: float = 1.0e-3
    weight_decay: float = 1.0e-6
    fe_b1: float = 0.5
    fe_b2: float = 0.9
    pool_size: int = 10
    fe_log_interval: int = 500
    fe_checkpoint_interval: int = 10_000
    fe_eval_sample_size: int = 16
    fe_eval_probe_points: int = 256
    fe_exact_eval_interval: int = 5_000
    ol_log_interval: int = 500
    ol_checkpoint_interval: int = 10_000
    # Independent monitoring uses a smaller batch and fewer probe coordinates
    # than optimization to avoid doubling the peak accelerator memory.
    ol_eval_sample_size: int = 16
    ol_eval_probe_points: int = 256
    ol_exact_eval_interval: int = 5_000

    # Fixed analytic monitor. It is never used in the training loss.
    exact_eval_radial_size: int = 64
    exact_eval_theta_size: int = 128
    exact_eval_k: float = 1.0
    exact_eval_mode: int = 1
    exact_eval_phase: float = 0.0
    exact_eval_amplitude: float = 1.0
    exact_eval_save_figure: bool = True
    seed: int = 0

    # Runtime
    dtype: str = "float32"
    out_dir: str = "./out_polar_annulus_sno_v4"
    run_name: str = "polar_v4"

    def __post_init__(self) -> None:
        if not self.r_outer > self.r_inner > 0.0:
            raise ValueError("Require 0 < r_inner < r_outer.")
        if self.k_max < self.k_min:
            raise ValueError("Require k_max >= k_min.")
        if (
            self.sample_size <= 0
            or self.prior_generation_chunk_size <= 0
            or self.hidden_bnn <= 0
        ):
            raise ValueError(
                "sample_size, prior_generation_chunk_size, and hidden_bnn "
                "must be positive."
            )
        for name, bounds in (
            ("sigma_theta_range", self.sigma_theta_range),
            ("sigma_r_range", self.sigma_r_range),
        ):
            if len(bounds) != 2:
                raise ValueError(f"{name} must contain exactly two values.")
            lower, upper = bounds
            if lower <= 0.0 or upper < lower:
                raise ValueError(
                    f"Require 0 < {name}[0] <= {name}[1]."
                )
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
        positive_integer_fields = {
            "fe_log_interval": self.fe_log_interval,
            "fe_checkpoint_interval": self.fe_checkpoint_interval,
            "fe_eval_sample_size": self.fe_eval_sample_size,
            "fe_eval_probe_points": self.fe_eval_probe_points,
            "fe_exact_eval_interval": self.fe_exact_eval_interval,
            "ol_log_interval": self.ol_log_interval,
            "ol_checkpoint_interval": self.ol_checkpoint_interval,
            "ol_eval_sample_size": self.ol_eval_sample_size,
            "ol_eval_probe_points": self.ol_eval_probe_points,
            "ol_exact_eval_interval": self.ol_exact_eval_interval,
            "exact_eval_radial_size": self.exact_eval_radial_size,
            "exact_eval_theta_size": self.exact_eval_theta_size,
        }
        for name, value in positive_integer_fields.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if not self.k_min <= self.exact_eval_k <= self.k_max:
            raise ValueError("exact_eval_k must lie inside the trained k range.")
        if self.exact_eval_k < 0.0:
            raise ValueError("exact_eval_k must be non-negative.")
        if self.exact_eval_mode < 0:
            raise ValueError("exact_eval_mode must be non-negative.")
        if self.exact_eval_amplitude == 0.0:
            raise ValueError("exact_eval_amplitude must be non-zero.")

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
        return self.sample_size

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
