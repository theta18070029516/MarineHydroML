from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class VarPolarConfig:
    """Configuration for the variable-geometry polar SNO.

    Reference coordinates are ``[theta_hat, r_hat]`` in ``[-1, 1]^2`` with

        theta = pi * (theta_hat + 1)
        r = a(theta) * (3 + 2 * r_hat)

    so ``r_hat=-1`` is the variable inner boundary and ``r_hat=1`` is the
    outer boundary ``r=5*a(theta)``.
    """

    # Geometry generator inherited from var_boundary_sno_code.
    geom_base: float = 0.12
    geom_amp: float = 0.08
    geom_tanh_scale: float = 0.3
    outer_scale: float = 5.0
    hidden_geom_bnn: int = 256
    geom_sigma: float = 3.0

    # PDE: Delta P - k^2 P = f.
    k_min: float = 0.2
    k_max: float = 2.0

    # q=P_r prior. Each sample draws both scales independently.
    sigma_theta_range: tuple[float, float] = (0.5, 2.0)
    sigma_r_range: tuple[float, float] = (0.5, 5.0)
    sample_size: int = 768
    ol_sample_size: int = 384
    prior_generation_chunk_size: int = 256
    ol_prior_generation_chunk_size: int = 128
    prior_point_chunk_size: int = 1024
    hidden_bnn: int = 256
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

    # Periodic CNN branch.
    cnn_channels: tuple[int, int, int] = (32, 64, 128)
    cnn_kernel_size: tuple[int, int] = (3, 5)
    cnn_stride: tuple[int, int] = (1, 2)
    cnn_dense_width: int = 1024

    # Operator Transformer.
    transformer_dim: int = 512
    transformer_heads: int = 8
    transformer_layers: int = 4
    transformer_mlp_dim: int = 2048
    seq_chunks: int = 32
    cond_chunks: int = 32

    # Training and monitoring.
    fe_steps: int = 500_000
    ol_steps: int = 300_000
    fe_lr: float = 1.0e-3
    ol_lr: float = 1.0e-3
    weight_decay: float = 1.0e-6
    fe_b1: float = 0.5
    fe_b2: float = 0.9
    log_interval: int = 500
    fem_eval_interval: int = 5_000
    checkpoint_interval: int = 10_000
    eval_sample_size: int = 16
    eval_probe_points: int = 256
    fem_eval_chunk_size: int = 10
    seed: int = 0

    # Fixed FEM validation set.
    fem_monitor_size: int = 100
    fem_monitor_seed: int = 20_260_716
    fem_eval_radial_size: int = 64
    fem_eval_theta_size: int = 256
    fem_mesh_levels: tuple[tuple[int, int], ...] = (
        (65, 256),
        (129, 512),
        (257, 1024),
    )
    fem_convergence_tol: float = 5.0e-4
    fem_pcg_tol: float = 1.0e-10
    fem_pcg_maxiter: int = 2000
    fem_manifest_filename: str = "fem_monitor_manifest.mat"
    fem_monitor_filename: str = "fem_monitor_100.mat"
    matlab_executable: str = r"C:\Program Files\MATLAB\R2026a\bin\matlab.exe"

    # Runtime.
    dtype: str = "float32"
    out_dir: str = "./out_polar_var_boundary_sno_v1"
    run_name: str = "varpolar_v1"

    def __post_init__(self) -> None:
        if self.outer_scale != 5.0:
            raise ValueError("This implementation requires outer_scale=5.0.")
        if not self.geom_base > self.geom_amp > 0.0:
            raise ValueError("Require geom_base > geom_amp > 0.")
        if not 0.0 < self.k_min <= self.k_max:
            raise ValueError("Require 0 < k_min <= k_max.")
        for name, bounds in (
            ("sigma_theta_range", self.sigma_theta_range),
            ("sigma_r_range", self.sigma_r_range),
        ):
            if len(bounds) != 2 or bounds[0] <= 0.0 or bounds[1] < bounds[0]:
                raise ValueError(f"Invalid {name}: {bounds}.")
        positive = {
            "sample_size": self.sample_size,
            "ol_sample_size": self.ol_sample_size,
            "prior_generation_chunk_size": self.prior_generation_chunk_size,
            "ol_prior_generation_chunk_size": self.ol_prior_generation_chunk_size,
            "prior_point_chunk_size": self.prior_point_chunk_size,
            "theta_size": self.theta_size,
            "radial_size": self.radial_size,
            "random_probe_points": self.random_probe_points,
            "normalizer_batches": self.normalizer_batches,
            "n_basis": self.n_basis,
            "fem_monitor_size": self.fem_monitor_size,
            "fem_eval_chunk_size": self.fem_eval_chunk_size,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.n_basis % self.seq_chunks != 0:
            raise ValueError("n_basis must be divisible by seq_chunks.")
        if self.theta_size % self.cond_chunks != 0:
            raise ValueError("theta_size must be divisible by cond_chunks.")
        if self.transformer_dim % self.transformer_heads != 0:
            raise ValueError("transformer_dim must be divisible by transformer_heads.")
        if len(self.cnn_channels) != 3:
            raise ValueError("cnn_channels must contain exactly three entries.")
        if self.fem_convergence_tol <= 0.0 or self.fem_pcg_tol <= 0.0:
            raise ValueError("FEM tolerances must be positive.")
        if len(self.fem_mesh_levels) < 2:
            raise ValueError("fem_mesh_levels must contain at least two levels.")
        previous_level = (0, 0)
        for level in self.fem_mesh_levels:
            if len(level) != 2 or level[0] < 2 or level[1] < 3:
                raise ValueError(f"Invalid FEM mesh level: {level}.")
            if level[0] <= previous_level[0] or level[1] <= previous_level[1]:
                raise ValueError("FEM mesh levels must be strictly increasing.")
            previous_level = level

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
        return 5

    @property
    def cond_chunk_width(self) -> int:
        return self.boundary_chunk_size * self.boundary_feature_dim

    @property
    def output_dir(self) -> Path:
        path = Path(self.out_dir) / self.run_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def fem_manifest_path(self) -> Path:
        return self.output_dir / self.fem_manifest_filename

    @property
    def fem_monitor_path(self) -> Path:
        return self.output_dir / self.fem_monitor_filename

    def save_json(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.output_dir / "config.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return target

    @classmethod
    def from_json(cls, path: str | Path) -> "VarPolarConfig":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        for name in (
            "sigma_theta_range",
            "sigma_r_range",
            "cnn_channels",
            "cnn_kernel_size",
            "cnn_stride",
        ):
            if name in values:
                values[name] = tuple(values[name])
        if "fem_mesh_levels" in values:
            values["fem_mesh_levels"] = tuple(
                tuple(level) for level in values["fem_mesh_levels"]
            )
        return cls(**values)
