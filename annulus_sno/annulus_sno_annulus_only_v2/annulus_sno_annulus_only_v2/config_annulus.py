from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class AnnulusConfig:
    """Configuration for the fixed-annulus SNO pipeline.

    This version treats the annulus itself as the reference domain. No mapping to a
    unit disk or any other canonical geometry is used.
    """

    # Geometry
    r_inner: float = 1.0
    r_outer: float = 5.0

    # PDE parameter range
    k_min: float = 0.05
    k_max: float = 1.5

    # PI-sampler prior
    dim: int = 2
    hidden_bnn: int = 256
    sigma_list: tuple[float, ...] = (3.0, 5.0, 7.0)
    num_repeats: int = 3
    sample_size: int = 256

    # Sampling and discretization
    theta_size: int = 128
    radial_size: int = 32
    random_probe_points: int = 1024
    pod_snapshots: int = 200
    n_basis: int = 512

    # Function encoder
    branch_width: int = 256
    branch_depth: int = 4
    trunk_width: int = 256
    trunk_depth: int = 4

    # FE physics loss
    fe_phys_weight: float = 1.0

    # CNN branch
    branch_type: str = "cnn"   # "mlp" or "cnn"
    cnn_channels: tuple[int, ...] = (32, 64, 128)
    cnn_kernel_size: tuple[int, int] = (3, 5) #int = 5
    cnn_stride: tuple[int, int] = (1, 2)
    cnn_dense_width: int = 1024

    # Transformer
    transformer_dim: int = 256
    transformer_heads: int = 8
    transformer_layers: int = 4
    transformer_mlp_dim: int = 1024
    seq_chunks: int = 32
    cond_chunks: int = 32

    # Training
    fe_b1 = 0.5
    fe_b2 = 0.9
    fe_steps: int = 100_000
    pool_size: int = 100
    ol_steps: int = 150_000
    fe_lr: float = 1e-3
    ol_lr: float = 1e-3
    weight_decay: float = 1e-6
    seed: int = 0

    # Runtime
    dtype: str = "float32"
    out_dir: str = "./out_annulus_sno_annulus_only"
    run_name: str = "annulus_sno_annulus_only_v1"

    def __post_init__(self) -> None:
        if self.n_basis % self.seq_chunks != 0:
            raise ValueError("n_basis must be divisible by seq_chunks.")
        if self.theta_size % self.cond_chunks != 0:
            raise ValueError("theta_size must be divisible by cond_chunks.")
        if self.r_outer <= self.r_inner:
            raise ValueError("Require r_outer > r_inner.")
        if self.k_max <= self.k_min:
            raise ValueError("Require k_max > k_min.")

    @property
    def output_dir(self) -> Path:
        path = Path(self.out_dir) / self.run_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def sigma_array(self) -> tuple[float, ...]:
        return self.sigma_list

    @property
    def seq_chunk_width(self) -> int:
        return self.n_basis // self.seq_chunks

    @property
    def cond_chunk_width(self) -> int:
        return (self.theta_size // self.cond_chunks) * 3

    @property
    def boundary_chunk_size(self) -> int:
        return self.theta_size // self.cond_chunks

    def save_json(self, path: str | Path | None = None) -> Path:
        out = Path(path) if path is not None else self.output_dir / 'config.json'
        out.write_text(json.dumps(asdict(self), indent=2))
        return out
