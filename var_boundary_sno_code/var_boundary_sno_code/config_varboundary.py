from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass(unsafe_hash=True)
class VarBoundaryConfig:
    """Configuration for SNO on variable inner-boundary annular domains.

    Canonical domain:
        rho in [canonical_r_inner, canonical_r_outer] = [0.2, 1.0]

    Physical domain for each sample:
        a(theta) <= r <= b(theta),   b(theta) = outer_scale * a(theta)

    Inner boundary generator:
        a(theta) = geom_base + geom_amp * tanh(geom_tanh_scale * r_BNN(theta))
    """

    # Canonical annulus used by FE/Transformer
    canonical_r_inner: float = 0.2
    canonical_r_outer: float = 1.0

    # Variable physical geometry: a(theta) in approximately (0.04, 0.20)
    geom_base: float = 0.12
    geom_amp: float = 0.08
    geom_tanh_scale: float = 0.3
    outer_scale: float = 5.0

    # PDE parameter range. If fixed k is desired, set k_min = k_max. maybe (0.2, 2.0).
    k_min: float = 1.0
    k_max: float = 1.0

    # Training pool
    fe_pool_size: int = 1000
    use_fe_pool: bool = True
    save_fe_pool: bool = True
    reuse_fe_pool: bool = True
    fe_pool_filename: str = "fe_pool.pkl.gz"

    # PI-sampler priors
    dim: int = 2
    hidden_bnn: int = 256
    hidden_geom_bnn: int = 256
    sigma_list: tuple[float, ...] = (3.0, 5.0, 7.0)
    geom_sigma: float = 3.0
    num_repeats: int = 3
    sample_size: int = 128

    # Sampling and discretization
    theta_size: int = 128
    radial_size: int = 32
    random_probe_points: int = 1024
    normalizer_batches: int = 100
    n_basis: int = 512

    # Function encoder
    trunk_width: int = 512
    trunk_depth: int = 5
    fe_phys_weight: float = 1.0

    # CNN branch
    cnn_channels: tuple[int, ...] = (32, 64, 128)
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
    fe_steps: int = 300_000
    ol_steps: int = 200_000
    fe_lr: float = 1.0e-3
    ol_lr: float = 1.0e-3
    weight_decay: float = 1.0e-6
    fe_b1: float = 0.5
    fe_b2: float = 0.9
    seed: int = 0

    # Runtime
    dtype: str = "float32"
    out_dir: str = "./out_var_boundary_sno"
    run_name: str = "var_boundary_sno_v1"

    def __post_init__(self) -> None:
        if self.n_basis % self.seq_chunks != 0:
            raise ValueError("n_basis must be divisible by seq_chunks.")
        if self.theta_size % self.cond_chunks != 0:
            raise ValueError("theta_size must be divisible by cond_chunks.")
        if self.canonical_r_outer <= self.canonical_r_inner:
            raise ValueError("Require canonical_r_outer > canonical_r_inner.")
        if self.outer_scale <= 1.0:
            raise ValueError("Require outer_scale > 1.")
        if self.k_max < self.k_min:
            raise ValueError("Require k_max >= k_min.")
        if len(self.cnn_channels) != 3:
            raise ValueError("cnn_channels must contain exactly 3 entries.")

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
    def boundary_chunk_size(self) -> int:
        return self.theta_size // self.cond_chunks

    @property
    def cond_chunk_width(self) -> int:
        # Each boundary point contributes [x_b, y_b, induced_flux].
        return self.boundary_chunk_size * 3

    @property
    def n_pod(self) -> int:
        return self.radial_size * self.theta_size

    def save_json(self, path: str | Path | None = None) -> Path:
        out = Path(path) if path is not None else self.output_dir / "config.json"
        out.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return out
