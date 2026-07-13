"""Minimal entry point for the variable-inner-boundary SNO pretraining."""

import os

# Set these before importing JAX if you need a specific GPU.
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from config_varboundary import VarBoundaryConfig
from train_varboundary import train_fe, train_ol


def main():
    cfg = VarBoundaryConfig()

    # Recommended first debug run: reduce these values in a notebook/script.
    # cfg.sample_size = 4
    # cfg.num_repeats = 1
    # cfg.theta_size = 32
    # cfg.radial_size = 16
    # cfg.random_probe_points = 64
    # cfg.hidden_bnn = 64
    # cfg.hidden_geom_bnn = 64
    # cfg.n_basis = 128
    # cfg.seq_chunks = 16
    # cfg.cond_chunks = 16
    # cfg.fe_steps = 100
    # cfg.ol_steps = 100

    fe_state, normalizer = train_fe(cfg)
    train_ol(cfg, fe_state=fe_state, normalizer=normalizer)


if __name__ == "__main__":
    main()
