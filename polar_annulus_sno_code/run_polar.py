from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from config_polar import PolarAnnulusConfig
from train_polar import train_fe, train_operator


def main() -> None:
    config = PolarAnnulusConfig()
    config.save_json()
    print("Output directory:", config.output_dir)
    print("Effective PI-sampler batch size:", config.effective_batch_size)

    fe_state, normalizer = train_fe(config)
    train_operator(config, fe_state, normalizer)


if __name__ == "__main__":
    main()
