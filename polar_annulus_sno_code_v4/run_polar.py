from __future__ import annotations

import os
from dataclasses import replace

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from config_polar import PolarAnnulusConfig
from train_polar import train_fe, train_operator


def main() -> None:
    fe_config = PolarAnnulusConfig()
    print("Output directory:", fe_config.output_dir)
    print("FE total batch size:", fe_config.effective_batch_size)

    fe_state, normalizer = train_fe(fe_config)
    ol_config = replace(
        fe_config,
        sample_size=384,
        prior_generation_chunk_size=128,
    )
    print("OL total batch size:", ol_config.effective_batch_size)
    train_operator(ol_config, fe_state, normalizer)


if __name__ == "__main__":
    main()
