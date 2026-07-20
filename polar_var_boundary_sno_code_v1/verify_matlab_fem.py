"""Optional MATLAB integration check with an analytic circular-annulus solution."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.io import loadmat, savemat
from scipy.special import iv, kv

from config_varpolar import VarPolarConfig
from fem_monitor import export_fem_manifest, load_fem_monitor, run_matlab_fem_builder


def exact_circle_pressure(theta, radius, inner_radius, k_value):
    outer_radius = 5.0 * inner_radius
    i1_outer = iv(1, k_value * outer_radius)
    k1_outer = kv(1, k_value * outer_radius)
    i1_prime_inner = 0.5 * (
        iv(0, k_value * inner_radius) + iv(2, k_value * inner_radius)
    )
    k1_prime_inner = -0.5 * (
        kv(0, k_value * inner_radius) + kv(2, k_value * inner_radius)
    )
    coefficients = np.linalg.solve(
        np.asarray(
            [
                [i1_outer, k1_outer],
                [-k_value * i1_prime_inner, -k_value * k1_prime_inner],
            ]
        ),
        np.asarray([0.0, 1.0]),
    )
    radial = (
        coefficients[0] * iv(1, k_value * radius)
        + coefficients[1] * kv(1, k_value * radius)
    )
    return radial * np.cos(theta)


def main() -> None:
    config = replace(
        VarPolarConfig(),
        hidden_geom_bnn=16,
        theta_size=16,
        radial_size=12,
        n_basis=16,
        seq_chunks=4,
        cond_chunks=4,
        fem_monitor_size=1,
        fem_eval_theta_size=64,
        fem_eval_radial_size=24,
        fem_mesh_levels=((17, 64), (33, 128)),
        # This smoke check always reaches the second mesh and then accepts it.
        fem_convergence_tol=1.0,
    )
    k_value = 1.1
    with TemporaryDirectory(prefix="varpolar_matlab_") as temporary:
        root = Path(temporary)
        manifest = export_fem_manifest(config, root / "circle_manifest.mat")
        data = {
            key: value
            for key, value in loadmat(manifest).items()
            if not key.startswith("__")
        }
        data["geometry_w1"].fill(0.0)
        data["geometry_b1"].fill(0.0)
        data["geometry_w2"].fill(0.0)
        data["k_values"].fill(k_value)
        data["check_a"].fill(config.geom_base)
        data["check_a_theta"].fill(0.0)
        savemat(manifest, data)

        monitor_path = run_matlab_fem_builder(
            config, manifest, root / "circle_monitor.mat"
        )
        monitor = load_fem_monitor(monitor_path)
        theta = np.pi * (np.asarray(monitor.eval_coords[:, 0]) + 1.0)
        eta = np.asarray(monitor.eval_coords[:, 1])
        radius = config.geom_base * (3.0 + 2.0 * eta)
        exact = exact_circle_pressure(theta, radius, config.geom_base, k_value)
        prediction = np.asarray(monitor.p_eval[0])
        weights = np.asarray(monitor.area_weights[0])
        relative_l2 = np.sqrt(
            np.sum(weights * (prediction - exact) ** 2)
            / np.sum(weights * exact**2)
        )
        correlation = np.corrcoef(prediction, exact)[0, 1]
        print(
            f"circle FEM: area_RL2={relative_l2:.6e}, "
            f"correlation={correlation:.9f}, "
            f"PCG_relres={float(monitor.pcg_relres[0]):.3e}"
        )
        if relative_l2 >= 3.0e-2 or correlation <= 0.99:
            raise RuntimeError("Circular FEM sign/accuracy integration check failed.")


if __name__ == "__main__":
    main()
