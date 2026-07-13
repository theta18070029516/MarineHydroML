# Variable-inner-boundary SNO code

This code is a clean variable-boundary version of the annulus SNO pipeline.

Core design:

- Canonical FE/Transformer domain: `0.2 <= rho <= 1.0`.
- Physical domain: `a(theta) <= r <= b(theta)`, where `b(theta)=5*a(theta)`.
- Geometry generator: `a(theta)=0.08*tanh(0.3*r_BNN(theta))+0.12`, with BNN input `[sin(theta), cos(theta)]`.
- PI-sampler uses `P(x,y)=(r-b(theta))*P_BNN(x,y)`, so the outer Dirichlet condition is hard-enforced.
- Inner boundary flux is induced from the constructed `P`, not imposed.
- FE loss = normalized data reconstruction loss + physical PDE residual loss.
- PDE residual is computed in physical `(x,y)` coordinates by differentiating `u_hat(Phi^{-1}(x,y))`; it does not reuse the fixed-annulus Laplacian formula.

Files:

- `config_varboundary.py`: configuration.
- `data_varboundary.py`: geometry generation, mapping, PI-sampler, normalization, Transformer tokens.
- `models_varboundary.py`: CNN-branch FE and encoder-only Transformer.
- `train_varboundary.py`: FE/OL training and FE physical loss.
- `run_train_varboundary.py`: minimal training entry point.

Important notes:

1. For debugging, start with small `sample_size`, `theta_size`, `radial_size`, `random_probe_points`, and `hidden_bnn`.
2. The FE physical loss is mathematically correct but expensive because it differentiates through the inverse geometry map. Use it after confirming the pure data path works.
3. For final inference with the actual problem condition, construct boundary flux with `target_boundary_flux_from_problem`, which uses `cos(theta)+(a_dot/a)*sin(theta)`.
