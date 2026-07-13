# Fixed-annulus SNO (annulus-only version)

This version implements the simplified problem on the **fixed annulus**
\(\Omega = \{(x,y): 1 \le \sqrt{x^2+y^2} \le 5\}\) directly.

## Key modeling decisions

1. The annulus itself is the reference domain.
2. No mapping to a unit disk or any other canonical geometry is used.
3. The PI-sampler generates hard-constrained fields
   \[
   P(x,y) = -(r-5)\cos	heta + (r-5)(r-1)^2 U_{raw}(x,y).
   \]
4. The PDE is
   \[
   \Delta P - k^2 P = f.
   \]
5. The inner-boundary outward normal is the outward normal of the fluid domain.
   Therefore at \(r=1\),
   \[
   \partial_n P = -\partial_r P = \cos	heta,
   \]
   so the lifting term must carry the corrected **negative sign**.
6. PCA is performed **separately** for \(P\) and \(f\).
7. The function encoder uses **one trunk**, a **shared branch body**, and **two input heads**.
8. The Transformer input is
   - latent of \(f\),
   - inner boundary token \([x_B, y_B, \cos	heta]\),
   - scalar \(k\) token.

## Files

- `config.py`: configuration
- `data.py`: annulus sampling, PI-sampler, PCA utilities
- `models.py`: function encoder and Transformer
- `train.py`: training/inference entry points, notebook-friendly functions

## Notebook usage

```python
import sys
sys.path.append('./annulus_sno_annulus_only')

from config import AnnulusConfig
from train import train_fe, train_ol, run_inference

cfg = AnnulusConfig(run_name='annulus_nb', out_dir='./out_annulus_nb')
fe_state, pca_stats = train_fe(cfg)
ol_state = train_ol(cfg, fe_state=fe_state, pca_stats=pca_stats)
pred = run_inference(cfg, k_value=1.0, fe_state=fe_state, ol_state=ol_state, pca_stats=pca_stats)
```

## Command line usage

```bash
python train.py --stage fe --run_name annulus_run
python train.py --stage ol --run_name annulus_run
python train.py --stage infer --run_name annulus_run --k_value 1.0
```

## Validation checklist

Before trusting results, verify numerically that:

1. `u_pred` is close to zero on `r=5`.
2. `\partial_r P pprox -\cos	heta` on `r=1`.
3. The induced source term satisfies `f pprox \Delta P - k^2 P`.
4. PCA reconstruction errors for both `u` and `f` are acceptable.


Note: `sample_batch` is intentionally not JIT-compiled with `config` as a static argument, so the mutable notebook-friendly `AnnulusConfig` dataclass works without hash errors.
