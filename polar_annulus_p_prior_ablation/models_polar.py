from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

from config_polar import PolarAnnulusConfig
from data_polar import periodic_polar_features


def _polar_same_pad(
    x: jnp.ndarray,
    kernel_size: tuple[int, int],
) -> jnp.ndarray:
    """Pad radial direction by edge values and theta direction periodically."""
    pad_r = kernel_size[0] // 2
    pad_theta = kernel_size[1] // 2

    if pad_r > 0:
        x = jnp.pad(
            x,
            ((0, 0), (pad_r, pad_r), (0, 0), (0, 0)),
            mode="edge",
        )
    if pad_theta > 0:
        x = jnp.concatenate(
            [x[:, :, -pad_theta:, :], x, x[:, :, :pad_theta, :]],
            axis=2,
        )
    return x


class PolarCNNBranchNet(nn.Module):
    config: PolarAnnulusConfig
    out_dim: int
    activation: callable = nn.relu

    @nn.compact
    def __call__(self, field_values):
        """Encode a field sampled on the regular [Nr, Nt] polar grid."""
        batch_size = field_values.shape[0]
        x = field_values.reshape(
            batch_size,
            self.config.radial_size,
            self.config.theta_size,
            1,
        )

        for block_index, channels in enumerate(self.config.cnn_channels):
            stride = self.config.cnn_stride if block_index < 2 else (2, 2)
            x = _polar_same_pad(x, self.config.cnn_kernel_size)
            x = nn.Conv(
                features=channels,
                kernel_size=self.config.cnn_kernel_size,
                strides=stride,
                padding="VALID",
            )(x)
            x = nn.LayerNorm()(x)
            x = self.activation(x)

        x = x.reshape((batch_size, -1))
        x = nn.Dense(self.config.cnn_dense_width)(x)
        x = nn.LayerNorm()(x)
        x = self.activation(x)
        return nn.Dense(self.out_dim)(x)


class PolarTrunkNet(nn.Module):
    config: PolarAnnulusConfig

    @nn.compact
    def __call__(self, coords_hat):
        """Continuous basis on normalized polar coordinates.

        coords_hat has columns [theta_hat, r_hat]. The raw theta_hat is not sent
        into the MLP. It is converted to periodic [sin(theta), cos(theta)].
        """
        width = self.config.trunk_width
        features = periodic_polar_features(coords_hat)

        fourier_dim = max(width // 2, 16)
        projection = nn.Dense(
            fourier_dim,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=2.0),
        )(features)
        x = jnp.concatenate(
            [jnp.sin(projection), jnp.cos(projection), features],
            axis=-1,
        )

        for _ in range(self.config.trunk_depth - 1):
            residual = x
            x = nn.Dense(width)(x)
            x = nn.tanh(x)
            if residual.shape[-1] != x.shape[-1]:
                residual = nn.Dense(width)(residual)
            x = x + residual

        return nn.Dense(self.config.n_basis)(x)


class FunctionEncoder(nn.Module):
    config: PolarAnnulusConfig

    def setup(self):
        self.branch_p = PolarCNNBranchNet(self.config, self.config.n_basis)
        self.branch_f = PolarCNNBranchNet(self.config, self.config.n_basis)
        self.trunk = PolarTrunkNet(self.config)

    def encode_p(self, p_values):
        return self.branch_p(p_values)

    def encode_f(self, f_values):
        return self.branch_f(f_values)

    def trunk_raw_basis(self, coords_hat):
        return self.trunk(coords_hat)

    def trunk_p_basis(self, coords_hat):
        # r_hat=1 is the physical outer boundary. The affine mask enforces
        # P(r_outer, theta)=0 exactly for every latent vector.
        outer_mask = 0.5 * (1.0 - coords_hat[..., 1:2])
        return outer_mask * self.trunk(coords_hat)

    def reconstruct_p(self, latent, coords_hat):
        basis = self.trunk_p_basis(coords_hat)
        return jnp.einsum("bd,nd->bn", latent, basis) / jnp.sqrt(
            self.config.n_basis
        )

    def reconstruct_f(self, latent, coords_hat):
        basis = self.trunk_raw_basis(coords_hat)
        return jnp.einsum("bd,nd->bn", latent, basis) / jnp.sqrt(
            self.config.n_basis
        )

    def init_all(self, p_values, f_values, coords_hat):
        latent_p = self.encode_p(p_values)
        latent_f = self.encode_f(f_values)
        pred_p = self.reconstruct_p(latent_p, coords_hat)
        pred_f = self.reconstruct_f(latent_f, coords_hat)
        return pred_p, pred_f


class MLP(nn.Module):
    dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        return nn.Dense(self.dim)(x)


class MLPEmbed(nn.Module):
    dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        return nn.Dense(self.dim)(x)


class TransformerBlock(nn.Module):
    dim: int
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x):
        x = x + nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim,
        )(nn.LayerNorm()(x))
        x = x + MLP(self.dim, self.mlp_dim)(nn.LayerNorm()(x))
        return x


class OperatorTransformer(nn.Module):
    config: PolarAnnulusConfig

    @nn.compact
    def __call__(self, f_tokens, boundary_tokens, k_values):
        dim = self.config.transformer_dim
        x_f = MLPEmbed(dim, 2 * dim)(f_tokens)
        x_boundary = MLPEmbed(dim, 2 * dim)(boundary_tokens)
        x_k = nn.Dense(dim)(k_values[:, None, None])

        x = jnp.concatenate([x_f, x_boundary, x_k], axis=1)
        max_len = self.config.seq_chunks + self.config.cond_chunks + 1
        pos = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, max_len, dim),
        )
        x = x + pos[:, : x.shape[1], :]

        for _ in range(self.config.transformer_layers):
            x = TransformerBlock(
                dim=dim,
                num_heads=self.config.transformer_heads,
                mlp_dim=self.config.transformer_mlp_dim,
            )(x)

        x = nn.LayerNorm()(x)
        x = jnp.mean(x, axis=1)
        return nn.Dense(self.config.n_basis)(x)
