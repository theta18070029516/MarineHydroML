from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

from config_varpolar import VarPolarConfig
from data_varpolar import periodic_reference_features


def _polar_same_pad(
    x: jnp.ndarray,
    kernel_size: tuple[int, int],
) -> jnp.ndarray:
    """Edge-pad radially and circularly pad the angular direction."""
    pad_r = kernel_size[0] // 2
    pad_theta = kernel_size[1] // 2
    if pad_r:
        x = jnp.pad(
            x,
            ((0, 0), (pad_r, pad_r), (0, 0), (0, 0)),
            mode="edge",
        )
    if pad_theta:
        x = jnp.concatenate(
            [x[:, :, -pad_theta:, :], x, x[:, :, :pad_theta, :]],
            axis=2,
        )
    return x


class PolarCNNBranchNet(nn.Module):
    config: VarPolarConfig
    out_dim: int
    activation: callable = nn.relu

    @nn.compact
    def __call__(self, field_values):
        batch_size = field_values.shape[0]
        x = field_values.reshape(
            batch_size,
            self.config.radial_size,
            self.config.theta_size,
            1,
        )
        for index, channels in enumerate(self.config.cnn_channels):
            stride = self.config.cnn_stride if index < 2 else (2, 2)
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
    config: VarPolarConfig

    @nn.compact
    def __call__(self, coords_hat):
        width = self.config.trunk_width
        features = periodic_reference_features(coords_hat)
        fourier_dim = max(width // 2, 16)
        projection = nn.Dense(
            fourier_dim,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=2.0),
        )(features)
        x = jnp.concatenate(
            [jnp.sin(projection), jnp.cos(projection), features], axis=-1
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
    config: VarPolarConfig

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
        return (
            self.reconstruct_p(latent_p, coords_hat),
            self.reconstruct_f(latent_f, coords_hat),
        )


class MLPEmbed(nn.Module):
    out_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        return nn.Dense(self.out_dim)(x)


class TransformerBlock(nn.Module):
    dim: int
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(x, x)
        x = x + residual
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.mlp_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim)(x)
        return x + residual


class OperatorTransformer(nn.Module):
    config: VarPolarConfig

    @nn.compact
    def __call__(self, f_tokens, boundary_tokens, k_values):
        dim = self.config.transformer_dim
        x_f = MLPEmbed(dim, 2 * dim)(f_tokens)
        x_boundary = MLPEmbed(dim, 2 * dim)(boundary_tokens)
        x_k = nn.Dense(dim)(k_values[:, None, None])
        x = jnp.concatenate([x_f, x_boundary, x_k], axis=1)
        max_len = self.config.seq_chunks + self.config.cond_chunks + 1
        position = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=0.02),
            (1, max_len, dim),
        )
        x = x + position[:, : x.shape[1], :]
        for _ in range(self.config.transformer_layers):
            x = TransformerBlock(
                dim=dim,
                num_heads=self.config.transformer_heads,
                mlp_dim=self.config.transformer_mlp_dim,
            )(x)
        x = nn.LayerNorm()(x)
        return nn.Dense(self.config.n_basis)(jnp.mean(x, axis=1))
