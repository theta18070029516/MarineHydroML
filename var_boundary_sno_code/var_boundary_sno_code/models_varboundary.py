from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

from config_varboundary import VarBoundaryConfig


class CNNBranchNet(nn.Module):
    config: VarBoundaryConfig
    out_dim: int
    activation: callable = nn.relu

    @nn.compact
    def __call__(self, field_values):
        """Encode normalized field values on the canonical annulus grid.

        Args:
            field_values: [B, radial_size * theta_size]
        Returns:
            latent coefficients: [B, out_dim]
        """
        B = field_values.shape[0]
        Nr = self.config.radial_size
        Nt = self.config.theta_size
        x = field_values.reshape(B, Nr, Nt, 1)

        for i, channels in enumerate(self.config.cnn_channels):
            stride = self.config.cnn_stride if i < 2 else (2, 2)
            x = nn.Conv(
                features=channels,
                kernel_size=self.config.cnn_kernel_size,
                strides=stride,
                padding="SAME",
            )(x)
            x = nn.LayerNorm()(x)
            x = self.activation(x)

        x = x.reshape((B, -1))
        x = nn.Dense(self.config.cnn_dense_width)(x)
        x = nn.LayerNorm()(x)
        x = self.activation(x)
        x = nn.Dense(self.out_dim)(x)
        return x


class TrunkNet(nn.Module):
    config: VarBoundaryConfig

    @nn.compact
    def __call__(self, coords):
        """Canonical-coordinate trunk basis.

        coords are canonical Cartesian coordinates:
            x_hat = rho cos(theta), y_hat = rho sin(theta)
        """
        width = self.config.trunk_width
        depth = self.config.trunk_depth
        out_dim = self.config.n_basis

        fourier_dim = max(width // 2, 16)
        proj = nn.Dense(
            fourier_dim,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=2.0),
        )(coords)
        x = jnp.concatenate([jnp.sin(proj), jnp.cos(proj), coords], axis=-1)

        for _ in range(depth - 1):
            residual = x
            x = nn.Dense(width)(x)
            x = nn.tanh(x)
            if residual.shape[-1] != x.shape[-1]:
                residual = nn.Dense(width)(residual)
            x = x + residual

        x = nn.Dense(out_dim)(x)
        return x


class FunctionEncoder(nn.Module):
    config: VarBoundaryConfig

    def setup(self):
        self.branch_u = CNNBranchNet(self.config, out_dim=self.config.n_basis)
        self.branch_f = CNNBranchNet(self.config, out_dim=self.config.n_basis)
        self.trunk = TrunkNet(self.config)

    def encode_u(self, u_values):
        return self.branch_u(u_values)

    def encode_f(self, f_values):
        return self.branch_f(f_values)

    def trunk_basis(self, coords):
        return self.trunk(coords)

    def reconstruct(self, latent, coords):
        basis = self.trunk(coords)
        return jnp.einsum("bd,nd->bn", latent, basis) / jnp.sqrt(self.config.n_basis)

    def init_all(self, u_values, f_values, coords):
        z_u = self.encode_u(u_values)
        z_f = self.encode_f(f_values)
        pred_u = self.reconstruct(z_u, coords)
        pred_f = self.reconstruct(z_f, coords)
        return pred_u, pred_f


class MLP(nn.Module):
    dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim)(x)
        return x


class MLPEmbed(nn.Module):
    dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim)(x)
        return x


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
    config: VarBoundaryConfig

    @nn.compact
    def __call__(self, f_tokens, bc_tokens, k_values):
        """Encoder-only Transformer for latent operator learning.

        Args:
            f_tokens: [B, seq_chunks, n_basis / seq_chunks]
            bc_tokens: [B, cond_chunks, boundary_chunk_size * 3]
                       each boundary point contributes [x_b, y_b, induced_flux]
            k_values: [B]
        Returns:
            predicted u latent: [B, n_basis]
        """
        dim = self.config.transformer_dim
        x_f = MLPEmbed(dim=dim, hidden_dim=dim * 2)(f_tokens)
        x_bc = MLPEmbed(dim=dim, hidden_dim=dim * 2)(bc_tokens)
        x_k = nn.Dense(dim)(k_values[:, None, None])

        x = jnp.concatenate([x_f, x_bc, x_k], axis=1)
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
        x = x.mean(axis=1)
        return nn.Dense(self.config.n_basis)(x)
