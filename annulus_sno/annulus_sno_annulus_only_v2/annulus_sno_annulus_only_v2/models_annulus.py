from __future__ import annotations

import jax.numpy as jnp
from flax import linen as nn

from config import AnnulusConfig

class CNNBranchNet(nn.Module):
    config: AnnulusConfig
    out_dim: int
    activation: callable = nn.relu

    @nn.compact
    def __call__(self, field_values):
        """
        field_values:
            [B, Npod] where Npod = radial_size * theta_size
        return:
            [B, out_dim]
        """
        B = field_values.shape[0]
        Nr = self.config.radial_size
        Nt = self.config.theta_size

        x = field_values.reshape(B, Nr, Nt, 1)

        # Conv block 1
        x = nn.Conv(
            features=self.config.cnn_channels[0], #config.cnn_channels(0) #32
            kernel_size=self.config.cnn_kernel_size, #(3, 5)
            strides=self.config.cnn_stride, #(1, 2)
            padding='SAME',
        )(x)
        x = nn.LayerNorm()(x)
        x = self.activation(x)

        # Conv block 2
        x = nn.Conv(
            features=self.config.cnn_channels[1], #config.cnn_channels(1) #64
            kernel_size=self.config.cnn_kernel_size, #(3, 5)
            strides=self.config.cnn_stride, #(1, 2)
            padding='SAME',
        )(x)
        x = nn.LayerNorm()(x)
        x = self.activation(x)

        # Conv block 3
        x = nn.Conv(
            features=self.config.cnn_channels[2], #config.cnn_channels(2) #128
            kernel_size=self.config.cnn_kernel_size, #(3, 5)
            strides=(2, 2),
            padding='SAME',
        )(x)
        x = nn.LayerNorm()(x)
        x = self.activation(x)

        x = x.reshape((B, -1))

        # x = nn.Dense(self.config.branch_width)(x)
        x = nn.Dense(self.config.cnn_dense_width)(x)
        x = nn.LayerNorm()(x)
        x = self.activation(x)

        x = nn.Dense(self.out_dim)(x)
        return x
        
class ResidualMLP(nn.Module):
    width: int
    depth: int
    out_dim: int
    activation: callable = nn.gelu

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth - 1):
            residual = x
            x = nn.Dense(self.width)(x)
            x = self.activation(x)
            if residual.shape[-1] != x.shape[-1]:
                residual = nn.Dense(self.width)(residual)
            x = x + residual
        x = nn.Dense(self.out_dim)(x)
        return x


class TrunkNet(nn.Module):
    config: AnnulusConfig

    @nn.compact
    def __call__(self, coords):
        width = self.config.trunk_width
        depth = self.config.trunk_depth
        out_dim = self.config.n_basis

        fourier_dim = max(width // 2, 16)
        proj = nn.Dense(fourier_dim, use_bias=False, kernel_init=nn.initializers.normal(stddev=2.0))(coords)
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


# class FunctionEncoder(nn.Module):
#     config: AnnulusConfig

#     def setup(self):
#         self.head_u = ResidualMLP(width=self.config.branch_width, depth=3, out_dim=self.config.branch_width)
#         self.head_f = ResidualMLP(width=self.config.branch_width, depth=3, out_dim=self.config.branch_width)
#         self.shared_body = ResidualMLP(
#             width=self.config.branch_width,
#             depth=self.config.branch_depth,
#             out_dim=self.config.n_basis,
#         )
#         self.trunk = TrunkNet(self.config)

#     def encode_u(self, coeffs_u):
#         x = self.head_u(coeffs_u)
#         return self.shared_body(x)

#     def encode_f(self, coeffs_f):
#         x = self.head_f(coeffs_f)
#         return self.shared_body(x)

#     def reconstruct(self, latent, coords):
#         basis = self.trunk(coords)
#         return jnp.einsum('bd,nd->bn', latent, basis) / jnp.sqrt(self.config.n_basis)

#     def init_all(self, coeffs_u, coeffs_f, coords):
#         latent_u = self.encode_u(coeffs_u)
#         latent_f = self.encode_f(coeffs_f)
#         pred_u = self.reconstruct(latent_u, coords)
#         pred_f = self.reconstruct(latent_f, coords)
#         return pred_u, pred_f

class FunctionEncoder(nn.Module):
    config: AnnulusConfig

    def setup(self):
        self.branch_u = CNNBranchNet(
            config=self.config,
            out_dim=self.config.n_basis,
        )
        self.branch_f = CNNBranchNet(
            config=self.config,
            out_dim=self.config.n_basis,
        )
        self.trunk = TrunkNet(self.config)

    def encode_u(self, u_values):
        """
        u_values: [B, Npod]
        """
        return self.branch_u(u_values)

    def encode_f(self, f_values):
        """
        f_values: [B, Npod]
        """
        return self.branch_f(f_values)

    def trunk_basis(self, coords):
        return self.trunk(coords)

    def reconstruct(self, latent, coords):
        basis = self.trunk(coords)
        return jnp.einsum('bd,nd->bn', latent, basis) / jnp.sqrt(self.config.n_basis)

    def init_all(self, u_values, f_values, coords):
        latent_u = self.encode_u(u_values)
        latent_f = self.encode_f(f_values)
        pred_u = self.reconstruct(latent_u, coords)
        pred_f = self.reconstruct(latent_f, coords)
        return pred_u, pred_f

class SingleFieldFunctionEncoder(nn.Module):
    config: AnnulusConfig
    field_name: str = "u"

    def setup(self):
        self.branch = CNNBranchNet(
            config=self.config,
            out_dim=self.config.n_basis,
        )
        self.trunk = TrunkNet(self.config)

    def encode(self, field_values):
        """
        field_values:
            [B, Npod], normalized field values on regular annulus grid.
        return:
            latent coefficients [B, n_basis].
        """
        return self.branch(field_values)

    def reconstruct(self, latent, coords):
        """
        latent:
            [B, n_basis]
        coords:
            [Nquery, 2]
        return:
            predicted normalized field [B, Nquery]
        """
        basis = self.trunk(coords)
        return jnp.einsum(
            "bd,nd->bn",
            latent,
            basis,
        ) / jnp.sqrt(self.config.n_basis)

    def init_all(self, field_values, coords):
        latent = self.encode(field_values)
        pred = self.reconstruct(latent, coords)
        return pred


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
    dim: int        # 输出维度（= Transformer dim）
    hidden_dim: int # 中间隐藏层维度

    @nn.compact
    def __call__(self, x):
        # x: [B, L, D_in]
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim)(x)  # 输出为 dim
        return x

class TransformerBlock(nn.Module):
    dim: int
    num_heads: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x):
        x = x + nn.SelfAttention(num_heads=self.num_heads, qkv_features=self.dim)(nn.LayerNorm()(x))
        x = x + MLP(self.dim, self.mlp_dim)(nn.LayerNorm()(x))
        return x


class OperatorTransformer(nn.Module):
    config: AnnulusConfig

    @nn.compact
    def __call__(self, f_tokens, bc_tokens, k_values):
        dim = self.config.transformer_dim
        x_f = MLPEmbed(dim=dim, hidden_dim=dim * 2)(f_tokens) #nn.Dense(dim)(f_tokens)
        x_bc = MLPEmbed(dim=dim, hidden_dim=dim * 2)(bc_tokens) #nn.Dense(dim)(bc_tokens)
        x_k = nn.Dense(dim)(k_values[:, None, None])
        x = jnp.concatenate([x_f, x_bc, x_k], axis=1)

        max_len = self.config.seq_chunks + self.config.cond_chunks + 1
        pos = self.param('pos_embedding', nn.initializers.normal(stddev=0.02), (1, max_len, dim))
        x = x + pos[:, :x.shape[1], :]

        for _ in range(self.config.transformer_layers):
            x = TransformerBlock(dim=dim, num_heads=self.config.transformer_heads, mlp_dim=self.config.transformer_mlp_dim)(x)

        x = nn.LayerNorm()(x)
        x = x.mean(axis=1)
        x = nn.Dense(self.config.n_basis)(x)
        return x
