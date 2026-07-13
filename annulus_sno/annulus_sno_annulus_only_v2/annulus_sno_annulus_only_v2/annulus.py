##########################################
############## 训练 FE ###################
##########################################
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import serialization

PROJECT_DIR = "/home/user/data/Hollon/海洋工程水动力/annulus_sno_annulus_only_v2/annulus_sno_annulus_only_v2"
sys.path.append(PROJECT_DIR)

from config import AnnulusConfig
from train import (
    create_fe_state,
    load_field_normalizer,
    save_params,
    fe_train_step,
    fe_eval_step,
    fe_physics_eval_step,
    train_ol,
)
from data import (
    sample_batch,
    normalize_u,
    normalize_f,
)


cfg = AnnulusConfig()
cfg.run_name = "test"
cfg.out_dir = "/home/user/data/Hollon/海洋工程水动力/annulus_sno_annulus_only_v2/out"

# Geometry
cfg.r_inner = 0.2
cfg.r_outer = 1.0

# Sampling and discretization
cfg.n_basis = 512
cfg.theta_size = 128
cfg.radial_size = 32
cfg.pod_snapshots = 100
cfg.random_probe_points = 1024

# PDE parameter range
cfg.k_min = 1.0
cfg.k_max = 1.0

# PI-sampler prior
cfg.sigma_list = (3.0, 5.0, 7.0)
cfg.sample_size = 256

# Function encoder
# cfg.branch_width = 256
# cfg.branch_depth = 4
cfg.trunk_width = 512
cfg.trunk_depth = 5

# CNN branch
cfg.cnn_dense_width = 1024

# Training
cfg.fe_steps = 300_000
cfg.ol_steps = 200_000
# cfg.pool_size = 10

# Transformer
# cfg.transformer_layers = 6
cfg.transformer_dim = 512


fe_state, normalizer = train_fe(cfg)


### 导出随机生成的 u 和 f 的分布

import jax
import jax.numpy as jnp
from scipy.io import savemat

from config import AnnulusConfig
from data import sample_batch  

cfg = AnnulusConfig()
cfg.sample_size = 10  # 指定生成样本的数量

key = jax.random.PRNGKey(0)
batch = sample_batch(key, cfg)

u = jnp.array(batch.u_pod)  # 或 batch.u_pod
f = jnp.array(batch.f_pod)  # 或 batch.f_pod
coords = jnp.array(batch.pod_coords)

savemat("/home/user/data/Hollon/海洋工程水动力/annulus_sno_annulus_only_v2/out/test/sample_u_f.mat", {
    "u": u,
    "f": f,
    "coords": coords,
})


import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from scipy.io import savemat

# ===== 1. 路径设置 =====
sys.path.append("/home/user/data/Hollon/海洋工程水动力/annulus_sno_annulus_only_v2/annulus_sno_annulus_only_v2")

from config import AnnulusConfig
from data import (
    sample_batch,
    normalize_u,
    normalize_f,
    denormalize_u,
    denormalize_f,
)
from train import (
    create_fe_state,
    rl2_error,
)

from train import load_field_normalizer

# 如果你的 train.py 里已经有 load_field_normalizer，就直接导入
# try:
#     from train import load_field_normalizer
# except ImportError:
#     from data import FieldNormalizer

#     def load_field_normalizer(out_dir):
#         return FieldNormalizer(
#             mean_u=jnp.asarray(jnp.load(out_dir / "norm_mean_u.npy")),
#             std_u=jnp.asarray(jnp.load(out_dir / "norm_std_u.npy")),
#             mean_f=jnp.asarray(jnp.load(out_dir / "norm_mean_f.npy")),
#             std_f=jnp.asarray(jnp.load(out_dir / "norm_std_f.npy")),
#         )


# ===== 2. 配置必须和训练 FE 时保持一致 =====
cfg = AnnulusConfig()
cfg.run_name = "test"
cfg.out_dir = "/home/user/data/Hollon/海洋工程水动力/annulus_sno_annulus_only_v2/out"

# 这些参数必须与你训练 fe_params.msgpack 时一致
cfg.r_inner = 0.2
cfg.r_outer = 1.0

cfg.n_basis = 512
cfg.theta_size = 128
cfg.radial_size = 32
cfg.random_probe_points = 1024

cfg.k_min = 0.02
cfg.k_max = 0.02

cfg.sigma_list = (3.0, 5.0, 7.0)
cfg.sample_size = 10          # 这里只生成 1 个样本，方便导出和画图

# cfg.branch_width = 256
# cfg.branch_depth = 4
cfg.trunk_width = 512
cfg.trunk_depth = 5

# CNN branch
cfg.cnn_dense_width = 1024

out_dir = cfg.output_dir

print("Output dir:", out_dir)


# ===== 3. 加载 normalizer =====
normalizer = load_field_normalizer(out_dir)

print("[Normalizer]")
print("mean_u =", float(normalizer.mean_u), "std_u =", float(normalizer.std_u))
print("mean_f =", float(normalizer.mean_f), "std_f =", float(normalizer.std_f))


# ===== 4. 初始化模型结构并加载 FE 参数 =====
key = jax.random.PRNGKey(cfg.seed)
key, key_init, key_batch = jax.random.split(key, 3)

fe_state, fe_model = create_fe_state(cfg, key_init)

# 注意：如果你的文件名确实是 fe_params.nsgpack，把这里改成对应名字
param_path = out_dir / "fe_params.msgpack"

if not param_path.exists():
    raise FileNotFoundError(f"Cannot find FE parameter file: {param_path}")

with open(param_path, "rb") as f:
    loaded_params = serialization.from_bytes(fe_state.params, f.read())

fe_state = fe_state.replace(params=loaded_params)

print("Loaded FE params from:", param_path)


# ===== 5. 随机生成 1 个 batch =====
batch = sample_batch(key_batch, cfg)

# 规则网格坐标和原始物理场
grid = batch.pod_coords          # [Npod, 2]
u_ref = batch.u_pod              # [1, Npod]
f_ref = batch.f_pod              # [1, Npod]

print("grid shape:", grid.shape)
print("u_ref shape:", u_ref.shape)
print("f_ref shape:", f_ref.shape)


# ===== 6. 在规则网格上重建 u / f =====
# 输入 branch net 前：归一化
u_in_norm = normalize_u(u_ref, normalizer)
f_in_norm = normalize_f(f_ref, normalizer)

# 得到 branch latent
latent_u = fe_state.apply_fn(
    {"params": fe_state.params},
    u_in_norm,
    method=fe_model.encode_u,
)

latent_f = fe_state.apply_fn(
    {"params": fe_state.params},
    f_in_norm,
    method=fe_model.encode_f,
)

# 在规则网格 grid 上用 trunk 重建
u_recon_norm = fe_state.apply_fn(
    {"params": fe_state.params},
    latent_u,
    grid,
    method=fe_model.reconstruct,
)

f_recon_norm = fe_state.apply_fn(
    {"params": fe_state.params},
    latent_f,
    grid,
    method=fe_model.reconstruct,
)

# 反归一化，回到物理尺度
u_recon = denormalize_u(u_recon_norm, normalizer)
f_recon = denormalize_f(f_recon_norm, normalizer)


# ===== 7. 计算物理空间 RL2 重建误差 =====
rl2_u_each = rl2_error(u_recon, u_ref)   # [20]
rl2_f_each = rl2_error(f_recon, f_ref)   # [20]

rl2_u_mean = rl2_error(u_recon, u_ref).mean()
rl2_f_mean = rl2_error(f_recon, f_ref).mean()

print(f"[FE reconstruction on regular grid]")
print(f"RL2_u = {float(rl2_u_mean):.6e}")
print(f"RL2_f = {float(rl2_f_mean):.6e}")


# ===== 8. reshape 成规则网格形式，方便 MATLAB 画图 =====
Nr = cfg.radial_size
Nt = cfg.theta_size

grid_np = np.asarray(grid)
x_grid = grid_np[:, 0].reshape(Nr, Nt)
y_grid = grid_np[:, 1].reshape(Nr, Nt)

u_ref_np = np.asarray(u_ref)
f_ref_np = np.asarray(f_ref)
u_recon_np = np.asarray(u_recon)
f_recon_np = np.asarray(f_recon)

u_ref_grid = u_ref_np.reshape(cfg.num_repeats * cfg.sample_size, Nr, Nt)
f_ref_grid = f_ref_np.reshape(cfg.num_repeats * cfg.sample_size, Nr, Nt)
u_recon_grid = u_recon_np.reshape(cfg.num_repeats * cfg.sample_size, Nr, Nt)
f_recon_grid = f_recon_np.reshape(cfg.num_repeats * cfg.sample_size, Nr, Nt)

u_error_grid = u_recon_grid - u_ref_grid
f_error_grid = f_recon_grid - f_ref_grid


# ===== 9. 导出 MATLAB 文件 =====
mat_path = out_dir / "fe_reconstruction_check.mat"

savemat(
    mat_path,
    {
        # vector format
        "grid": grid_np,                         # [Npod, 2]
        "u_ref": u_ref_np,                       # [Npod]
        "f_ref": f_ref_np,
        "u_recon": u_recon_np,
        "f_recon": f_recon_np,
        "u_error": u_recon_np - u_ref_np,
        "f_error": f_recon_np - f_ref_np,

        # grid format: [radial_size, theta_size]
        "x_grid": x_grid,
        "y_grid": y_grid,
        "u_ref_grid": u_ref_grid,
        "f_ref_grid": f_ref_grid,
        "u_recon_grid": u_recon_grid,
        "f_recon_grid": f_recon_grid,
        "u_error_grid": u_error_grid,
        "f_error_grid": f_error_grid,

        # scalar diagnostics
        "rl2_u_each": np.asarray(rl2_u_each),
        "rl2_f_each": np.asarray(rl2_f_each),
        "rl2_u_mean": np.asarray(float(rl2_u_mean)),
        "rl2_f_mean": np.asarray(float(rl2_f_mean)),
        "r_inner": np.asarray(cfg.r_inner),
        "r_outer": np.asarray(cfg.r_outer),
        "radial_size": np.asarray(cfg.radial_size),
        "theta_size": np.asarray(cfg.theta_size),
    },
)

print("Saved MATLAB file to:", mat_path)


#############################################
################# 训练 OL ###################
#############################################
ol_state = train_ol(cfg)


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys
import copy
import math
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from scipy.io import savemat

print("JAX devices:", jax.devices())


PROJECT_DIR = "/home/user/data/Hollon/海洋工程水动力/annulus_sno_annulus_only_v2/annulus_sno_annulus_only_v2"
sys.path.append(PROJECT_DIR)

from config import AnnulusConfig
from data import (
    sample_batch,
    normalize_u,
    normalize_f,
    denormalize_u,
    make_source_tokens,
    make_condition_tokens,
)
from models import FunctionEncoder
from train import (
    create_fe_state,
    create_ol_state,
    load_field_normalizer,
    rl2_error,
)


cfg = AnnulusConfig()

cfg.run_name = "test"
cfg.out_dir = "/home/user/data/Hollon/海洋工程水动力/annulus_sno_annulus_only_v2/out"

# Geometry
cfg.r_inner = 0.2
cfg.r_outer = 1.0

# Sampling and discretization
cfg.n_basis = 512
cfg.theta_size = 128
cfg.radial_size = 32
cfg.pod_snapshots = 100
cfg.random_probe_points = 1024

# PDE parameter range
cfg.k_min = 1.0
cfg.k_max = 1.0

# PI-sampler prior
cfg.sigma_list = (3.0, 5.0, 7.0)
cfg.num_repeats = 3
cfg.sample_size = 256

# Function encoder
cfg.trunk_width = 512
cfg.trunk_depth = 5

# CNN branch
cfg.branch_type = "cnn"
cfg.cnn_dense_width = 1024

# Transformer
cfg.transformer_dim = 512
cfg.transformer_heads = 8
cfg.transformer_layers = 4
cfg.transformer_mlp_dim = 1024
cfg.seq_chunks = 32
cfg.cond_chunks = 32

# Training metadata，用于初始化同结构 state
cfg.fe_steps = 300_000
cfg.ol_steps = 200_000
cfg.fe_lr = 1e-3
cfg.ol_lr = 1e-3
cfg.weight_decay = 1e-6
cfg.seed = 0

print("output_dir =", cfg.output_dir)


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Cannot find file: {path}")


def load_trained_states(config: AnnulusConfig):
    key = jax.random.PRNGKey(config.seed + 20260509)
    key_fe, key_ol = jax.random.split(key, 2)

    fe_param_path = config.output_dir / "fe_params.msgpack"
    ol_param_path = config.output_dir / "ol_params.msgpack"

    require_file(fe_param_path)
    require_file(ol_param_path)

    normalizer = load_field_normalizer(config.output_dir)

    fe_state, _ = create_fe_state(config, key_fe)
    fe_params = serialization.from_bytes(
        fe_state.params,
        fe_param_path.read_bytes(),
    )
    fe_state = fe_state.replace(params=fe_params)

    ol_state, _ = create_ol_state(config, key_ol)
    ol_params = serialization.from_bytes(
        ol_state.params,
        ol_param_path.read_bytes(),
    )
    ol_state = ol_state.replace(params=ol_params)

    return fe_state, ol_state, normalizer


fe_state, ol_state, normalizer = load_trained_states(cfg)

print("[Loaded normalizer]")
print("mean_u =", float(normalizer.mean_u))
print("std_u  =", float(normalizer.std_u))
print("mean_f =", float(normalizer.mean_f))
print("std_f  =", float(normalizer.std_f))


n_test = 128

cfg_eval = copy.deepcopy(cfg)
cfg_eval.sample_size = math.ceil(n_test / cfg_eval.num_repeats)

print("cfg_eval.sample_size =", cfg_eval.sample_size)
print("effective batch size =", cfg_eval.sample_size * cfg_eval.num_repeats)


key = jax.random.PRNGKey(cfg.seed + 999)
batch = sample_batch(key, cfg_eval)

sl = slice(0, n_test)

batch_100 = batch._replace(
    boundary_coords=batch.boundary_coords[sl],
    boundary_flux=batch.boundary_flux[sl],
    u_pod=batch.u_pod[sl],
    f_pod=batch.f_pod[sl],
    u_probe=batch.u_probe[sl],
    f_probe=batch.f_probe[sl],
    k_values=batch.k_values[sl],
)

print("u_pod shape       :", batch_100.u_pod.shape)
print("f_pod shape       :", batch_100.f_pod.shape)
print("u_probe shape     :", batch_100.u_probe.shape)
print("probe_coords shape:", batch_100.probe_coords.shape)
print("k_values shape    :", batch_100.k_values.shape)


# 1. 归一化 FE 输入
u_pod_norm = normalize_u(batch_100.u_pod, normalizer)
f_pod_norm = normalize_f(batch_100.f_pod, normalizer)

# 2. FE 编码
latent_f = fe_state.apply_fn(
    {"params": fe_state.params},
    f_pod_norm,
    method=FunctionEncoder.encode_f,
)

target_latent_u = fe_state.apply_fn(
    {"params": fe_state.params},
    u_pod_norm,
    method=FunctionEncoder.encode_u,
)

# 3. Transformer tokens
f_tokens = make_source_tokens(latent_f, cfg_eval)
cond_tokens = make_condition_tokens(batch_100, cfg_eval)

print("latent_f shape       :", latent_f.shape)
print("target_latent_u shape:", target_latent_u.shape)
print("f_tokens shape       :", f_tokens.shape)
print("cond_tokens shape    :", cond_tokens.shape)

# 4. Transformer 预测 latent_u
pred_latent_u = ol_state.apply_fn(
    {"params": ol_state.params},
    f_tokens,
    cond_tokens,
    batch_100.k_values,
)

print("pred_latent_u shape:", pred_latent_u.shape)


# SNO prediction on pod grid
u_pred_pod_norm = fe_state.apply_fn(
    {"params": fe_state.params},
    pred_latent_u,
    batch_100.pod_coords,
    method=FunctionEncoder.reconstruct,
)

u_pred_pod = denormalize_u(u_pred_pod_norm, normalizer)

# SNO prediction on random probe points
u_pred_probe_norm = fe_state.apply_fn(
    {"params": fe_state.params},
    pred_latent_u,
    batch_100.probe_coords,
    method=FunctionEncoder.reconstruct,
)

u_pred_probe = denormalize_u(u_pred_probe_norm, normalizer)

print("u_pred_pod shape  :", u_pred_pod.shape)
print("u_pred_probe shape:", u_pred_probe.shape)


u_fe_pod_norm = fe_state.apply_fn(
    {"params": fe_state.params},
    target_latent_u,
    batch_100.pod_coords,
    method=FunctionEncoder.reconstruct,
)

u_fe_probe_norm = fe_state.apply_fn(
    {"params": fe_state.params},
    target_latent_u,
    batch_100.probe_coords,
    method=FunctionEncoder.reconstruct,
)

u_fe_pod = denormalize_u(u_fe_pod_norm, normalizer)
u_fe_probe = denormalize_u(u_fe_probe_norm, normalizer)

err_sno_pod = rl2_error(u_pred_pod, batch_100.u_pod)
err_sno_probe = rl2_error(u_pred_probe, batch_100.u_probe)

err_fe_pod = rl2_error(u_fe_pod, batch_100.u_pod)
err_fe_probe = rl2_error(u_fe_probe, batch_100.u_probe)

err_latent = rl2_error(pred_latent_u, target_latent_u)

print("========== SNO zero-shot test ==========")
print("mean RL2 on pod grid     :", float(err_sno_pod.mean()))
print("mean RL2 on probe points :", float(err_sno_probe.mean()))
print("max  RL2 on probe points :", float(err_sno_probe.max()))
print("mean latent RL2          :", float(err_latent.mean()))

print("\n========== FE reconstruction diagnostic ==========")
print("mean FE RL2 on pod grid     :", float(err_fe_pod.mean()))
print("mean FE RL2 on probe points :", float(err_fe_probe.mean()))


def to_numpy(x):
    return np.asarray(jax.device_get(x))


mat_path = cfg.output_dir / "sno_100_flux_samples.mat"

mat_data = {
    # Coordinates
    "pod_coords": to_numpy(batch_100.pod_coords),              # [Npod, 2]
    "probe_coords": to_numpy(batch_100.probe_coords),          # [Nprobe, 2]
    "boundary_coords": to_numpy(batch_100.boundary_coords),    # [100, Nt, 2]
    "boundary_flux": to_numpy(batch_100.boundary_flux),        # [100, Nt]

    # Ground truth
    "u_true_pod": to_numpy(batch_100.u_pod),                   # [100, Npod]
    "f_true_pod": to_numpy(batch_100.f_pod),                   # [100, Npod]
    "u_true_probe": to_numpy(batch_100.u_probe),               # [100, Nprobe]
    "f_true_probe": to_numpy(batch_100.f_probe),               # [100, Nprobe]
    "k_values": to_numpy(batch_100.k_values[:, None]),         # [100, 1]

    # SNO prediction
    "u_pred_pod": to_numpy(u_pred_pod),                        # [100, Npod]
    "u_pred_probe": to_numpy(u_pred_probe),                    # [100, Nprobe]

    # FE reconstruction diagnostic
    "u_fe_pod": to_numpy(u_fe_pod),
    "u_fe_probe": to_numpy(u_fe_probe),

    # Latent variables
    "latent_f": to_numpy(latent_f),
    "target_latent_u": to_numpy(target_latent_u),
    "pred_latent_u": to_numpy(pred_latent_u),

    # Per-sample errors
    "err_sno_pod": to_numpy(err_sno_pod[:, None]),
    "err_sno_probe": to_numpy(err_sno_probe[:, None]),
    "err_fe_pod": to_numpy(err_fe_pod[:, None]),
    "err_fe_probe": to_numpy(err_fe_probe[:, None]),
    "err_latent": to_numpy(err_latent[:, None]),

    # Error statistics
    "mean_err_sno_pod": np.array([[float(err_sno_pod.mean())]]),
    "mean_err_sno_probe": np.array([[float(err_sno_probe.mean())]]),
    "max_err_sno_probe": np.array([[float(err_sno_probe.max())]]),
    "mean_err_fe_pod": np.array([[float(err_fe_pod.mean())]]),
    "mean_err_fe_probe": np.array([[float(err_fe_probe.mean())]]),
    "mean_err_latent": np.array([[float(err_latent.mean())]]),

    # Useful config
    "r_inner": np.array([[cfg_eval.r_inner]]),
    "r_outer": np.array([[cfg_eval.r_outer]]),
    "theta_size": np.array([[cfg_eval.theta_size]]),
    "radial_size": np.array([[cfg_eval.radial_size]]),
    "n_basis": np.array([[cfg_eval.n_basis]]),
    "random_probe_points": np.array([[cfg_eval.random_probe_points]]),
}

savemat(mat_path, mat_data, do_compression=True)

print("Saved to:", mat_path)


import matplotlib.pyplot as plt

idx = 10

coords = to_numpy(batch_100.pod_coords)
u_true = to_numpy(batch_100.u_pod[idx])
u_pred = to_numpy(u_pred_pod[idx])
abs_err = np.abs(u_pred - u_true)

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.scatter(coords[:, 0], coords[:, 1], c=u_true, s=10)
plt.axis("equal")
plt.colorbar()
plt.title("True u")

plt.subplot(1, 3, 2)
plt.scatter(coords[:, 0], coords[:, 1], c=u_pred, s=10)
plt.axis("equal")
plt.colorbar()
plt.title("SNO predicted u")

plt.subplot(1, 3, 3)
plt.scatter(coords[:, 0], coords[:, 1], c=abs_err, s=10)
plt.axis("equal")
plt.colorbar()
plt.title("|error|")

plt.tight_layout()
plt.show()

print("sample index:", idx)
print("RL2 error:", float(err_sno_probe[idx]))



##############################################
################ 预测 f=0 的解 ################
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from scipy.io import savemat

print("JAX devices:", jax.devices())


PROJECT_DIR = "/home/user/data/Hollon/海洋工程水动力/annulus_sno_annulus_only_v2/annulus_sno_annulus_only_v2"
sys.path.append(PROJECT_DIR)

from config import AnnulusConfig
from data import (
    sample_batch,
    normalize_f,
    denormalize_u,
    make_source_tokens,
    make_condition_tokens,
    inner_boundary_coords,
    make_theta,
)
from models import FunctionEncoder
from train import (
    create_fe_state,
    create_ol_state,
    load_field_normalizer,
)


cfg = AnnulusConfig()

cfg.run_name = "test"
cfg.out_dir = "/home/user/data/Hollon/海洋工程水动力/annulus_sno_annulus_only_v2/out"

# Geometry
cfg.r_inner = 0.2
cfg.r_outer = 1.0

# Sampling and discretization
cfg.n_basis = 512
cfg.theta_size = 128
cfg.radial_size = 32
cfg.pod_snapshots = 100
cfg.random_probe_points = 1024

# PDE parameter range
cfg.k_min = 1.0
cfg.k_max = 1.0

# PI-sampler prior
cfg.sigma_list = (3.0, 5.0, 7.0)

# 为了只导出一个 f=0 样本，这里用最小 batch。
# 这不会改变模型结构。
cfg.num_repeats = 1
cfg.sample_size = 1

# Function encoder
cfg.trunk_width = 512
cfg.trunk_depth = 5

# CNN branch
cfg.branch_type = "cnn"
cfg.cnn_dense_width = 1024

# Transformer
cfg.transformer_dim = 512
cfg.transformer_heads = 8
cfg.transformer_layers = 4
cfg.transformer_mlp_dim = 1024
cfg.seq_chunks = 32
cfg.cond_chunks = 32

# Training metadata，用于初始化同结构 state
cfg.fe_steps = 300_000
cfg.ol_steps = 200_000
cfg.fe_lr = 1e-3
cfg.ol_lr = 1e-3
cfg.weight_decay = 1e-6
cfg.seed = 0

print("output_dir =", cfg.output_dir)


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Cannot find file: {path}")


def load_trained_sno_states(config: AnnulusConfig):
    key = jax.random.PRNGKey(config.seed + 20260509)
    key_fe, key_ol = jax.random.split(key, 2)

    fe_param_path = config.output_dir / "fe_params_physv2.msgpack"
    ol_param_path = config.output_dir / "ol_params_physv2.msgpack"

    require_file(fe_param_path)
    require_file(ol_param_path)

    normalizer = load_field_normalizer(config.output_dir)

    fe_state, _ = create_fe_state(config, key_fe)
    fe_params = serialization.from_bytes(
        fe_state.params,
        fe_param_path.read_bytes(),
    )
    fe_state = fe_state.replace(params=fe_params)

    ol_state, _ = create_ol_state(config, key_ol)
    ol_params = serialization.from_bytes(
        ol_state.params,
        ol_param_path.read_bytes(),
    )
    ol_state = ol_state.replace(params=ol_params)

    return fe_state, ol_state, normalizer


fe_state, ol_state, normalizer = load_trained_sno_states(cfg)

print("[Loaded normalizer]")
print("mean_u =", float(normalizer.mean_u))
print("std_u  =", float(normalizer.std_u))
print("mean_f =", float(normalizer.mean_f))
print("std_f  =", float(normalizer.std_f))


import copy
import jax
import jax.numpy as jnp


def inner_boundary_flux_cos(config):
    """
    Target inner Neumann boundary condition:
        ∂u/∂n = cos(theta)

    Shape:
        [theta_size]
    """
    theta = make_theta(config)[:, 0]
    return jnp.cos(theta)


# ============================================================
# 0. 构造一个专门用于单样本推理的 config
# ============================================================
# 不建议直接改训练用 cfg；这里复制一个 cfg_pred。
cfg_pred = copy.deepcopy(cfg)

# 单样本推理，避免 sample_batch 输出 B_eff > 1
cfg_pred.num_repeats = 1
cfg_pred.sample_size = 1

# 如果当前模型是固定 k=1.0 训练的，则这样设置。
# 如果模型训练时是固定 k=0.02，则这里必须改回 0.02。
k_value = 1.0

# 审查提醒：
# k_value 必须落在训练时的 k 分布内。
# 如果训练配置是 cfg.k_min = cfg.k_max = 1.0，那么这里用 1.0 是对的。
# 如果训练配置是 cfg.k_min = cfg.k_max = 0.02，那么这里用 1.0 是明显 OOD。
print("Prediction k_value =", k_value)
print("Training config k_min/k_max =", cfg.k_min, cfg.k_max)


# ============================================================
# 1. 生成一个 batch，只使用其中的 pod_coords
# ============================================================
key = jax.random.PRNGKey(cfg_pred.seed + 12345)
batch = sample_batch(key, cfg_pred)

B = 1
Npod = cfg_pred.radial_size * cfg_pred.theta_size

print("sample_batch u_pod shape:", batch.u_pod.shape)
print("sample_batch boundary_coords shape:", batch.boundary_coords.shape)
print("sample_batch boundary_flux shape:", batch.boundary_flux.shape)


# ============================================================
# 2. 构造物理空间 f = 0
# ============================================================
f_zero_pod = jnp.zeros((B, Npod), dtype=jnp.float32)

# 关键：不能把 f_zero_pod_norm 直接设为 0。
# f_norm = 0 表示 f = mean_f，而不是 f = 0。
f_zero_pod_norm = normalize_f(f_zero_pod, normalizer)


# ============================================================
# 3. FE encode_f
# ============================================================
latent_f_zero = fe_state.apply_fn(
    {"params": fe_state.params},
    f_zero_pod_norm,
    method=FunctionEncoder.encode_f,
)
# latent_f_zero = jnp.zeros((B, cfg_pred.n_basis), dtype=jnp.float32)

# ============================================================
# 4. 手动构造目标边界条件：g(theta)=cos(theta)
# ============================================================
boundary_coords = inner_boundary_coords(cfg_pred)[None, :, :]      # [1, theta_size, 2]
boundary_flux = inner_boundary_flux_cos(cfg_pred)[None, :]         # [1, theta_size]
k_values = jnp.full((B,), k_value, dtype=jnp.float32)              # [1]

# 关键修正：
# 必须把手动构造的 boundary_coords / boundary_flux 写回 batch，
# 否则 make_condition_tokens(batch, cfg) 仍然使用 sample_batch 生成的边界。
batch_f0 = batch._replace(
    boundary_coords=boundary_coords,
    boundary_flux=boundary_flux,
    f_pod=f_zero_pod,
    k_values=k_values,
)


# ============================================================
# 5. 构造 Transformer tokens
# ============================================================
f_tokens = make_source_tokens(latent_f_zero, cfg_pred)
cond_tokens = make_condition_tokens(batch_f0, cfg_pred)

print("latent_f_zero shape:", latent_f_zero.shape)
print("f_tokens shape      :", f_tokens.shape)
print("cond_tokens shape   :", cond_tokens.shape)
print("k_values shape      :", k_values.shape)


# ============================================================
# 6. Transformer 预测 latent_u
# ============================================================
pred_latent_u = ol_state.apply_fn(
    {"params": ol_state.params},
    f_tokens,
    cond_tokens,
    k_values,
)

print("pred_latent_u shape:", pred_latent_u.shape)


# ============================================================
# 7. FE trunk 在规则 pod_coords 上重建 u
# ============================================================
u_sno_pod_norm = fe_state.apply_fn(
    {"params": fe_state.params},
    pred_latent_u,
    batch_f0.pod_coords,
    method=FunctionEncoder.reconstruct,
)


# ============================================================
# 8. 反归一化得到物理空间预测解
# ============================================================
u_sno_pod = denormalize_u(u_sno_pod_norm, normalizer)

print("pod_coords shape:", batch_f0.pod_coords.shape)
print("u_sno_pod shape :", u_sno_pod.shape)
print("k value         :", float(k_values[0]))

print("u_sno_pod min/max:", float(u_sno_pod.min()), float(u_sno_pod.max()))
print("f_zero_pod_norm min/max:", float(f_zero_pod_norm.min()), float(f_zero_pod_norm.max()))


def to_numpy(x):
    return np.asarray(jax.device_get(x))


mat_path = cfg.output_dir / "sno_f0_pod_prediction_physv2.mat"

mat_data = {
    "pod_coords": to_numpy(batch.pod_coords),          # [Npod, 2]
    "u_sno_pod": to_numpy(u_sno_pod[0, :]),            # [Npod]
    "f_zero_pod": to_numpy(f_zero_pod[0, :]),          # [Npod]
    "k_value": np.array([[k_value]], dtype=np.float64),
    "r_inner": np.array([[cfg.r_inner]], dtype=np.float64),
    "r_outer": np.array([[cfg.r_outer]], dtype=np.float64),
    "theta_size": np.array([[cfg.theta_size]], dtype=np.float64),
    "radial_size": np.array([[cfg.radial_size]], dtype=np.float64),
    "n_basis": np.array([[cfg.n_basis]], dtype=np.float64),
}

savemat(mat_path, mat_data)

print("Saved to:", mat_path)



##################################################
############## latent-only finetuning ############
##################################################
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from flax import serialization
from scipy.io import savemat
from scipy.special import iv, kv

print("JAX devices:", jax.devices())


PROJECT_DIR = "/home/user/data/Hollon/海洋工程水动力/annulus_sno_annulus_only_v2/annulus_sno_annulus_only_v2"
sys.path.append(PROJECT_DIR)

from config import AnnulusConfig
from data import (
    sample_batch,
    normalize_f,
    denormalize_f,
    denormalize_u,
    make_source_tokens,
    make_condition_tokens,
    sobol_annulus_points,
    inner_boundary_coords,
    make_theta
)
from models import FunctionEncoder
from train import (
    create_fe_state,
    create_ol_state,
    load_field_normalizer,
    rl2_error,
)


cfg = AnnulusConfig()

cfg.run_name = "test"
cfg.out_dir = "/home/user/data/Hollon/海洋工程水动力/annulus_sno_annulus_only_v2/out"

# Geometry
cfg.r_inner = 0.2
cfg.r_outer = 1.0

# Sampling and discretization
cfg.n_basis = 512
cfg.theta_size = 128
cfg.radial_size = 32
cfg.pod_snapshots = 100
cfg.random_probe_points = 1024

# PDE parameter
cfg.k_min = 1.0
cfg.k_max = 1.0

# PI-sampler prior
cfg.sigma_list = (3.0, 5.0, 7.0)

# 为了做单样本 f=0 微调，这里用 batch size 1。
# 不影响已训练模型结构。
cfg.num_repeats = 1
cfg.sample_size = 1

# Function encoder
cfg.trunk_width = 512
cfg.trunk_depth = 5

# CNN branch
cfg.branch_type = "cnn"
cfg.cnn_dense_width = 1024

# Transformer
cfg.transformer_dim = 512
cfg.transformer_heads = 8
cfg.transformer_layers = 4
cfg.transformer_mlp_dim = 1024
cfg.seq_chunks = 32
cfg.cond_chunks = 32

# Training metadata，仅用于初始化同结构 state
cfg.fe_steps = 300_000
cfg.ol_steps = 200_000
cfg.fe_lr = 1e-3
cfg.ol_lr = 1e-3
cfg.weight_decay = 1e-6
cfg.seed = 0

print("output_dir =", cfg.output_dir)


def require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Cannot find file: {path}")


def load_trained_sno(config: AnnulusConfig):
    key = jax.random.PRNGKey(config.seed + 20260512)
    key_fe, key_ol = jax.random.split(key, 2)

    fe_param_path = config.output_dir / "fe_params_physv2.msgpack"
    ol_param_path = config.output_dir / "ol_params_physv2.msgpack"

    require_file(fe_param_path)
    require_file(ol_param_path)

    normalizer = load_field_normalizer(config.output_dir)

    fe_state, fe_model = create_fe_state(config, key_fe)
    fe_params = serialization.from_bytes(
        fe_state.params,
        fe_param_path.read_bytes(),
    )
    fe_state = fe_state.replace(params=fe_params)

    ol_state, ol_model = create_ol_state(config, key_ol)
    ol_params = serialization.from_bytes(
        ol_state.params,
        ol_param_path.read_bytes(),
    )
    ol_state = ol_state.replace(params=ol_params)

    return fe_state, fe_model, ol_state, ol_model, normalizer


fe_state, fe_model, ol_state, ol_model, normalizer = load_trained_sno(cfg)

print("[Loaded normalizer]")
print("mean_u =", float(normalizer.mean_u))
print("std_u  =", float(normalizer.std_u))
print("mean_f =", float(normalizer.mean_f))
print("std_f  =", float(normalizer.std_f))


def inner_boundary_flux_cos(config):
    """
    Target inner Neumann boundary condition:
        ∂u/∂n = cos(theta)

    Shape:
        [theta_size]
    """
    theta = make_theta(config)[:, 0]
    return jnp.cos(theta)

key = jax.random.PRNGKey(cfg.seed + 12345)

# 只用 batch 中的 pod_coords 和 boundary 信息
batch = sample_batch(key, cfg)

B = 1
Npod = cfg.radial_size * cfg.theta_size
k_value = 1.0

# ------------------------------------------------------------
# 1. 物理空间 f = 0
# ------------------------------------------------------------
f_zero_pod = jnp.zeros((B, Npod), dtype=jnp.float32)

# ------------------------------------------------------------
# 2. 必须归一化后再送入 FE.encode_f
# ------------------------------------------------------------
f_zero_pod_norm = normalize_f(f_zero_pod, normalizer)

# ------------------------------------------------------------
# 3. FE encode_f
# ------------------------------------------------------------
latent_f_zero = fe_state.apply_fn(
    {"params": fe_state.params},
    f_zero_pod_norm,
    method=FunctionEncoder.encode_f,
)

# ============================================================
# 4. 手动构造目标边界条件：g(theta)=cos(theta)
# ============================================================
boundary_coords = inner_boundary_coords(cfg)[None, :, :]      # [1, theta_size, 2]
boundary_flux = inner_boundary_flux_cos(cfg)[None, :]         # [1, theta_size]
k_values = jnp.full((B,), k_value, dtype=jnp.float32)              # [1]

# 关键修正：
# 必须把手动构造的 boundary_coords / boundary_flux 写回 batch，
# 否则 make_condition_tokens(batch, cfg) 仍然使用 sample_batch 生成的边界。
batch_f0 = batch._replace(
    boundary_coords=boundary_coords,
    boundary_flux=boundary_flux,
    f_pod=f_zero_pod,
    k_values=k_values,
)

# ------------------------------------------------------------
# 4. Transformer tokens
# ------------------------------------------------------------
f_tokens = make_source_tokens(latent_f_zero, cfg)
cond_tokens = make_condition_tokens(batch_f0, cfg)
k_values = jnp.full((B,), k_value, dtype=jnp.float32)

# ------------------------------------------------------------
# 5. zero-shot latent prediction
# ------------------------------------------------------------
pred_latent_u_0 = ol_state.apply_fn(
    {"params": ol_state.params},
    f_tokens,
    cond_tokens,
    k_values,
)

z0 = pred_latent_u_0[0]   # [n_basis]

print("latent_f_zero shape:", latent_f_zero.shape)
print("pred_latent_u_0 shape:", pred_latent_u_0.shape)
print("z0 shape:", z0.shape)


import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from scipy.io import savemat


# ============================================================
# 1. FE trunk basis and derivatives
# ============================================================

def trunk_apply(coords):
    """
    coords: [N, 2]
    return:
        basis: [N, n_basis]
    """
    return fe_model.apply(
        {"params": fe_state.params},
        coords,
        method=lambda module, x: module.trunk(x),
    )


def precompute_basis_operators(coords):
    """
    Precompute basis, gradient basis, and Laplacian basis.

    coords:
        [N, 2]

    returns:
        basis:      [N, p]
        grad_basis: [N, p, 2]
        lap_basis:  [N, p]
    """
    coords = jnp.asarray(coords, dtype=jnp.float32)

    def trunk_single(x):
        return trunk_apply(x[None, :])[0]  # [p]

    basis = trunk_apply(coords)  # [N, p]

    grad_basis = jax.vmap(jax.jacfwd(trunk_single))(coords)  # [N, p, 2]

    hess_basis = jax.vmap(
        jax.jacfwd(jax.jacfwd(trunk_single))
    )(coords)  # [N, p, 2, 2]

    lap_basis = jnp.trace(
        hess_basis,
        axis1=-2,
        axis2=-1,
    )  # [N, p]

    return basis, grad_basis, lap_basis


# ============================================================
# 2. Evaluate physical u, grad u, lap u
# ============================================================

def eval_u_from_basis(z, basis, normalizer, config):
    """
    z:     [p]
    basis: [N, p]
    return physical u: [N]
    """
    u_norm = jnp.einsum("np,p->n", basis, z) / jnp.sqrt(config.n_basis)
    u = u_norm * normalizer.std_u + normalizer.mean_u
    return u


def eval_grad_u_from_basis(z, grad_basis, normalizer, config):
    """
    z:          [p]
    grad_basis: [N, p, 2]
    return physical grad u: [N, 2]
    """
    grad_u_norm = jnp.einsum("npd,p->nd", grad_basis, z) / jnp.sqrt(config.n_basis)
    grad_u = normalizer.std_u * grad_u_norm
    return grad_u


def eval_lap_u_from_basis(z, lap_basis, normalizer, config):
    """
    z:         [p]
    lap_basis: [N, p]
    return physical Laplacian u: [N]
    """
    lap_u_norm = jnp.einsum("np,p->n", lap_basis, z) / jnp.sqrt(config.n_basis)
    lap_u = normalizer.std_u * lap_u_norm
    return lap_u


def rel_l2(pred, ref, eps=1e-12):
    return jnp.linalg.norm(pred - ref) / jnp.clip(jnp.linalg.norm(ref), a_min=eps)


from scipy.stats import qmc
import jax
import jax.numpy as jnp


def sample_annulus_points_sobol_jaxkey(key, n_points, r_inner, r_outer, scramble=True):
    """
    Sobol low-discrepancy sampling in an annulus.

    Area-uniform sampling:
        r^2 ~ Uniform(r_inner^2, r_outer^2)
        theta ~ Uniform(0, 2*pi)

    Args:
        key: JAX PRNGKey, used only to generate scipy Sobol scramble seed.
        n_points: number of points.
        r_inner: inner radius.
        r_outer: outer radius.
        scramble: whether to scramble Sobol sequence.

    Return:
        coords: [n_points, 2], jnp.float32
    """
    # scipy Sobol uses a Python int seed.
    seed = int(jax.random.randint(key, (), 0, 2**31 - 1))

    sampler = qmc.Sobol(
        d=2,
        scramble=scramble,
        seed=seed,
    )

    u = sampler.random(n_points)  # [n_points, 2]

    u = jnp.asarray(u, dtype=jnp.float32)

    u1 = u[:, 0]
    u2 = u[:, 1]

    r = r_inner + u1 * (r_outer - r_inner)

    theta = 2.0 * jnp.pi * u2

    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)

    return jnp.stack([x, y], axis=-1)


def sample_inner_boundary_candidates(key, n_points, config):
    """
    Inner boundary:
        r = r_inner

    Fluid-domain outward normal:
        n = -e_r

    Target:
        ∂u/∂n = cos(theta)
    """
    theta = jax.random.uniform(
        key,
        shape=(n_points,),
        minval=0.0,
        maxval=2.0 * jnp.pi,
    )

    coords = jnp.stack(
        [
            config.r_inner * jnp.cos(theta),
            config.r_inner * jnp.sin(theta),
        ],
        axis=-1,
    )

    normal = -jnp.stack(
        [
            jnp.cos(theta),
            jnp.sin(theta),
        ],
        axis=-1,
    )

    flux = jnp.cos(theta)

    return coords, normal, flux


def sample_outer_boundary_candidates(key, n_points, config):
    """
    Outer boundary:
        r = r_outer

    Target:
        u = 0
    """
    theta = jax.random.uniform(
        key,
        shape=(n_points,),
        minval=0.0,
        maxval=2.0 * jnp.pi,
    )

    coords = jnp.stack(
        [
            config.r_outer * jnp.cos(theta),
            config.r_outer * jnp.sin(theta),
        ],
        axis=-1,
    )

    return coords


def finetune_loss_components_adaptive(
    z,
    z_ref,
    k_value,
    f_col,
    basis_col,
    grad_basis_col,
    lap_basis_col,
    basis_inner,
    grad_basis_inner,
    basis_outer,
    n_inner,
    g_inner,
    normalizer,
    config,
    eps=1.0e-8,
):
    # ------------------------------------------------------------
    # PDE residual:
    #     Δu - k^2 u - f = 0
    # ------------------------------------------------------------
    u_col = eval_u_from_basis(z, basis_col, normalizer, config)
    lap_u_col = eval_lap_u_from_basis(z, lap_basis_col, normalizer, config)

    eq_res = lap_u_col - (k_value ** 2) * u_col - f_col

    # 归一化尺度：对于 f=0，不能只除以 mean(f^2)
    eq_scale = (
        jnp.mean(lap_u_col ** 2)
        + jnp.mean((k_value ** 2 * u_col) ** 2)
        + jnp.mean(f_col ** 2)
        + eps
    )
    eq_loss = jnp.mean(eq_res ** 2) / eq_scale

    # ------------------------------------------------------------
    # Inner Neumann:
    #     ∂u/∂n = cos(theta)
    # ------------------------------------------------------------
    grad_u_inner = eval_grad_u_from_basis(
        z,
        grad_basis_inner,
        normalizer,
        config,
    )
    normal_du_inner = jnp.sum(grad_u_inner * n_inner, axis=-1)

    inner_res = normal_du_inner - g_inner
    inner_scale = jnp.mean(g_inner ** 2) + eps
    inner_loss = jnp.mean(inner_res ** 2) / inner_scale

    # ------------------------------------------------------------
    # Outer Dirichlet:
    #     u = 0
    # ------------------------------------------------------------
    u_outer = eval_u_from_basis(
        z,
        basis_outer,
        normalizer,
        config,
    )
    outer_scale = jnp.mean(u_col ** 2) + eps
    outer_loss = jnp.mean(u_outer ** 2) / outer_scale

    # ------------------------------------------------------------
    # Latent regularization
    # ------------------------------------------------------------
    reg_loss = jnp.mean((z - z_ref) ** 2)

    return eq_loss, inner_loss, outer_loss, reg_loss


def finetune_total_loss_adaptive(
    z,
    z_ref,
    k_value,
    train_batch,
    normalizer,
    config,
    w_eq=1.0,
    w_inner=1.0,
    w_outer=1.0,
    w_reg=1.0e-6,
):
    eq_loss, inner_loss, outer_loss, reg_loss = finetune_loss_components_adaptive(
        z=z,
        z_ref=z_ref,
        k_value=k_value,
        f_col=train_batch["f_col"],
        basis_col=train_batch["basis_col"],
        grad_basis_col=train_batch["grad_basis_col"],
        lap_basis_col=train_batch["lap_basis_col"],
        basis_inner=train_batch["basis_inner"],
        grad_basis_inner=train_batch["grad_basis_inner"],
        basis_outer=train_batch["basis_outer"],
        n_inner=train_batch["n_inner"],
        g_inner=train_batch["g_inner"],
        normalizer=normalizer,
        config=config,
    )

    total = (
        w_eq * eq_loss
        + w_inner * inner_loss
        + w_outer * outer_loss
        + w_reg * reg_loss
    )

    return total, (eq_loss, inner_loss, outer_loss, reg_loss)


def _normalize_score(score, eps=1.0e-8):
    return score / (jnp.mean(score) + eps)


def select_top_and_random(key, score, n_select, hard_fraction=0.75):
    """
    Select hard points according to score and fill the rest randomly.

    score:
        [N_candidate]
    """
    n_candidate = score.shape[0]
    n_hard = int(n_select * hard_fraction)
    n_rand = n_select - n_hard

    n_hard = max(1, min(n_hard, n_candidate))
    n_rand = max(0, min(n_rand, n_candidate))

    _, idx_hard = jax.lax.top_k(score, n_hard)

    if n_rand > 0:
        idx_rand = jax.random.choice(
            key,
            n_candidate,
            shape=(n_rand,),
            replace=False,
        )
        idx = jnp.concatenate([idx_hard, idx_rand], axis=0)
    else:
        idx = idx_hard

    return idx


def build_gradient_adaptive_training_batch(
    key,
    z_current,
    k_value,
    normalizer,
    config,
    n_col=1024,
    n_inner=256,
    n_outer=256,
    n_cand_col=4096,
    n_cand_inner=1024,
    n_cand_outer=1024,
    hard_fraction=0.75,
    grad_weight=0.25,
):
    """
    Build one adaptive physics-finetuning batch.

    Interior score:
        S = normalized(|PDE residual|) + grad_weight * normalized(||grad u||)

    Inner boundary score:
        S = |∂u/∂n - cos(theta)|

    Outer boundary score:
        S = |u|

    Returns:
        train_batch dict containing selected basis operators.
    """
    key_col, key_inner, key_outer, key_sel_col, key_sel_inner, key_sel_outer = jax.random.split(key, 6)

    # ============================================================
    # 1. Interior candidates
    # ============================================================
    x_col_cand = sample_annulus_points_sobol_jaxkey(
        key_col,
        n_cand_col,
        config.r_inner,
        config.r_outer,
    )

    basis_col_cand, grad_basis_col_cand, lap_basis_col_cand = precompute_basis_operators(
        x_col_cand
    )

    u_col_cand = eval_u_from_basis(
        z_current,
        basis_col_cand,
        normalizer,
        config,
    )
    grad_u_col_cand = eval_grad_u_from_basis(
        z_current,
        grad_basis_col_cand,
        normalizer,
        config,
    )
    lap_u_col_cand = eval_lap_u_from_basis(
        z_current,
        lap_basis_col_cand,
        normalizer,
        config,
    )

    f_col_cand = jnp.zeros((n_cand_col,), dtype=jnp.float32)

    eq_res_cand = lap_u_col_cand - (k_value ** 2) * u_col_cand - f_col_cand
    grad_norm_cand = jnp.linalg.norm(grad_u_col_cand, axis=-1)

    score_col = (
        _normalize_score(jnp.abs(eq_res_cand))
        + grad_weight * _normalize_score(grad_norm_cand)
    )

    # idx_col = select_top_and_random(
    #     key_sel_col,
    #     score_col,
    #     n_col,
    #     hard_fraction=hard_fraction,
    # )

    x_col = x_col_cand #[idx_col]
    basis_col = basis_col_cand #[idx_col]
    grad_basis_col = grad_basis_col_cand #[idx_col]
    lap_basis_col = lap_basis_col_cand #[idx_col]
    f_col = jnp.zeros((n_cand_col,), dtype=jnp.float32) #jnp.zeros((idx_col.shape[0],), dtype=jnp.float32)

    # ============================================================
    # 2. Inner boundary candidates
    # ============================================================
    x_inner_cand, n_inner_cand, g_inner_cand = sample_inner_boundary_candidates(
        key_inner,
        n_cand_inner,
        config,
    )

    basis_inner_cand, grad_basis_inner_cand, _ = precompute_basis_operators(
        x_inner_cand
    )

    grad_u_inner_cand = eval_grad_u_from_basis(
        z_current,
        grad_basis_inner_cand,
        normalizer,
        config,
    )
    normal_du_inner_cand = jnp.sum(grad_u_inner_cand * n_inner_cand, axis=-1)

    inner_res_cand = normal_du_inner_cand - g_inner_cand
    score_inner = jnp.abs(inner_res_cand)

    # idx_inner = select_top_and_random(
    #     key_sel_inner,
    #     score_inner,
    #     n_inner,
    #     hard_fraction=hard_fraction,
    # )

    x_inner = x_inner_cand #[idx_inner]
    n_inner_sel = n_inner_cand #[idx_inner]
    g_inner_sel = g_inner_cand #[idx_inner]
    basis_inner = basis_inner_cand #[idx_inner]
    grad_basis_inner = grad_basis_inner_cand #[idx_inner]

    # ============================================================
    # 3. Outer boundary candidates
    # ============================================================
    x_outer_cand = sample_outer_boundary_candidates(
        key_outer,
        n_cand_outer,
        config,
    )

    basis_outer_cand, _, _ = precompute_basis_operators(
        x_outer_cand
    )

    u_outer_cand = eval_u_from_basis(
        z_current,
        basis_outer_cand,
        normalizer,
        config,
    )

    score_outer = jnp.abs(u_outer_cand)

    # idx_outer = select_top_and_random(
    #     key_sel_outer,
    #     score_outer,
    #     n_outer,
    #     hard_fraction=hard_fraction,
    # )

    x_outer = x_outer_cand #[idx_outer]
    basis_outer = basis_outer_cand #[idx_outer]

    return {
        "x_col": x_col,
        "f_col": f_col,
        "basis_col": basis_col,
        "grad_basis_col": grad_basis_col,
        "lap_basis_col": lap_basis_col,
        "x_inner": x_inner,
        "basis_inner": basis_inner,
        "grad_basis_inner": grad_basis_inner,
        "n_inner": n_inner_sel,
        "g_inner": g_inner_sel,
        "x_outer": x_outer,
        "basis_outer": basis_outer,
        "mean_score_col": jnp.mean(score_col),
        "max_score_col": jnp.max(score_col),
        "mean_score_inner": jnp.mean(score_inner),
        "max_score_inner": jnp.max(score_inner),
        "mean_score_outer": jnp.mean(score_outer),
        "max_score_outer": jnp.max(score_outer),
    }


lr = 1.0e-3
optimizer = optax.adam(lr)


@jax.jit
def gradient_adaptive_finetune_step(
    z,
    opt_state,
    z_ref,
    k_value,
    train_batch,
    normalizer,
):
    def loss_fn(zz):
        total, aux = finetune_total_loss_adaptive(
            z=zz,
            z_ref=z_ref,
            k_value=k_value,
            train_batch=train_batch,
            normalizer=normalizer,
            config=cfg,
            w_eq=1.0,
            w_inner=1.0,
            w_outer=1.0,
            w_reg=0.0,
        )
        return total, aux

    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(z)
    updates, opt_state = optimizer.update(grads, opt_state, z)
    z = optax.apply_updates(z, updates)

    grad_norm_z = jnp.linalg.norm(grads)

    return z, opt_state, loss, aux, grad_norm_z


def exact_f0_solution_np(coords_np, k, a, R):
    """
    Analytical solution:
        u(r,theta) = [A I1(kr) + B K1(kr)] cos(theta)
    with:
        u(R,theta)=0
        ∂r u(a,theta)=-cos(theta)
    """
    x = coords_np[:, 0]
    y = coords_np[:, 1]

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    ka = k * a
    kR = k * R

    I1_R = iv(1, kR)
    K1_R = kv(1, kR)

    I1p_a = 0.5 * (iv(0, ka) + iv(2, ka))
    K1p_a = -0.5 * (kv(0, ka) + kv(2, ka))

    M = np.array(
        [
            [I1_R, K1_R],
            [k * I1p_a, k * K1p_a],
        ],
        dtype=np.float64,
    )

    rhs = np.array([0.0, -1.0], dtype=np.float64)
    A, B = np.linalg.solve(M, rhs)

    u = (A * iv(1, k * r) + B * kv(1, k * r)) * np.cos(theta)
    return u.astype(np.float32), A, B

# k_value = 5.0
pod_coords_np = np.asarray(jax.device_get(batch.pod_coords))
u_exact_pod_np, A_exact, B_exact = exact_f0_solution_np(
    pod_coords_np,
    k_value,
    cfg.r_inner,
    cfg.r_outer,
)

u_exact_pod = jnp.asarray(u_exact_pod_np)

print("A_exact =", A_exact)
print("B_exact =", B_exact)
print("u_exact_pod shape:", u_exact_pod.shape)


# pod grid evaluation basis
basis_pod, grad_basis_pod, lap_basis_pod = precompute_basis_operators(
    batch_f0.pod_coords
)

# zero-shot evaluation
u_zero_pod = eval_u_from_basis(
    z0,
    basis_pod,
    normalizer,
    cfg,
)

print("u_zero_pod shape:", u_zero_pod.shape)
print("u_zero_pod min/max:", float(u_zero_pod.min()), float(u_zero_pod.max()))

# 如果你已经有理论解 u_exact_pod，则开启这段
try:
    u_exact_pod_vec = u_exact_pod
    if u_exact_pod_vec.ndim == 2:
        u_exact_pod_vec = u_exact_pod_vec[0]

    err_zero = rel_l2(u_zero_pod, u_exact_pod_vec)
    print("Zero-shot RL2 vs exact:", float(err_zero))
except NameError:
    u_exact_pod_vec = None
    err_zero = jnp.nan
    print("u_exact_pod not found; only physics losses will be monitored.")


# ============================================================
# Initial latent
# ============================================================
z = z0.copy()
z_ref = z0.copy()

opt_state = optimizer.init(z)

# ============================================================
# Dynamic sampling parameters
# ============================================================
n_steps = 1000

resample_every = 1

n_col = 1024
n_inner = 256
n_outer = 256

n_cand_col = 1024
n_cand_inner = 256
n_cand_outer = 256

hard_fraction = 0.0
grad_weight = 0.25

key_adapt = jax.random.PRNGKey(cfg.seed + 77777)

history = []
train_batch = None

for step in range(n_steps + 1):

    # ------------------------------------------------------------
    # Rebuild adaptive training points every K steps
    # ------------------------------------------------------------
    if step % resample_every == 0:
        key_adapt, key_sample = jax.random.split(key_adapt)

        train_batch = build_gradient_adaptive_training_batch(
            key=key_sample,
            z_current=z,
            k_value=k_value,
            normalizer=normalizer,
            config=cfg,
            n_col=n_col,
            n_inner=n_inner,
            n_outer=n_outer,
            n_cand_col=n_cand_col,
            n_cand_inner=n_cand_inner,
            n_cand_outer=n_cand_outer,
            hard_fraction=hard_fraction,
            grad_weight=grad_weight,
        )

    # ------------------------------------------------------------
    # One optimization step on selected points
    # ------------------------------------------------------------
    z, opt_state, loss, aux, grad_norm_z = gradient_adaptive_finetune_step(
        z=z,
        opt_state=opt_state,
        z_ref=z_ref,
        k_value=k_value,
        train_batch=train_batch,
        normalizer=normalizer,
    )

    # ------------------------------------------------------------
    # Monitor
    # ------------------------------------------------------------
    if step % 100 == 0:
        eq_loss, inner_loss, outer_loss, reg_loss = aux

        u_ft_pod = eval_u_from_basis(
            z,
            basis_pod,
            normalizer,
            cfg,
        )

        if u_exact_pod_vec is not None:
            err_ft = rel_l2(u_ft_pod, u_exact_pod_vec)
        else:
            err_ft = jnp.nan

        history.append(
            [
                step,
                float(loss),
                float(eq_loss),
                float(inner_loss),
                float(outer_loss),
                float(reg_loss),
                float(grad_norm_z),
                float(err_ft),
                float(train_batch["mean_score_col"]),
                float(train_batch["max_score_col"]),
                float(train_batch["mean_score_inner"]),
                float(train_batch["max_score_inner"]),
                float(train_batch["mean_score_outer"]),
                float(train_batch["max_score_outer"]),
            ]
        )

        print(
            f"step={step:05d} "
            f"loss={float(loss):.4e} "
            f"eq={float(eq_loss):.4e} "
            f"inner={float(inner_loss):.4e} "
            f"outer={float(outer_loss):.4e} "
            f"reg={float(reg_loss):.2e} "
            f"|grad_z|={float(grad_norm_z):.3e} "
            f"rl2={float(err_ft):.4e} "
            f"score_col_max={float(train_batch['max_score_col']):.3e}"
        )

history = np.asarray(history)



plt.figure(figsize=(8, 5))

plt.semilogy(history[:, 0], history[:, 1], label="total")
plt.semilogy(history[:, 0], history[:, 2], label="PDE")
plt.semilogy(history[:, 0], history[:, 3], label="inner BC")
plt.semilogy(history[:, 0], history[:, 4], label="outer BC")

if not np.all(np.isnan(history[:, 7])):
    plt.semilogy(history[:, 0], history[:, 7], label="RL2 vs exact")

plt.xlabel("step")
plt.ylabel("value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


u_ft_pod = eval_u_from_basis(
    z,
    basis_pod,
    normalizer,
    cfg,
)

coords_np = np.asarray(jax.device_get(batch_f0.pod_coords))
x = coords_np[:, 0]
y = coords_np[:, 1]

u_ft_np = np.asarray(jax.device_get(u_ft_pod))
u_zero_np = np.asarray(jax.device_get(u_zero_pod))

plt.figure(figsize=(15, 4))

if u_exact_pod_vec is not None:
    u_exact_np = np.asarray(jax.device_get(u_exact_pod_vec))
    abs_err_ft = np.abs(u_ft_np - u_exact_np)

    cmin = min(u_ft_np.min(), u_exact_np.min())
    cmax = max(u_ft_np.max(), u_exact_np.max())

    plt.subplot(1, 3, 1)
    plt.scatter(x, y, c=u_ft_np, s=10, cmap="turbo", vmin=cmin, vmax=cmax)
    plt.axis("equal")
    plt.colorbar()
    plt.title("Adaptive-finetuned SNO")

    plt.subplot(1, 3, 2)
    plt.scatter(x, y, c=u_exact_np, s=10, cmap="turbo", vmin=cmin, vmax=cmax)
    plt.axis("equal")
    plt.colorbar()
    plt.title("Exact")

    plt.subplot(1, 3, 3)
    plt.scatter(x, y, c=abs_err_ft, s=10, cmap="hot")
    plt.axis("equal")
    plt.colorbar()
    plt.title(f"|error|, RL2={float(rel_l2(u_ft_pod, u_exact_pod_vec)):.3e}")

else:
    plt.subplot(1, 2, 1)
    plt.scatter(x, y, c=u_zero_np, s=10, cmap="turbo")
    plt.axis("equal")
    plt.colorbar()
    plt.title("Zero-shot")

    plt.subplot(1, 2, 2)
    plt.scatter(x, y, c=u_ft_np, s=10, cmap="turbo")
    plt.axis("equal")
    plt.colorbar()
    plt.title("Adaptive-finetuned")

plt.tight_layout()
plt.show()



###########################################
####### 检查理论解能否正确表示 ##############
###########################################
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.special import iv, kv
from scipy.io import savemat

from data import normalize_u, denormalize_u
from models import FunctionEncoder


def exact_f0_solution_np(coords_np, k, a, R):
    """
    Analytical solution for:
        Δu - k^2 u = 0
        u(R,theta) = 0
        ∂_r u(a,theta) = -cos(theta)

    Solution:
        u(r,theta) = [A I_1(k r) + B K_1(k r)] cos(theta)
    """
    x = coords_np[:, 0]
    y = coords_np[:, 1]

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    ka = k * a
    kR = k * R

    I1_R = iv(1, kR)
    K1_R = kv(1, kR)

    I1p_a = 0.5 * (iv(0, ka) + iv(2, ka))
    K1p_a = -0.5 * (kv(0, ka) + kv(2, ka))

    M = np.array(
        [
            [I1_R, K1_R],
            [k * I1p_a, k * K1p_a],
        ],
        dtype=np.float64,
    )

    rhs = np.array([0.0, -1.0], dtype=np.float64)
    A, B = np.linalg.solve(M, rhs)

    u = (A * iv(1, k * r) + B * kv(1, k * r)) * np.cos(theta)

    return u.astype(np.float32), A, B


k_value = 1.0

pod_coords = batch.pod_coords
probe_coords = batch.probe_coords
pod_coords_np = np.asarray(jax.device_get(pod_coords))

u_exact_pod_np, A_exact, B_exact = exact_f0_solution_np(
    coords_np=pod_coords,
    k=k_value,
    a=cfg.r_inner,
    R=cfg.r_outer,
)

u_exact_probe_np, A_exact, B_exact = exact_f0_solution_np(
    coords_np=probe_coords,
    k=k_value,
    a=cfg.r_inner,
    R=cfg.r_outer,
)

u_exact_pod = jnp.asarray(u_exact_pod_np)[None, :]  # [1, Npod]
u_exact_probe = jnp.asarray(u_exact_probe_np)[None, :]

print("u_exact_pod shape:", u_exact_pod.shape)
print("A_exact =", A_exact)
print("B_exact =", B_exact)
print("u_exact min/max:", float(u_exact_pod.min()), float(u_exact_pod.max()))


# 1. 归一化理论解
u_exact_pod_norm = normalize_u(u_exact_pod, normalizer)

# 2. 输入 u-branch 得到理论解对应的 FE latent
latent_u_exact = fe_state.apply_fn(
    {"params": fe_state.params},
    u_exact_pod_norm,
    method=FunctionEncoder.encode_u,
)

# 3. 用 FE trunk 在同一组 pod_coords 上重构
u_fe_rec_pod_norm = fe_state.apply_fn(
    {"params": fe_state.params},
    latent_u_exact,
    pod_coords,
    method=FunctionEncoder.reconstruct,
)

# 4. 反归一化回物理空间
u_fe_rec_pod = denormalize_u(u_fe_rec_pod_norm, normalizer)

print("latent_u_exact shape:", latent_u_exact.shape)
print("u_fe_rec_pod shape:", u_fe_rec_pod.shape)
print("u_fe_rec min/max:", float(u_fe_rec_pod.min()), float(u_fe_rec_pod.max()))


def rel_l2(pred, ref):
    return jnp.linalg.norm(pred - ref, axis=-1) / jnp.clip(
        jnp.linalg.norm(ref, axis=-1),
        a_min=1e-12,
    )


err_fe_exact_pod = rel_l2(u_fe_rec_pod, u_exact_pod)

abs_err_fe_exact_pod = jnp.abs(u_fe_rec_pod - u_exact_pod)

print("FE reconstruction relative L2 error on exact f=0 solution:")
print(float(err_fe_exact_pod[0]))

print("Max abs error:", float(abs_err_fe_exact_pod.max()))
print("Mean abs error:", float(abs_err_fe_exact_pod.mean()))


# probe_coords_np = np.asarray(jax.device_get(probe_coords))
x = pod_coords_np[:, 0]
y = pod_coords_np[:, 1]

u_exact_np = np.asarray(jax.device_get(u_exact_pod[0]))
u_rec_np = np.asarray(jax.device_get(u_fe_rec_pod[0]))
abs_err_np = np.abs(u_rec_np - u_exact_np)

cmin = min(u_exact_np.min(), u_rec_np.min())
cmax = max(u_exact_np.max(), u_rec_np.max())

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.scatter(x, y, c=u_exact_np, s=10, cmap="turbo", vmin=cmin, vmax=cmax)
plt.axis("equal")
plt.colorbar()
plt.title(r"$u_{\mathrm{exact}}$")

plt.subplot(1, 3, 2)
plt.scatter(x, y, c=u_rec_np, s=10, cmap="turbo", vmin=cmin, vmax=cmax)
plt.axis("equal")
plt.colorbar()
plt.title(r"$u_{\mathrm{FE\ rec}}$")

plt.subplot(1, 3, 3)
plt.scatter(x, y, c=abs_err_np, s=10, cmap="hot")
plt.axis("equal")
plt.colorbar()
plt.title(rf"$|u_{{FE}}-u_{{exact}}|$, RL2={float(err_fe_exact_pod[0]):.3e}")

plt.tight_layout()
plt.show()



################################################
## 检查在FE的基函数下，是否存在一组最优系数表达解 ##
################################################
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.special import iv, kv
from scipy.io import savemat

from data import make_annulus_grid, sobol_annulus_points, normalize_u, denormalize_u
from models import FunctionEncoder


def exact_f0_solution_np(coords_np, k, a, R):
    """
    Analytical solution for:
        Δu - k^2 u = 0
        u(R,theta) = 0
        ∂_n u(a,theta) = cos(theta)

    Since inner outward normal n = -e_r:
        ∂_r u(a,theta) = -cos(theta)

    Solution:
        u(r,theta) = [A I_1(k r) + B K_1(k r)] cos(theta)
    """
    x = coords_np[:, 0]
    y = coords_np[:, 1]

    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    ka = k * a
    kR = k * R

    I1_R = iv(1, kR)
    K1_R = kv(1, kR)

    I1p_a = 0.5 * (iv(0, ka) + iv(2, ka))
    K1p_a = -0.5 * (kv(0, ka) + kv(2, ka))

    M = np.array(
        [
            [I1_R, K1_R],
            [k * I1p_a, k * K1p_a],
        ],
        dtype=np.float64,
    )

    rhs = np.array([0.0, -1.0], dtype=np.float64)

    A, B = np.linalg.solve(M, rhs)

    u = (A * iv(1, k * r) + B * kv(1, k * r)) * np.cos(theta)

    return u.astype(np.float32), A, B


k_value = 1.0   # 根据你当前训练/测试设定修改

pod_coords = make_annulus_grid(cfg)
pod_coords_np = np.asarray(jax.device_get(pod_coords))

u_exact_pod_np, A_exact, B_exact = exact_f0_solution_np(
    coords_np=pod_coords_np,
    k=k_value,
    a=cfg.r_inner,
    R=cfg.r_outer,
)

u_exact_pod = jnp.asarray(u_exact_pod_np)[None, :]  # [1, Npod]

print("pod_coords shape:", pod_coords.shape)
print("u_exact_pod shape:", u_exact_pod.shape)
print("A_exact =", A_exact)
print("B_exact =", B_exact)
print("u_exact min/max:", float(u_exact_pod.min()), float(u_exact_pod.max()))


def trunk_apply(coords):
    """
    coords: [N, 2]
    return:
        basis: [N, n_basis]
    """
    return fe_model.apply(
        {"params": fe_state.params},
        coords,
        method=lambda module, x: module.trunk(x),
    )


basis_pod = trunk_apply(pod_coords)  # [Npod, n_basis]

print("basis_pod shape:", basis_pod.shape)


def solve_latent_ridge_lstsq(
    basis,
    u_target_phys,
    normalizer,
    config,
    ridge=1e-8,
):
    """
    Solve:
        min_z || A z - u_target_norm ||^2 + ridge ||z||^2

    where:
        A = basis / sqrt(n_basis)

    Inputs:
        basis: [N, p]
        u_target_phys: [N] or [1, N]

    Return:
        z_star: [p]
        u_rec_phys: [N]
        u_rec_norm: [N]
    """
    if u_target_phys.ndim == 2:
        u_target_phys = u_target_phys[0]

    # Normalize target using the FE normalizer
    u_target_norm = (u_target_phys - normalizer.mean_u) / normalizer.std_u

    A = basis / jnp.sqrt(config.n_basis)  # [N, p]
    y = u_target_norm                     # [N]

    p = A.shape[1]

    # Ridge normal equation:
    # (A^T A + ridge I) z = A^T y
    ATA = A.T @ A
    ATy = A.T @ y

    z_star = jnp.linalg.solve(
        ATA + ridge * jnp.eye(p, dtype=A.dtype),
        ATy,
    )

    u_rec_norm = A @ z_star
    u_rec_phys = denormalize_u(u_rec_norm[None, :], normalizer)[0]

    return z_star, u_rec_phys, u_rec_norm


z_lstsq, u_lstsq_pod, u_lstsq_pod_norm = solve_latent_ridge_lstsq(
    basis=basis_pod,
    u_target_phys=u_exact_pod,
    normalizer=normalizer,
    config=cfg,
    ridge=1e-8,
)

print("z_lstsq shape:", z_lstsq.shape)
print("u_lstsq_pod shape:", u_lstsq_pod.shape)
print("u_lstsq min/max:", float(u_lstsq_pod.min()), float(u_lstsq_pod.max()))


def rel_l2(pred, ref, eps=1e-12):
    return jnp.linalg.norm(pred - ref) / jnp.clip(jnp.linalg.norm(ref), a_min=eps)


u_exact_vec = u_exact_pod[0]

err_lstsq_pod = rel_l2(u_lstsq_pod, u_exact_vec)
max_abs_err = jnp.max(jnp.abs(u_lstsq_pod - u_exact_vec))
mean_abs_err = jnp.mean(jnp.abs(u_lstsq_pod - u_exact_vec))

print("===== Best trunk-basis projection error on POD grid =====")
print("Relative L2 error:", float(err_lstsq_pod))
print("Max abs error    :", float(max_abs_err))
print("Mean abs error   :", float(mean_abs_err))


# Branch encoded latent
u_exact_pod_norm = normalize_u(u_exact_pod, normalizer)

z_branch = fe_state.apply_fn(
    {"params": fe_state.params},
    u_exact_pod_norm,
    method=FunctionEncoder.encode_u,
)[0]

# Reconstruct branch result using same basis
A_pod = basis_pod / jnp.sqrt(cfg.n_basis)
u_branch_norm = A_pod @ z_branch
u_branch_pod = denormalize_u(u_branch_norm[None, :], normalizer)[0]

err_branch_pod = rel_l2(u_branch_pod, u_exact_vec)

latent_diff_rel = (
    jnp.linalg.norm(z_branch - z_lstsq)
    / jnp.clip(jnp.linalg.norm(z_lstsq), a_min=1e-12)
)

print("===== Branch latent vs optimal latent =====")
print("Branch reconstruction RL2:", float(err_branch_pod))
print("Optimal projection RL2   :", float(err_lstsq_pod))
print("Relative latent diff     :", float(latent_diff_rel))


key_probe = jax.random.PRNGKey(cfg.seed + 24680)

probe_coords = sobol_annulus_points(
    key_probe,
    cfg.random_probe_points,
    cfg.r_inner,
    cfg.r_outer,
    cfg.dim,
)

probe_coords_np = np.asarray(jax.device_get(probe_coords))

u_exact_probe_np, _, _ = exact_f0_solution_np(
    coords_np=probe_coords_np,
    k=k_value,
    a=cfg.r_inner,
    R=cfg.r_outer,
)

u_exact_probe = jnp.asarray(u_exact_probe_np)

basis_probe = trunk_apply(probe_coords)
A_probe = basis_probe / jnp.sqrt(cfg.n_basis)

u_lstsq_probe_norm = A_probe @ z_lstsq
u_lstsq_probe = denormalize_u(u_lstsq_probe_norm[None, :], normalizer)[0]

err_lstsq_probe = rel_l2(u_lstsq_probe, u_exact_probe)

print("===== Generalization to probe points =====")
print("Projection RL2 on pod grid    :", float(err_lstsq_pod))
print("Projection RL2 on probe points:", float(err_lstsq_probe))


x = pod_coords_np[:, 0]
y = pod_coords_np[:, 1]

u_exact_np = np.asarray(jax.device_get(u_exact_vec))
u_lstsq_np = np.asarray(jax.device_get(u_lstsq_pod))
u_branch_np = np.asarray(jax.device_get(u_branch_pod))

abs_err_lstsq = np.abs(u_lstsq_np - u_exact_np)
abs_err_branch = np.abs(u_branch_np - u_exact_np)

cmin = min(u_exact_np.min(), u_lstsq_np.min(), u_branch_np.min())
cmax = max(u_exact_np.max(), u_lstsq_np.max(), u_branch_np.max())

plt.figure(figsize=(18, 8))

plt.subplot(2, 3, 1)
plt.scatter(x, y, c=u_exact_np, s=10, cmap="turbo", vmin=cmin, vmax=cmax)
plt.axis("equal")
plt.colorbar()
plt.title("Exact solution")

plt.subplot(2, 3, 2)
plt.scatter(x, y, c=u_lstsq_np, s=10, cmap="turbo", vmin=cmin, vmax=cmax)
plt.axis("equal")
plt.colorbar()
plt.title(f"Optimal latent projection\nRL2={float(err_lstsq_pod):.3e}")

plt.subplot(2, 3, 3)
plt.scatter(x, y, c=u_branch_np, s=10, cmap="turbo", vmin=cmin, vmax=cmax)
plt.axis("equal")
plt.colorbar()
plt.title(f"Branch-encoded reconstruction\nRL2={float(err_branch_pod):.3e}")

plt.subplot(2, 3, 5)
plt.scatter(x, y, c=abs_err_lstsq, s=10, cmap="hot")
plt.axis("equal")
plt.colorbar()
plt.title("|projection - exact|")

plt.subplot(2, 3, 6)
plt.scatter(x, y, c=abs_err_branch, s=10, cmap="hot")
plt.axis("equal")
plt.colorbar()
plt.title("|branch reconstruction - exact|")

plt.tight_layout()
plt.show()