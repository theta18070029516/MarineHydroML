# 变几何极坐标 SNO v1

本项目是独立同级实现，参考 `var_boundary_sno_code` 的周期 BNN 几何生成器和
`polar_annulus_sno_code_v4` 的周期 CNN、共享 trunk、Function Encoder（FE）与
Operator Transformer（OL）。两个参考项目均不需要修改。

## 1. 已实现的问题

物理域和方程为

$$
\Omega_a=\{(\theta,r):a(\theta)\le r\le5a(\theta)\},
\qquad
\Delta P-k^2P=f,
\qquad
k\sim U[0.2,2.0].
$$

外边界由 decoder mask 严格施加 $P=0$。内边界使用非单位算子

$$
\mathcal B_aP=
\left(-e_r+\frac{a'}a e_\theta\right)\cdot\nabla P,
$$

最终目标载荷为

$$
g_{\mathrm{target}}=\cos\theta+\frac{a'}a\sin\theta.
$$

MATLAB 弱式使用计算域单位外法向通量

$$
q_\Omega=\frac{g_{\mathrm{target}}}
{\sqrt{1+(a'/a)^2}}.
$$

因此网络中的 $\mathcal B_aP=g_{\mathrm{target}}$ 与 FEM 中的
$\partial_{n_\Omega}P=q_\Omega$ 是同一个边界条件的非单位、单位两种表示。

## 2. 几何和归一化坐标

几何生成器为

$$
a(\theta)=0.12+0.08\tanh\!\left(0.3r_{\mathrm{BNN}}(\theta)\right),
$$

输入为 $(\sin\theta,\cos\theta)$，隐藏宽度为 256，`geom_sigma=3`。
代码解析计算 $a$、$a'$ 和 $a''$，避免在 PI-sampler 热路径中重复自动微分。

共享参考坐标为

$$
\xi=\frac{\theta}{\pi}-1,
\qquad
\eta=\frac12\left(\frac r{a(\theta)}-3\right),
$$

反变换为

$$
\theta=\pi(\xi+1),
\qquad
r=a(\theta)(3+2\eta).
$$

`data_varpolar.transformed_derivatives` 实现完整变系数链式关系：

$$
\partial_r=\frac1{2a}\partial_\eta,
\qquad
\left.\partial_\theta\right|_r=
\frac1\pi\partial_\xi-\frac{a'}{2a}(3+2\eta)\partial_\eta.
$$

二阶角向变系数项 $a''$ 也包含在物理拉普拉斯中，并由直接物理坐标 JAX
高阶自动微分测试覆盖。

## 3. PI-sampler 与训练数据契约

每个样本独立抽取

$$
\sigma_\theta\sim U[0.5,2],
\qquad
\sigma_r\sim U[0.5,5].
$$

先由周期 BNN 构造 $q=P_r$，再解析积分

$$
P(\theta,\eta)=2a(\theta)\int_1^\eta q(\theta,\zeta)\,\mathrm d\zeta.
$$

这样 $P(\eta=1)=0$ 和 $P_r=q$ 均为构造恒等式。积分使用稳定 `sinc`
差商；同一组 prior 参数、几何和 $k$ 同时产生 POD、probe 以及内边界数据。
随机训练边界载荷始终是该随机 $P$ 的 $\mathcal B_aP$，不会被
$g_{\mathrm{target}}$ 替换。

`SampleBatch` 保存共享的参考坐标、`p_pod/f_pod/p_probe/f_probe`、几何参数、
边界 $a$、$a'/a$、随机边界载荷、$k$、$\sigma_\theta$ 和 $\sigma_r$。

## 4. Operator Transformer 边界特征

每个边界点使用

$$
[\sin\theta,\cos\theta,\hat a,\hat h,\hat g],
\qquad
\hat a=\frac{a-0.12}{0.08}.
$$

$\hat h$ 和 $\hat g$ 使用 PI 训练分布统计量标准化。这里五项分别提供周期位置、
局部尺度、局部斜率和实际边界载荷；仅给物理坐标无法显式区分几何斜率，
而边界算子正好依赖该斜率。特征先按连续角向小段分块，所以
`cond_chunk_width = 5 * boundary_chunk_size`。

FE 的 trunk 输入为 $(\sin\theta,\cos\theta,\eta)$。压力采用零均值 RMS
缩放，压力 decoder 使用 $(1-\eta)/2$ mask，因此反归一化后的外边界仍严格为零；
$f$ 使用均值和标准差归一化。

## 5. 固定 FEM 验证集

`export_fem_manifest` 使用独立固定种子一次性产生 100 组几何参数和 100 个独立
$k$。MATLAB 对每个案例求解 $f=0$、$g=g_{\mathrm{target}}$，并执行：

- double 精度 `ichol + pcg`，要求 `flag=0` 且 `relres <= 1e-10`；
- 网格序列 $(65,256)$、$(129,512)$、必要时 $(257,1024)$；
- 相邻网格投影到共同监控网格后的面积加权相对差不超过 $5\times10^{-4}$；
- 任一案例在最高层仍不满足条件时，整个数据集生成失败并报告案例编号。

物理面积权重按 $2ar$ 保存。固定 100 案例不参与训练、梯度或 normalizer，
但用于 `best_fem` checkpoint 选择，因此它们是验证集而不是最终论文测试集。

## 6. 运行

本仓库要求使用 JAX Conda 环境：

```powershell
$python = 'C:\Users\Hollon\miniconda3\envs\jax\python.exe'
Set-Location 'D:\A4S\海洋工程动水力-机器学习\polar_var_boundary_sno_code_v1'
```

先生成固定清单，再运行高精度 MATLAB FEM：

```powershell
& $python run_varpolar.py manifest
& $python run_varpolar.py fem --reuse-manifest
```

得到 100 案例后分别训练 FE 和 OL：

```powershell
& $python run_varpolar.py fe
& $python run_varpolar.py ol
```

也可一次运行完整流程：

```powershell
& $python run_varpolar.py all
```

默认完整流程包含 500k FE 步、300k OL 步以及最多 100 个最高
$(257,1024)$ 网格 FEM 求解，属于长时间计算任务。若需修改配置，可先保存一份
`VarPolarConfig` JSON，再用 `--config path/to/config.json` 载入；也可用
`--out-dir` 和 `--run-name` 单独覆盖输出位置。

## 7. 实时监控和输出

训练每 500 步运行普通随机验证，每 5000 步以每块 10 个案例评估完整 FEM
验证集，每 10000 步保存 `latest`。FEM 面积加权压力 RL2 均值降低时保存
`best_fem`。

FE 记录压力面积/网格 RL2、RMSE、相对最大误差、$f=0$ 的 RMSE/最大误差、
外边界误差和解码内边界算子误差。OL 还记录预测 latent 相对误差。
每项保存逐案例值以及 mean、median、P95、max；历史同时原子写入 CSV 和 NPZ，
并保存固定案例图。

## 8. 验证

快速 Python 测试包括几何导数、坐标闭合、物理 AD 拉普拉斯、笛卡尔边界点积、
prior 恒等式、随机分布、圆形极限、FE/OL 前后向和 FEM 分块监控：

```powershell
& $python run_varpolar.py test
```

可选 MATLAB 圆环联调会将 FEM 解与 Bessel 解析解比较，同时验证几何跨语言一致性、
边界符号和 `.mat` 展平顺序：

```powershell
& $python verify_matlab_fem.py
```

该联调使用缩小网格，仅作为接口和符号测试；正式 100 案例仍必须使用默认高精度配置。

## 9. 主要文件

- `config_varpolar.py`：全部默认配置和 JSON 载入/保存。
- `data_varpolar.py`：几何、坐标变换、解析 PI-sampler、normalizer 和 token。
- `models_varpolar.py`：周期 CNN、共享 trunk、FE 和 Operator Transformer。
- `fem_monitor.py`：固定清单导出、MATLAB 调用、监控集载入和指标。
- `matlab/solve_varpolar_fem.m`：边界贴合 P1 FEM 装配与 PCG。
- `matlab/build_fem_monitor_set.m`：100 案例收敛门槛和数据集写出。
- `train_varpolar.py`：FE/OL 训练、实时监控、历史和 checkpoint。
- `run_varpolar.py`：命令行流水线入口。
