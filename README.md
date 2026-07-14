# MarineHydroML

面向海洋工程动水力问题的物理信息机器学习实验仓库。项目以二维环域上的压力型标量场 $P$ 为研究对象，围绕参数化椭圆型方程

$$
\Delta P-k^2P=f
$$

构建并比较多种 **Scientific Neural Operator（SNO）** 管线。核心目标是学习前向算子

$$
\mathcal G:(f,\,g,\,k,\,\Gamma)\mapsto P,
$$

其中 $f$ 为源项，$g$ 为内边界通量，$k$ 为方程参数，$\Gamma$ 表示可选的内边界几何。

> 本仓库是研究型代码库，不是通用 CFD 求解器。所有模型结果均应通过边界条件、PDE 残差及独立 BVP/FEM 对照验证后再用于工程解释。

## 项目路线

|模块|问题与几何|主要目的|当前重点|
|---|---|---|---|
|[`annulus_sno/`](annulus_sno/)|固定环域，Notebook 运行基线为 $r\in[0.2,1.0]$|建立固定几何 SNO 基线|PI-sampler、FE 与算子 latent 学习|
|[`var_boundary_sno_code/`](var_boundary_sno_code/)|任意星形内边界，外边界为 $b(\theta)=5a(\theta)$|研究几何变化下的算子泛化|标准域映射、物理域残差、FEM/PCG 对照|
|[`polar_annulus_sno_code/`](polar_annulus_sno_code/)|固定极坐标环域 $[0.2,1.0]\times[0,2\pi)$|构建周期极坐标实现|解析 PI-sampler、周期特征、严格外边界|
|[`polar_annulus_p_prior_ablation/`](polar_annulus_p_prior_ablation/)|极坐标固定环域|隔离压力先验幅值的影响|比较 `p_raw` 与 `p_rms`，含可恢复训练与预检|

## 研究框架

每条 SNO 管线都遵循相同的三层思路：

1. **PI-sampler**：从先验随机场生成满足外边界条件的 $P$，再由自动微分或解析公式得到 $f$ 和内边界通量。
2. **Function Encoder（FE）**：将 $P$ 与 $f$ 编码为低维 latent，并通过共享 trunk 在查询点重构连续场。
3. **Operator Transformer（OL）**：学习源项 latent、边界 token 与参数 $k$ 到解 latent 的映射，最后由 FE 重构 $P$。

固定环域与变边界方案的完整实现约定见：

- [固定环域 SNO 实施方案](annulus_sno/annulus-方案.md)
- [任意内边界环域 SNO 实施方案](var_boundary_sno_code/varboundary-方案.md)

## 目录结构

```text
MarineHydroML/
├── annulus_sno/
│   ├── annulus-方案.md
│   ├── annulus_sno_annulus_only_v2/
│   │   └── annulus_sno_annulus_only_v2/   # 当前固定环域实现与 Notebook
│   └── data/                              # 早期实验脚本与诊断材料
├── var_boundary_sno_code/
│   ├── varboundary-方案.md
│   ├── var_boundary_sno_code/             # 变内边界 SNO 主实现
│   └── data/                              # FEM、重构和 PCG 对照脚本
├── polar_annulus_sno_code/                # 周期极坐标 SNO
├── polar_annulus_p_prior_ablation/        # P-prior 消融实验
├── report/                                # 项目材料与图件
└── 问题描述.pdf                            # 问题背景与约定
```

## 环境

本项目统一使用 Miniconda 的 `jax` 环境：

```powershell
$py = 'C:\Users\Hollon\miniconda3\envs\jax\python.exe'
& $py --version
```

不要使用系统 Python、Windows Store Python 或 Miniconda `base` 环境。各子模块可能维护独立的 `requirements.txt`；安装依赖时请在目标模块内执行，避免将实验依赖混为一套环境。

```powershell
& $py -m pip install -r .\polar_annulus_sno_code\requirements.txt
& $py -m pip install -r .\polar_annulus_p_prior_ablation\requirements.txt
```

## 推荐阅读与运行顺序

### 1. 固定环域基线

从 [`annulus_sno/annulus-方案.md`](annulus_sno/annulus-方案.md) 开始。当前实验通过 Notebook 覆盖配置：

```python
cfg.r_inner = 0.2
cfg.r_outer = 1.0
```

因此，Notebook 的配置快照与所加载的检查点必须一致；不要将 Python 配置文件中的历史默认半径直接视为当前实验基线。建议按“sampler 验证 → FE 重构 → OL 训练 → 独立 BVP 对照”的顺序推进。

### 2. 极坐标固定环域实现

该模块包含解析 PI-sampler、单元测试与完整训练入口，是检查边界符号和物理残差链路的首选实现。

```powershell
Push-Location .\polar_annulus_sno_code
& $py -m unittest -v test_polar_prior.py
& $py -m unittest -v test_model_smoke.py
& $py -m unittest -v test_physics_smoke.py
& $py .\run_polar.py
Pop-Location
```

正式训练前，应先在 `run_polar.py` 中设置小规模 GPU 冒烟配置。

### 3. 变内边界泛化

变几何模块在标准域 $(\rho,\theta)$ 上编码，在物理域 $(x,y)$ 上计算 PDE 与边界算子。先验证映射、PI-sampler 和纯数据 FE；物理残差需要逆映射与二阶自动微分，当前代码已实现但默认训练损失中关闭。

```powershell
Push-Location .\var_boundary_sno_code\var_boundary_sno_code
# 先在 run_train_varboundary.py 中设定小配置，再运行：
& $py .\run_train_varboundary.py
Pop-Location
```

### 4. P-prior 消融

该模块是独立快照，用于比较未经缩放的 `p_raw` 与边界通量 RMS 预注册缩放的 `p_rms`。先运行测试和两步 smoke 流程：

```powershell
Push-Location .\polar_annulus_p_prior_ablation
& $py -m pytest -q
& $py .\run_prior_ablation.py --variant p_raw --seed 0 --stage all --resume --smoke
& $py .\run_prior_ablation.py --variant p_rms --seed 0 --stage all --resume --smoke
Pop-Location
```

## 验证标准

一个实验结果至少应同时报告下列证据：

|层面|检查内容|
|---|---|
|数据生成|外边界 $P=0$、$f=\Delta P-k^2P$、通量符号与构造场一致|
|FE|$P$/$f$ 重构相对 $L_2$ 误差、归一化统计量和必要的物理残差|
|OL|latent MSE 以及重构后物理解的相对 $L_2$ 误差|
|目标边界|目标 $g(\theta)$ 的 token 必须显式构造，不能复用随机先验样本的诱导通量|
|独立对照|解析解、BVP 或 FEM 参考解；变边界时还应报告 PCG warm-start 收益|

变内边界问题中的物理残差必须对 $P\!\left(\Phi^{-1}(x,y)\right)$ 关于物理坐标求导；不能沿用固定环域的拉普拉斯表达式。

## 数据、输出与版本控制

源代码、配置、Notebook、实施方案和必要图件应进入 Git。训练权重、检查点、缓存和大规模实验产物不应直接提交：

- `out/`、`out_*/`：训练输出与评估结果；
- `*.msgpack`、`*.npy`、`*.npz`：模型参数和数组产物；
- `*.mat`：MATLAB 数据与大型诊断文件；
- `__pycache__/`、`.ipynb_checkpoints/`：本地缓存。

每次正式实验应独立保存配置快照、随机种子、归一化统计量、检查点、训练曲线和评估指标。若需要共享大模型或数据集，建议使用 Git LFS、Zenodo 或其他专用数据存储，而非普通 Git 历史。

## 贡献约定

- 新实验应明确其几何、边界通量符号、参数范围和参考解来源。
- 修改采样、归一化、网络结构或损失时，必须重新生成对应配置快照和验证结果。
- 运行训练、测试和 Notebook 时，始终使用 `C:\Users\Hollon\miniconda3\envs\jax\python.exe`。
- 提交前检查输出目录与大文件是否被 `.gitignore` 排除。

## 状态说明

项目正处于研究迭代阶段。模块之间共享问题设定与验证原则，但训练配置、先验设计和检查点不可混用。解释模型优劣时，应区分“坐标体系、先验、网络结构、训练策略”这些可能同时变化的因素，避免把单次比较归因于单一设计选择。
