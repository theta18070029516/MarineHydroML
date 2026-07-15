# MarineHydroML

面向海洋工程动水力问题的物理信息机器学习实验仓库。项目以二维环域上的压力型标量场 $P$ 为研究对象，围绕参数化椭圆型方程

$$
\Delta P-k^2P=f
$$

构建并比较多种 **Self-supervised Neural Operator（SNO，自监督神经算子）** 管线，方法框架参照论文 [*Self-supervised neural operator for solving partial differential equations*](https://arxiv.org/abs/2509.00867)。核心目标是学习前向算子

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
|[`polar_annulus_sno_code/`](polar_annulus_sno_code/)|固定极坐标环域 $[0.2,1.0]\times[0,2\pi)$|构建周期极坐标实现|$q=P_r$ 先验、解析积分与解析解验证|
|[`polar_annulus_p_prior_ablation/`](polar_annulus_p_prior_ablation/)|极坐标固定环域|区分坐标方案与压力先验幅值的影响|`p_raw` 阶段性结果；`p_rms` 对照待完成|

## 研究框架

每条 SNO 管线都遵循相同的三层思路：

1. **PI-sampler**：从先验随机场生成满足外边界条件的 $P$，再由自动微分或解析公式得到 $f$ 和内边界通量。
2. **Function Encoder（FE）**：将 $P$ 与 $f$ 编码为低维 latent，并通过共享 trunk 在查询点重构连续场。
3. **Operator Transformer（OL）**：学习源项 latent、边界 token 与参数 $k$ 到解 latent 的映射，最后由 FE 重构 $P$。

固定环域与变边界方案的完整实现约定见：

- [固定环域 SNO 实施方案](annulus_sno/annulus-方案.md)
- [任意内边界环域 SNO 实施方案](var_boundary_sno_code/varboundary-方案.md)
- [极坐标固定环域 SNO 实施与测试结果](polar_annulus_sno_code/polar_annulus_sno_code实施方案.md)
- [极坐标 `p_raw` 测试报告](polar_annulus_p_prior_ablation/P_RAW_TEST_REPORT.md)

## 极坐标固定环域：阶段性结果与研究判断

两套最新测试均采用圆环 $r\in[0.2,1]$ 上的解析工况

$$
f=0,
\qquad
g_n(\theta)=\cos\theta,
\qquad
k=1,
$$

并以解析解、外边界 Dirichlet 条件和内边界通量作为共同验证依据。当前结果支持极坐标方案能够恢复目标压力场的主导 $\cos\theta$ 模态与径向衰减，但所有结论仍属于阶段性结论。

### `polar_annulus_sno_code`：径向导数先验

该方案先生成物理径向导数 $q=P_r$，再从外边界向内积分：

$$
P(r,\theta)=\int_{r_{\mathrm{out}}}^{r}q(s,\theta)\,\mathrm{d}s.
$$

因此 $P(r_{\mathrm{out}},\theta)=0$ 由积分端点自然满足，内边界通量则直接为 $g_n=-q(r_{\mathrm{in}},\theta)$。当前 `polar_v3` 解析测试使用 `ol_params_latest.msgpack`，结果如下；正式训练完成后仍需以最终 checkpoint 复核。

|指标|结果|
|---|---:|
|网格场相对 $L_2$|10.55%|
|面积加权相对 $L_2$|13.34%|
|相对 $L_{\infty}$|6.90%|
|RMSE|$5.01\times10^{-3}$|
|外边界最大 $\lvert P\rvert$|0|
|内边界通量相对 $L_2$|24.05%|

预测场已恢复解析解的主要正负压力区、角向相位和径向衰减趋势。当前主要误差来源是局部场偏差及内边界通量误差；后者也是下一阶段优化先验、训练稳定性和 checkpoint 选择时的重点指标。

### `polar_annulus_p_prior_ablation`：直接压力先验

`p_raw` 直接构造

$$
P=(r-r_{\mathrm{out}})U,
$$

以乘法因子严格施加零外边界。当前报告对应 `seed=0`、$32\times128$ 网格、512 个压力基函数、FE 100k 与 OL 140k checkpoint。

|评估对象|指标|结果|
|---|---|---:|
|当前训练 batch，OL 140k|分布内压力相对 $L_2$|3.01%|
|固定解析解，OL 140k|面积加权压力相对 $L_2$|22.53%|
|固定解析解，OL 140k|压力相对 $L_{\infty}$|11.28%|
|固定解析解训练监控|最佳面积加权压力相对 $L_2$|7.36%（OL 108k）|

该结果说明极坐标压力先验已具备学习目标边值问题的能力，但最新 checkpoint 并非最佳 checkpoint，且分布内误差与固定解析解误差之间仍有明显泛化差距。FE/OL 尚未达到预定训练步数，目前只有一个随机种子，`p_rms` 正式对照也尚未完成，因此暂不能对幅值校准给出最终结论。

### 为什么极坐标相关性可能更适合圆环问题

现有跨方案结果显示，极坐标及其配套周期实现相较旧笛卡尔方案带来了显著的目标工况改善。一个合理的机理解释是：坐标表示改变了先验随机场隐含的相关性度量。

对同一半径 $r$ 上、角度相差 $\Delta\theta$ 的两点，笛卡尔物理距离为

$$
d_{xy}
=\sqrt{\left(\Delta x\right)^2+\left(\Delta y\right)^2}
=2r\left|\sin\left(\frac{\Delta\theta}{2}\right)\right|.
$$

因此，若核相关性随 $d_{xy}$ 单调衰减，则即使 $\Delta\theta$ 不变，两点的相关性也会随 $r$ 增大而减弱。相比之下，对周期角度嵌入

$$
e_{\theta}=(\sin\theta,\cos\theta)
$$

使用平稳相关性时，固定角差对应的嵌入距离为

$$
d_{\theta}=2\left|\sin\left(\frac{\Delta\theta}{2}\right)\right|,
$$

与 $r$ 无关。当前极坐标实现正是使用 `sin(theta)`、`cos(theta)` 与独立的径向坐标特征，并在角向卷积中采用 circular padding。这种表示更直接地表达了圆环边界、周期角向模态以及“同一角差在不同半径上保持相似相关性”的先验，因而可能更接近本问题的压力传播结构。周期输入变换作为结构化函数先验的依据可参见 [Pearce 等（2020）](https://proceedings.mlr.press/v115/pearce20a.html)。

这里的“相关性与 $r$ 无关”是对所选角向先验度量的条件性陈述，并不表示真实欧氏距离或全部物理协方差与 $r$ 无关。极坐标方程中的 $1/r$ 和 $1/r^2$ 系数仍会引入径向依赖。现有比较还同时涉及周期特征、网格拓扑、先验形式和训练过程，因此当前证据支持的是“极坐标方案及其配套实现更匹配问题结构”，尚不是纯坐标变换的严格因果消融。

### 两种外边界处理及当前路线选择

以下区别发生在 PI-sampler 的函数先验层面；两条管线的压力 decoder 还会另外使用外边界 mask，以保证解码结果严格满足零 Dirichlet 条件。

|方案|先验构造|外边界如何进入|主要特征|
|---|---|---|---|
|直接压力先验（`p_raw`/`p_rms`）|$P=c_{\sigma_r}(r-r_{\mathrm{out}})U$|通过乘法因子直接施加|形式简单且严格满足零边界，但 $P_r=c_{\sigma_r}[U+(r-r_{\mathrm{out}})U_r]$，压力幅值、径向尺度与边界通量统计相互耦合|
|径向导数先验（`polar_annulus_sno_code`）|先生成 $q=P_r$，再积分得到 $P$|通过积分端点和积分常数自然纳入|直接控制与输入边界通量同型的导数量，并由 $q\mapsto P$ 的物理关系恢复压力|

两种方法都能严格编码零外边界；直接乘边界距离因子也是常见的硬边界约束方法，可参见 [Sukumar 与 Srivastava（2021）](https://arxiv.org/abs/2104.08426)。就本项目而言，目前更倾向径向导数先验：目标条件本身就是内边界通量，先对 $P_r$ 赋予先验再积分，可以同时把通量变量和外边界条件放在同一构造链路中，物理含义更直接，也避免将外边界影响仅作为压力振幅的乘法包络施加。

这一选择目前是基于问题结构和阶段性结果的研究判断，而不是已经完成的严格优越性证明。后续应在相同网络、样本预算、训练步数和随机种子下比较 $P$-prior 与 $P_r$-prior，并同时报告面积加权场误差、内边界通量误差、训练波动与多半径相关性诊断；同时完成 `p_rms` 对照，检验改善究竟来自坐标度量、先验变量还是通量幅值校准。

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
