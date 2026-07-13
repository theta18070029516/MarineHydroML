# 极坐标 P-prior 消融实验

本目录是一个完全独立的源码快照，用于区分两类可能的性能来源：极坐标及其配套实现，与先验从 $P$ 改为 $\partial_r P$。本项目只训练极坐标 $P$-prior，并比较未经缩放的 `p_raw` 与按边界通量 RMS 预注册缩放的 `p_rms`。旧目录不会被导入、写入或用作训练时依赖。

## 实验变量

周期 BNN 定义

$$
U=\sqrt{\frac{2}{H}}\sum_j a_j\cos\phi_j,
$$

压力先验及解析导数为

$$
P=c_{\sigma_r}(r-r_{\mathrm{out}})U,
$$

$$
P_r=c_{\sigma_r}[U+(r-r_{\mathrm{out}})U_r],
$$

$$
P_{rr}=c_{\sigma_r}[2U_r+(r-r_{\mathrm{out}})U_{rr}],
\qquad
P_{\theta\theta}=c_{\sigma_r}(r-r_{\mathrm{out}})U_{\theta\theta}.
$$

`p_raw` 使用 $c_{\sigma_r}=1$；`p_rms` 使用

$$
c_{\sigma_r}=\frac{1}{\sqrt{1+4\sigma_r^2}}.
$$

对 $\sigma_r=1,3,5$，系数分别为 `0.4472136`、`0.1643990`、`0.0995037`。缩放只依赖预设的 $\sigma_r$，不读取目标 $\cos\theta$，也不按样本调整。

除这一变量外，配置固定为历史 `polar_v2`：`32×128` 网格、1024 probe、512 个基、`512×5` trunk、512 维 8-head 4-layer Transformer、有效 batch size 384、FE/OL 各 500,000 步、`seed=0`。实际来源文件及 SHA-256 见 `baseline_manifest.json`。

## 运行环境

本项目必须使用：

```powershell
C:\Users\Hollon\miniconda3\envs\jax\python.exe
```

先运行测试：

```powershell
cd D:\A4S\海洋工程动水力-机器学习\polar_annulus_p_prior_ablation
C:\Users\Hollon\miniconda3\envs\jax\python.exe -m pytest -q
```

CPU 上可用两步 smoke 流程核验端到端入口：

```powershell
C:\Users\Hollon\miniconda3\envs\jax\python.exe run_prior_ablation.py --variant p_raw --seed 0 --stage all --resume --smoke
C:\Users\Hollon\miniconda3\envs\jax\python.exe run_prior_ablation.py --variant p_rms --seed 0 --stage all --resume --smoke
```

正式训练建议在可用的 JAX GPU/TPU 运行时执行：

```powershell
C:\Users\Hollon\miniconda3\envs\jax\python.exe run_prior_ablation.py --variant p_raw --seed 0 --stage all --resume
C:\Users\Hollon\miniconda3\envs\jax\python.exe run_prior_ablation.py --variant p_rms --seed 0 --stage all --resume
```

也可分阶段执行 `--stage diagnostics`、`fe`、`ol` 或 `eval`。正式 FE/OL 之前，入口会强制执行解析导数、边界符号、RMS 校准、有限值、decoder 外边界和 checkpoint 恢复检查。任何一项失败都会终止训练并生成 `preflight_report.json`。

新模型达到 10k OL 里程碑后，可单独生成与旧 `polar_v2` 的近似同进度描述性报告：

```powershell
C:\Users\Hollon\miniconda3\envs\jax\python.exe evaluate_prior_ablation.py --variant p_raw --seed 0 --checkpoint-step 10000
```

## 输出与续训

两组输出分别位于：

```text
out_p_prior_ablation/polar_p_raw_seed0/
out_p_prior_ablation/polar_p_rms_seed0/
```

每个阶段的 `*_checkpoint_latest.msgpack` 保存参数、优化器状态、已完成步数、训练/评估 PRNG key 和完整配置指纹；对应 JSON 是可读元数据。每 10,000 步覆盖 `latest`，并在 10k、50k、100k、200k、300k、400k、500k 保存参数里程碑。指纹不一致时拒绝恢复，避免把不同实验条件拼接为一次训练。

评估结果写入每组的 `evaluation/`，包括：

- $\alpha\in\{0.25,0.5,1,2,4\}$ 的精确解指标与预测数组；
- 主指标面积加权压力相对 $L_2$；
- 网格相对 $L_2$、相对 $L_\infty$、RMSE、内边界通量相对 $L_2$、外边界误差和 PDE residual；
- 每个先验尺度 256 个固定随机流样本的 FE/OL 分布内误差；
- FE/OL 步数与累计耗时；
- 对旧极坐标 $q$-prior 和笛卡尔 $P$-prior 产物的只读描述性比较。

## 结论边界

当 `p_rms` 的面积加权相对 $L_2$ 和通量误差都较 `p_raw` 改善至少 20%，报告“通量幅值放大是重要影响因素”；两项改善均不足 10% 时报告“幅值校准不是主要因素”；其余情况为“不确定”。本实验只有一个随机种子，不计算显著性或置信区间。

旧 `polar_v2` checkpoint 缺少可靠的训练 step 元数据，因此与新模型 10k 仅能称为“近似同进度描述性比较”。即使极坐标 $P$-prior 优于旧笛卡尔结果，也只能归因于“极坐标方案及其配套实现”，不能声称获得了纯坐标变化的严格因果效应。
