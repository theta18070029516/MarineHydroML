# polar_annulus_sno_code_v4

## 目标

v4 在不修改 `polar_annulus_sno_code` 的前提下完成两项改动：

1. 每个样本独立生成 $(\sigma_{\theta},\sigma_r)$，不再使用固定尺度对；
2. FE 和 OL 训练期间同时记录随机验证与固定解析真解监控。

所有 Python 代码均应在 Miniconda `jax` 环境中运行。

## 核心方案

### 1. 连续且独立的尺度采样

对 batch 中每个样本 $i$，分别使用独立随机子键采样

$$
\sigma_{\theta}^{(i)}\sim U[\sigma_{\theta,\min},\sigma_{\theta,\max}],
\qquad
\sigma_r^{(i)}\sim U[\sigma_{r,\min},\sigma_{r,\max}].
$$

默认范围均为 $[3,7]$，因此训练分布由原来的三个位于对角线上的固定点扩展为整个尺度矩形。采样结果保存在 `SampleBatch.sigma_theta` 和 `SampleBatch.sigma_r` 中，可以直接审查或导出。

角向尺度仍同时作用于 $\sin{\theta}$ 和 $\cos{\theta}$ 特征；偏置尺度按每个样本的 $(2\sigma_{\theta}+\sigma_r)/3$ 计算。样本生成采用分块计算控制峰值显存，分块只改变计算方式，不改变采样分布。

v4 中 `sample_size` 表示总 batch 大小：

| 阶段 | v3 总 batch | v4 总 batch | 生成分块 |
| --- | ---: | ---: | ---: |
| FE | $3\times256=768$ | 768 | 256 |
| OL | $3\times128=384$ | 384 | 128 |

这样可以在扩大尺度分布覆盖范围的同时，基本保持原实验的每步样本数和生成阶段峰值显存。由于先验分布已经改变，v4 会重新计算归一化统计量；不应混用 v1–v3 的 normalizer、FE 或 OL checkpoint。

### 2. 两条相互独立的评估链

随机验证链每次使用新的随机 batch，衡量模型对完整连续先验分布的平均表现。

解析监控链固定使用

$$
f=0,
\qquad
g_n=\cos{\theta},
\qquad
k=1,
$$

并与圆环问题的 Fourier–Bessel 解析解比较。该样本只执行前向评估，不参与损失函数、梯度或归一化估计。

FE 解析监控包括：

- 压力真实解的网格相对 $L_2$、面积加权相对 $L_2$、RMSE 和相对 $L_{\infty}$；
- 对真实源项 $f=0$ 的重构 RMSE 与最大绝对误差；
- 外边界 Dirichlet 最大误差。

OL 解析监控包括：

- 压力预测的网格相对 $L_2$、面积加权相对 $L_2$、RMSE 和相对 $L_{\infty}$；
- 预测 latent 与解析压力经 FE 编码所得 latent 的相对 $L_2$；
- 外边界 Dirichlet 最大误差和内边界通量相对 $L_2$。

面积加权指标在高分辨率单元中心网格上使用径向权重 $r$，对应极坐标面积元。内边界通量通过连续 FE 解码器的自动微分计算，不使用有限差分。

### 3. 记录与显存控制

解析评估前会等待当前训练步完成并释放大训练 batch，再创建小随机验证 batch 或单样本解析基准，避免训练张量和验证张量同时占用峰值显存。

主要可调间隔如下：

| 配置项 | 默认值 | 含义 |
| --- | ---: | --- |
| `fe_log_interval` | 500 | FE 随机验证与训练日志 |
| `fe_exact_eval_interval` | 5000 | FE 解析真解重构 |
| `fe_checkpoint_interval` | 10000 | FE checkpoint |
| `ol_log_interval` | 500 | OL 随机验证与训练日志 |
| `ol_exact_eval_interval` | 5000 | OL 解析真解预测 |
| `ol_checkpoint_interval` | 10000 | OL checkpoint |

训练输出位于 `out_polar_annulus_sno_v4/polar_v4`：

- `config_fe.json`、`config_ol.json`：分别保留两个阶段的完整配置；
- `fe_training_history.csv/.npz`：FE 随机验证历史；
- `fe_exact_history.csv/.npz`：FE 解析真解历史；
- `operator_training_history.csv/.npz`：OL 随机验证历史；
- `operator_exact_history.csv/.npz`：OL 解析真解历史；
- `exact_monitor/fe_step_*.png`、`exact_monitor/ol_step_*.png`：解析场快照；
- FE、OL 参数和归一化文件。

## 合理性评估

### 结论：方案合理，可以进入 v4 训练

独立均匀采样直接满足尺度解耦要求，并覆盖原先没有出现的组合，例如“小角向尺度 + 大径向尺度”。保持 FE 和 OL 的原总 batch 大小可以减少优化噪声变化，使 v3 与 v4 的对比更接近“只改变先验尺度分布”。

同时保留随机验证和解析监控是必要的：固定解析问题便于解释训练进展，但它只代表一个 $f=0$、一阶角向模态的切片，不能替代整个连续先验矩形上的随机验证。

需要保留以下实验边界：

- 连续均匀分布几乎不会精确抽到区间端点。若最终结论关注最极端尺度组合，应在训练后额外使用尺度矩形的角点和规则网格做独立压力先验测试。
- 反复查看固定解析曲线并据此选择 checkpoint 后，该解析问题属于 validation monitor，不再是无偏最终测试。最终测试应更换 Fourier 模态、相位或训练期间未查看的 $k$ 组合。
- 更大的尺度矩形会改变 $P$ 和 $f$ 的幅值与频谱分布，因此必须使用 v4 数据重新估计 normalizer；代码已强制采用这一流程。

## 运行入口

1. 运行 `train_function_encoder_polar_v4.ipynb`；首次迁移环境时可先使用 `RUN_MODE="smoke"`。
2. 确认 FE checkpoint、归一化文件和两份 FE 历史已经生成。
3. 运行 `train_operator_polar_v4.ipynb`。
4. 训练后使用未参与调参的解析案例和尺度角点测试完成最终评估。

也可以直接运行 `run_polar.py` 连续训练 FE 与 OL。

## 已执行检查

- 逐样本尺度范围、独立性和 batch 元数据检查；
- BNN 解析导数、外边界条件与零径向频率极限检查；
- FE、OL 小模型前向与单步反向传播；
- Fourier–Bessel 解析解的外边界值和内边界通量检查；
- FE 与 OL 解析监控的形状、有限值和严格外边界检查。
