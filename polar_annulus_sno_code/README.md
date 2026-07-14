# 极坐标固定圆环 SNO

本代码针对固定圆环

\[
(r,\theta)\in[0.2,1.0]\times[0,2\pi)
\]

求解

\[
P_{rr}+\frac1rP_r+\frac1{r^2}P_{\theta\theta}-k^2P=f,
\]

边界条件为

\[
P(1,\theta)=0,
\qquad
g_n(\theta)=\frac{\partial P}{\partial n}=-P_r(0.2,\theta).
\]

真实推理问题使用

\[
f=0,
\qquad g_n(\theta)=\cos\theta.
\]

## 1. 归一化坐标

所有 FE/DeepONet 坐标保存为

\[
\widehat\theta=\frac{\theta}{\pi}-1,
\qquad
\widehat r=\frac{2(r-r_{\mathrm{in}})}{r_{\mathrm{out}}-r_{\mathrm{in}}}-1,
\]

所以计算域映射到 `[-1,1]^2`。

为避免角度接缝，BNN 和 trunk 不直接使用 `theta_hat`，而使用

```text
[sin(theta), cos(theta), r_hat]
```

作为周期特征。

## 2. PI-sampler

随机 BNN 直接生成物理径向导数

\[
q(r,\theta)=P_r(r,\theta).
\]

单隐层 Fourier BNN 为

\[
q=\sqrt{\frac2H}\sum_{j=1}^H a_j
\cos\!\left(
 w_{s,j}\sin\theta+w_{c,j}\cos\theta
 +w_{r,j}\widehat r+b_j-\frac\pi4
\right).
\]

压力由外边界向内解析积分：

\[
P(r,\theta)=\int_{r_{\mathrm{out}}}^r q(s,\theta)\,ds.
\]

代码使用稳定差商

\[
\frac{\sin(A+w\widehat r)-\sin(A+w)}{w}
=(\widehat r-1)
\cos\!\left(A+\frac{w(\widehat r+1)}2\right)
\mathrm{sinc}\!\left(\frac{w(\widehat r-1)}2\right),
\]

因此 `w_r=0` 或非常小时自动取到正确极限，不需要除零分支。

源项解析计算为

\[
f=q_r+\frac qr+\frac1{r^2}P_{\theta\theta}-k^2P.
\]

训练边界通量为

\[
g_n=-q(r_{\mathrm{in}},\theta).
\]

## 3. Function Encoder

- Branch：输入极坐标规则网格 `[Nr,Ntheta]` 上的 `P` 或 `f`。
- `theta` 方向采用 circular padding。
- `r` 方向采用 edge padding。
- Trunk 输入存储坐标 `[theta_hat,r_hat]`，内部转换为周期特征。
- `P` 解码基函数乘以

\[
\frac{1-\widehat r}{2},
\]

从而严格满足 `P(r_outer,theta)=0`。

为保证解码后物理零边界不被均值平移破坏，`P` 使用 scale-only normalization：

\[
P_{\mathrm{norm}}=P/\mathrm{RMS}(P),
\]

即 `mean_p` 固定为零；`f` 仍使用普通均值/标准差归一化。

## 4. FE 物理损失

trunk 对归一化坐标求导后，代码显式加入尺度因子：

\[
\partial_r=\frac{2}{r_{\mathrm{out}}-r_{\mathrm{in}}}\partial_{\widehat r},
\qquad
\partial_\theta=\frac1\pi\partial_{\widehat\theta}.
\]

默认只在 `fe_physics_points=128` 个 probe 点计算 Hessian，数据重构损失仍使用全部 probe 点。

## 5. Transformer

Transformer 学习

\[
(z_f,z_g,k)\mapsto z_P.
\]

边界 token 为

```text
[sin(theta), cos(theta), r_hat, g_n]
```

真实推理函数 `predict_zero_source_cosine_flux` 使用

```text
f = 0
g_n = -P_r = cos(theta)
```

## 6. 文件

- `config_polar.py`：配置与尺度因子。
- `data_polar.py`：解析 PI-sampler、坐标变换、归一化和 token。
- `models_polar.py`：周期 CNN、极坐标 trunk、Function Encoder、Transformer。
- `train_polar.py`：FE/Transformer 训练、物理损失、保存加载与目标推理。
- `run_polar.py`：完整训练入口。
- `test_polar_prior.py`：解析公式与符号测试。
- `test_model_smoke.py`：FE 和 Transformer 单步训练测试。
- `test_physics_smoke.py`：极坐标物理残差反向传播测试。

## 7. 运行

```bash
pip install -r requirements.txt
python -m unittest -v test_polar_prior.py
python -m unittest -v test_model_smoke.py
python -m unittest -v test_physics_smoke.py
python run_polar.py
```

正式训练前建议先在 `run_polar.py` 中使用小配置做 GPU 冒烟测试，再恢复默认配置。

## 8. 已完成验证

- `P_r=q` 与 JAX 自动微分一致。
- `P_{rr}=q_r` 与 JAX 自动微分一致。
- `P_{theta theta}` 与 JAX 二阶自动微分一致。
- `P(r_outer,theta)=0` 严格成立。
- `w_r=0` 的解析积分极限正确且无 NaN/Inf。
- `g_n=-P_r` 的符号实现正确。
- `g_n=cos(theta)` 的真实推理 token 正确。
- Function Encoder 单步反向传播通过。
- 极坐标物理残差单步反向传播通过。
- Transformer 单步反向传播通过。
