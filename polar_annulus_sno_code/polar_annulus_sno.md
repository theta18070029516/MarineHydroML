Project Memory: polar_annulus_sno_code

- 该子项目在固定圆环 (r\in[0.2,1]) 上求解 (\Delta P-k^2P=f)：外边界严格为 (P=0)，内边界外法向通量为 (g_n=-P_r)。标准解析验证问题为 (f=0,\ g_n=\cos\theta)，解析解在 exact_solution.py；对实数实现使用 (k\ge0)。数据由 q=P_r 的 BNN 先验解析积分生成，保证 (P,f,g_n) 相互一致。
- FE 将 (P,f) 编码到 latent，OL 将源项 token、边界通量 token 和 (k) 映射为 (P) latent；FunctionEncoder.reconstruct_p 的外边界 mask 强制 (P(r_{out})=0)。核心代码分别在 config_polar.py、data_polar.py、models_polar.py、train_polar.py；训练入口是 train_function_encoder_polar.ipynb 与 train_operator_polar.ipynb。
- OL 训练每步从先验重新采样，并以 tqdm、CSV/NPZ history 和 checkpoint 记录进度。评估必须用 ol_eval_sample_size 与 ol_eval_probe_points 的独立小批量，且只传模型参数，避免训练/评估显存峰值叠加。
- 推理与解析对比使用 evaluate_sno_exact_solution.ipynb：优先读取 ol_params.msgpack，缺失时才回退到 ol_params_latest.msgpack；latest 不代表训练完成，必须同时查看 operator_training_history.csv。评估应同时报告原始 POD 网格节点误差，以及高分辨率单元中心的连续场/面积加权误差；中心点用于 pcolormesh 和 (r\,dr\,d\theta) 求积，不替代节点精度。

