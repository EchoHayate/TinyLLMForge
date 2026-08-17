# T1.3 可行性探针实验

目标：验证“被裁 KV 是否能被保留 KV 低成本复原”这个核心假设。

当前已落地的第一版探针是 [tools/probe_kv_recovery.py](/Users/bytedance/dev/TinyLLMForge/tools/probe_kv_recovery.py)，它读取 HuggingFace 模型的 `past_key_values`，计算：

- 跨 layer 同 head 的 K/V token cosine heatmap。
- 跨 layer 同 head 的 per-channel diagonal affine R2 heatmap。
- 跨 head 同 layer 的最佳可预测源 head。
- 初步 go / borderline / no-go 结论。

## 快速自检

不需要真实模型权重：

```bash
python3 tools/probe_kv_recovery.py \
  --synthetic \
  --output-dir light-doc-cache-plan/t1_3_probe/runs/synthetic
```

## 真实模型运行

使用本地 HuggingFace 权重路径：

```bash
python3 tools/probe_kv_recovery.py \
  --model ~/Qwen3-0.6B \
  --text-file docs/kv-sparse-attention.md \
  --max-tokens 2048 \
  --max-sample-tokens 1024 \
  --output-dir light-doc-cache-plan/t1_3_probe/runs/qwen3_0_6b
```

如果要允许 HuggingFace 下载模型文件，显式加 `--allow-download`。当前脚本默认 `local_files_only=True`，避免在离线环境里卡住。

## 产物

每次运行会在 output dir 下生成：

- `summary.json`：机器可读指标和结论。
- `report.md`：人可读报告。
- `layer_k_cosine.csv`、`layer_v_cosine.csv`：跨层相关性热力图数据。
- `layer_k_diag_r2.csv`、`layer_v_diag_r2.csv`：低成本对角仿射复原 R2 数据。
- `best_cross_head.csv`、`best_cross_layer.csv`：每个目标 layer/head 的最佳候选来源。
- 若安装了 matplotlib，还会生成对应 PNG heatmap。

## 判定口径

默认阈值：

- `GO`：最佳 cross-layer 或 cross-head 的平均 joint K/V diagonal-affine R2 >= 0.50。
- `BORDERLINE`：平均 joint R2 介于 0.35 和 0.50。
- `NO-GO`：平均 joint R2 < 0.35。

这里的结论只是 T1.3 第一轮探针，不代表最终压缩收益。若结果是 GO 或 BORDERLINE，下一步才值得进入 T2.1/T2.2 的复原算子设计。
