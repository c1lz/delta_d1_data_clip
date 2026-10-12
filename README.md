# data_clip — v1.3 BiLSTM 手势切分推理包

该目录是可离线复用的 v1.3 BiLSTM 推理包。模型以 v1.2 权重热启动，并在完整 v1.2 回放集上补充了 2026-10-08 Cabinet 的 16 条人工 GT；Cabinet_022、Cabinet_023 未参与训练，用作独立诊断样本。它只包含 MediaPipe 手部特征提取、模型推理、切分点解码和导出代码；不包含训练视频、特征缓存、数据集划分或 GT。

冻结实验选择策略为阈值 `0.05`、合并间隔 `6` 帧，每个阈值连续区域取最高分帧作为切分点。桌面软件仍使用可配置的运行时默认值：阈值 `0.80`、合并间隔 `6` 帧、相邻切分点最少 `4` 秒、首尾补点窗口 `2` 秒；其任务报告会同时记录实际设置和 v1.3 权重哈希。模型要求输入是可解码的 30 FPS 视频。

## 安装

建议使用 Python 3.12：

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0+cpu
.venv/bin/pip install -r requirements.inference.txt
```

有 CUDA 环境时，可安装与本机 CUDA 匹配的 PyTorch，再安装其余依赖。

## 校验与推理

```bash
./data_clip verify
./data_clip infer --video /path/to/cam0-00000.mp4 --output /path/to/result.json --export-frames
```

输出包括 `result.json`（切分点、时间、分段区间）、`result.probabilities.npz`（逐帧模型分数）、`result.features.npz`（MediaPipe 特征）以及可选的 `result_cut_frames/`。推理过程不读取 GT。

`SHA256SUMS` 和 `model/package_manifest.json` 可用于验证包内权重、MediaPipe 模型和运行时代码未被修改。
