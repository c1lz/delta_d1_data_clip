# data_clip 桌面视频切分工具（Linux）

这是冻结 `v1.1 BiLSTM` 推理包的本地桌面界面。它不会训练、替换或修改模型；每次任务均先校验已发布的模型包，再用当前界面设置解码模型分数。

## 安装与启动

需要 Python **3.12**、FFmpeg（含 `ffmpeg` 和 `ffprobe`）。以 Debian/Ubuntu 为例：

```bash
sudo apt install ffmpeg libxcb-cursor0
chmod +x install_desktop_linux.sh run_desktop.sh
./install_desktop_linux.sh
./run_desktop.sh
```

可用 `PYTHON_BIN=/path/to/python3.12 ./install_desktop_linux.sh` 指定 Python 3.12 解释器。安装脚本为 CPU PyTorch；有 CUDA 环境时可先安装与系统匹配的 PyTorch，再安装其余依赖。
虚拟环境目录固定为项目根目录下的 `video_clip/`。

## 使用方式

1. 点击“选择日期-任务文件夹”，在 `/mnt/Datasets/Delta-D1` 中选择形如 `2026-09-02-Oven` 的目录。软件显示日期、任务名称和可用编号范围。
2. 选择起始/结束任务序号，点击“载入所选序号”。每个编号任务只读取其 `cam0/cam0-00000.mp4`；`cam1`、`cam2`、传感器数据和旧结果不会被导入。
3. 在右侧设置阈值、合并间隔、最短片段时长、关键帧导出选项和结果目录，然后点击“开始推理”。
4. 选择已经完成的任务：蓝线是逐帧概率，橙线是模型边界。左键空白处新增边界、拖动边界调整、右键删除；修改后的边界标记为人工边界。
5. 点击“导出当前”或“导出全部”。

模型严格按 30 FPS 训练。非 30 FPS 视频会写入结果目录中的 `working_30fps.mp4`，用于推理、预览和精确导出；原始视频不会被修改。

默认情况下，结果写入 `/home/descfly/segment_data/`；也可以在界面中指定其他统一导出目录。每个视频具有独立结果目录，包含：

- `inference.json`：自动切分点、模型校验值和本次推理参数；
- `features.npz`、`probabilities.npz`：可复核的特征与逐帧概率；
- `result.json`：最终人工/自动边界、设置、警告和片段清单；
- `segments/`：H.264 MP4 分段视频，保留音频；
- 可选的 `cut_frames/` 与 `job.json`。

## HTTP API

API 与桌面程序相互独立，不会启动 Qt。安装 API 依赖后设置访问密钥并启动：

```bash
./install_api_linux.sh
export DATA_CLIP_API_KEY='请使用足够长的随机密钥'
./run_api.sh
```

默认只监听 `127.0.0.1:18082`。可用 `DATA_CLIP_API_HOST`、`DATA_CLIP_API_PORT`、
`DATA_CLIP_DATASET_ROOT`、`DATA_CLIP_EXPORT_ROOT`、`DATA_CLIP_API_DATABASE` 和
`DATA_CLIP_CORS_ORIGINS` 配置部署；生产环境应由网关或 Nginx 提供 TLS 和限流。
除 `/healthz` 外，请求均需 `X-API-Key`。启动后访问 `/docs` 可查看并调用 OpenAPI 文档。

典型流程是 `POST /v1/jobs` 提交日期任务路径和序号范围，轮询 `GET /v1/jobs/{job_id}`；
推理完成后用 `GET /v1/jobs/{job_id}/items/{sequence}?curve=true` 读取曲线与边界，
如有人工修改则 `PUT .../boundaries`，最后调用 `POST .../export` 并从 `downloads` 端点下载结果。

### 迭代模型版本

当前 API 的活跃模型是根目录 `model/` 的 v1.3。通过 `GET /v1/models` 可查看已注册版本。
训练出新的完整冻结模型包后，在服务器本地执行：

```bash
./video_clip/bin/python register_model_release.py v1.4 /path/to/frozen-model-package
DATA_CLIP_MODEL_VERSION=v1.4 ./run_api.sh
```

注册命令会先验证候选模型的清单、权重和 MediaPipe 资产哈希，再复制到
`model/releases/v1.4/`；同名版本拒绝覆盖。模型切换仅在 API 重启后生效，已创建任务不会在运行中更换模型。

最短片段时长只会自动移除低置信度的模型边界；人工边界永远不会被静默删除。两个人工边界若形成过短片段，软件会在报告中记录警告。
