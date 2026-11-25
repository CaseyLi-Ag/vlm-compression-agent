# VLM Compression Agent

给定显存上限、每项任务可接受的掉分和评测轮数，CLI 在本地 GPU 上为 Qwen2.5-VL 搜索 INT4 AWQ 与 VisionZip 的组合。它记录每轮为什么选这个配置，并用另一批图片复测入围配置。达到预算后，报告返回通过复测的推荐配置或具体超限项。

## 两分钟试用

使用 Python 3.10 或以上版本。在仓库根目录执行。

```text
python -m pip install .
vlm-compression-agent demo
```

`demo` 使用模拟测量，输出带有 `"simulated": true`。它可以在 CPU 上离线执行。已有 wheel 时，可用 `python -m pip install --no-index vlm_compression_agent-0.1.0-py3-none-any.whl` 安装。

两组 GPU 实测、自适应搜索与固定策略的取舍，以及同机 Olive 对照见[实验报告](docs/comparison.md)。

## 在 GPU 上运行

以下是已实测的组合。Python 3.12、Windows 11、RTX 4080 Laptop、PyTorch 2.10.0+cu128、TorchVision 0.25.0+cu128、Transformers 5.6.2、GPTQModel 7.0.0、Triton Windows 3.9.0.post30，以及支持 Qwen2.5-VL VisionZip 的 AngelSlim 源码。环境还需要 Accelerate、PyArrow、Pillow 和 PyYAML。CUDA 依赖以显卡驱动和平台为准，参考 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)、[GPTQModel 安装说明](https://github.com/ModelCloud/GPTQModel/wiki/Installation-and-Quickstart) 和 [AngelSlim](https://github.com/Tencent/AngelSlim)。Windows 上请从带 x64 C++ 工具链的 Developer PowerShell 运行 Triton，并使 `CC` 指向 `cl.exe`。

在 Windows 的 Python 3.12 虚拟环境中，可以按以下顺序准备 GPU 依赖。最后一条命令指向含有 Qwen2.5-VL VisionZip 配置的本地 AngelSlim 源码目录。

```powershell
python -m pip install "setuptools>=77,<83" wheel
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install transformers==5.6.2 GPTQModel==7.0.0 triton-windows==3.9.0.post30 accelerate pyarrow Pillow PyYAML
python -m pip install --no-build-isolation -e .\AngelSlim-main
```

准备本地目录。两个权重合计约 10.9 GB，指定的数据分片约 1.2 GB。程序只读取本地路径。

- `models/Qwen2.5-VL-3B-Instruct/` 来自 [Qwen 原始 checkpoint](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)。
- `models/Qwen2.5-VL-3B-Instruct-AWQ/` 来自 [Qwen INT4 AWQ checkpoint](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-AWQ)。
- `datasets/pope/` 放 [POPE](https://huggingface.co/datasets/lmms-lab/POPE) 的 `test*.parquet`。
- `datasets/textvqa/` 放 [TextVQA](https://huggingface.co/datasets/lmms-lab/textvqa) 的 `validation*.parquet`。

使用 Hugging Face CLI 准备评测数据时，可以指定分片，只下载所用划分。

```text
hf download lmms-lab/POPE --repo-type dataset --include "data/test*.parquet" --local-dir datasets/pope
hf download lmms-lab/textvqa --repo-type dataset --include "data/validation*.parquet" --local-dir datasets/textvqa
```

从包含上述目录的工作目录运行。也可添加 `--project-dir` 指向数据目录，或通过 `--bf16-model`、`--awq-model` 和 `--datasets-dir` 分别覆盖。Windows 下可以在当前 Developer PowerShell 会话执行 `$env:CC = (Get-Command cl.exe).Source`。

```text
vlm-compression-agent doctor
vlm-compression-agent run --max-vram-gib 4 --max-score-drop-pp 2 --max-rounds 8
vlm-compression-agent compare --seed 41 --search-samples 24 --validation-samples 24 --max-rounds 7
```

在共享服务器上可由调度器分配一张卡，并保留它设置的 `CUDA_VISIBLE_DEVICES`。`--cuda-device 0` 使用可见列表里的第一张卡。每次只评测一组配置，基线和候选按顺序运行。实验需要有空间加载 BF16 参考模型，本次基线使用约 7.18 GiB 的 PyTorch 已分配峰值显存。4 GiB 用来验收候选，此外运行时和显卡上的其他进程也会占用显存。Linux/A100 尚待实测。

## 怎么看结果

- `run` 生成 `report.json`。查看 `recommendation`，并用 `rounds` 核对候选、显存、各任务与 BF16 的得分差和决策原因。程序输出配置组合，安装好的模型和 VisionZip 在推理时应用该组合。
- `compare` 生成 `comparison.json`。其中自适应搜索和固定网格使用同一轮数上限，各自最多复测两个候选。纯 AWQ 和固定 25% VisionZip 是更低成本的预设。所有策略共享相同模型、图片、内核。完全一致的配置与数据划分只实测一次。每个策略仍按独立使用的轮数计费，`measured_seconds` 由所使用的底层实测轮次耗时求和。
- `samples.json` 记录搜索与复测样本和图片 ID。`measurements/` 保存逐题结果。显存统计使用 `torch.cuda.max_memory_allocated`，首 token 延迟从 GPU 输入就绪开始计时，排序使用 POPE 和 TextVQA 各自 TTFT 中位数的平均值。

## 开发

```text
python -m unittest discover -s experiments -v
python -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
```

VisionZip 配置来自 Tencent/AngelSlim。其许可文本随 wheel 一起分发。
