# VLM Compression Agent

面向 Qwen2.5-VL-3B 的本地压缩配置搜索工具。我用显存上限、每项任务允许下降的分数和评测轮数约束搜索，根据 GPU 实测结果选择 INT4 AWQ 与 VisionZip 的组合。

## 搜索与验证

1. 在 POPE 和 TextVQA 上评测 BF16 与 INT4 AWQ，记录任务得分、峰值显存和首 token 延迟（TTFT）。
2. 根据候选是否达标、TTFT 是否改善，调整 VisionZip 的视觉 token 剪枝比例。显存和得分约束分别在每项任务上检查。
3. 在一批与搜索集无共同图片的样本上重新评测 BF16 和入围候选，推荐满足约束且 TTFT 最低的配置。若没有候选通过复测，报告记录具体超限项。

## 安装

CLI 要求 Python 3.10 或更高版本。在仓库根目录执行以下命令。

```bash
python -m pip install .
vlm-compression-agent demo
```

`demo` 使用模拟数据验证搜索流程，可在 CPU 上离线运行，输出标记为 `"simulated": true`。已有 wheel 时，可以通过 `python -m pip install --no-index vlm_compression_agent-0.1.0-py3-none-any.whl` 安装。

## GPU 环境

实测环境为 Windows 11、Python 3.12 和 RTX 4080 Laptop GPU。PyTorch、Transformers 和量化后端的实测版本见下方安装命令。Linux 和 A100 尚待验证。

在 Windows 的 Python 3.12 虚拟环境中安装依赖。先将支持 Qwen2.5-VL VisionZip 的 AngelSlim 源码放在 `AngelSlim-main/` 下。CUDA 依赖需与显卡驱动及运行平台兼容。安装参考见 [PyTorch](https://pytorch.org/get-started/locally/)、[GPTQModel](https://github.com/ModelCloud/GPTQModel/wiki/Installation-and-Quickstart) 和 [AngelSlim](https://github.com/Tencent/AngelSlim)。

```powershell
python -m pip install "setuptools>=77,<83" wheel
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install transformers==5.6.2 GPTQModel==7.0.0 triton-windows==3.9.0.post30 accelerate pyarrow Pillow PyYAML
python -m pip install --no-build-isolation -e .\AngelSlim-main
```

在带 x64 C++ 工具链的 Developer PowerShell 中运行 GPU 评测，并执行 `$env:CC = (Get-Command cl.exe).Source` 为 Triton 配置编译器。

### 模型与数据

评测读取本地模型和 Parquet 文件。两份模型权重合计约 10.9 GB，评测数据分片约 1.2 GB。默认目录如下。

- `models/Qwen2.5-VL-3B-Instruct/` 存放 BF16 权重。[Qwen BF16 checkpoint](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- `models/Qwen2.5-VL-3B-Instruct-AWQ/` 存放 INT4 AWQ 权重。[Qwen INT4 AWQ checkpoint](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct-AWQ)
- `datasets/pope/` 存放 POPE 测试集的 `test*.parquet`。[POPE 数据集](https://huggingface.co/datasets/lmms-lab/POPE)
- `datasets/textvqa/` 存放 TextVQA 验证集的 `validation*.parquet`。[TextVQA 数据集](https://huggingface.co/datasets/lmms-lab/textvqa)

以下 Hugging Face CLI 命令下载评测使用的两个数据划分。

```bash
hf download lmms-lab/POPE --repo-type dataset --include "data/test*.parquet" --local-dir datasets/pope
hf download lmms-lab/textvqa --repo-type dataset --include "data/validation*.parquet" --local-dir datasets/textvqa
```

## 运行评测

从包含以上模型和数据的目录运行。可以用 `--project-dir` 指定根目录，或用 `--bf16-model`、`--awq-model` 和 `--datasets-dir` 分别指定路径。

```bash
vlm-compression-agent doctor
vlm-compression-agent run --max-vram-gib 4 --max-score-drop-pp 2 --max-rounds 8
vlm-compression-agent compare --seed 41 --search-samples 24 --validation-samples 24 --max-rounds 7
```

- `doctor` 检查模型、数据和 GPU 依赖。
- `run` 执行约束搜索。上例要求每项任务的峰值显存低于 4 GiB、相对 BF16 最多下降 2 个百分点，并在 8 轮内完成搜索和复测。
- `compare` 在相同图片、模型和推理内核下比较自适应搜索、固定网格、纯 AWQ 和 AWQ 加 25% VisionZip。自适应搜索和固定网格使用相同的轮数上限，各自最多复测两个候选。各策略共享相同配置与数据划分的实测记录，并独立计算评测轮数。

评测在单张 GPU 上顺序执行。`--cuda-device 0` 指向 `CUDA_VISIBLE_DEVICES` 中的第一张卡。4 GiB 是候选配置的验收上限，运行实验还需加载 BF16 基线。本机测得 BF16 的 PyTorch 峰值已分配显存约为 7.18 GiB，实际运行还需为 CUDA 运行时和其他进程预留空间。

### 输出文件与指标

输出默认保存在 `results/compression_agent/` 下，也可用 `--output-dir` 指定。

- `report.json` 记录各轮配置、选择理由、指标和超限项。通过复测的推荐配置位于 `recommendation` 字段。
- `comparison.json` 记录各策略的结论、轮数和共享测量。`measured_seconds` 为该策略所用实测轮次的耗时之和。
- `samples.json` 记录搜索集与复测集的题目和图片 ID。`measurements/` 保存逐题结果。

显存按 `torch.cuda.max_memory_allocated` 统计。TTFT 从 GPU 输入就绪开始计时，候选排序依据 POPE 和 TextVQA 各自 TTFT 中位数的平均值。

## 实验结果

两组实验都采用 4 GiB 显存上限和 2 个百分点的掉分上限。seed 41 选择了 AWQ 加 12.5% VisionZip，复测峰值显存为 3.77 GiB，两项得分与 BF16 持平。两项任务的 TTFT 中位数平均值为 383.5 ms，纯 AWQ 为 398.4 ms。seed 62 最终选择纯 AWQ，峰值显存为 3.35 GiB。[对照实验、Olive 实测与复现命令](docs/comparison.md)

## 开发

```bash
python -m unittest discover -s experiments -v
python -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
```

VisionZip 配置来自 Tencent/AngelSlim，许可文本随 wheel 一并分发。[AngelSlim 许可](vlm_compression_agent/AngelSlim-LICENSE.txt)
