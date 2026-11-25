# VLM 压缩 Agent 对照实验

## 评测方法

模型为官方 Qwen2.5-VL-3B-Instruct BF16 与 INT4 AWQ。四种策略共用同一台 RTX 4080 Laptop GPU、Triton GEMM AWQ 后端、POPE 与 TextVQA 样本、生成参数和指标。显存目标是每项任务的 PyTorch 峰值已分配显存低于 4 GiB，准确性目标是每项任务相对对应的 BF16 基线最多下降 2 个百分点。

- **自适应搜索**从剪去 25% 的视觉 token 开始。候选满足约束且延迟下降时试更高的比例，其余情况降低比例。
- **固定网格**按预先确定的序列探测。7 轮预算下包含 BF16、纯 AWQ、剪去 25% 和 50% 的 AWQ，最多使用 3 轮独立复测。
- **固定 AWQ**与**固定 VisionZip**各自用固定配置做搜索集评估和独立复测。VisionZip 预设使用 AWQ 并剪去 25% 的视觉 token，控制实现和后端保持一致。该预设使用 AngelSlim 的算法，在配对样本上重跑，比例由本实验设定。[AngelSlim Qwen2.5-VL 仓库及预设命令](https://github.com/Tencent/AngelSlim#3-token-compression-vlm)

两次运行分别用 seed 41 与 62 生成每任务 24 道搜索题和 24 道复测题。同一次运行的搜索与复测图片无重叠。跨 seed 重新抽样，仍来自相同的数据集。复测结果用于最终候选排序，如需测部署泛化效果，需另留一批最终测试图片。报告保存每道题的预测、得分、首 token 延迟、峰值显存和视觉 token 保留量。首 token 延迟以两项任务各自的中位数求平均。程序在同一轮运行中共享相同配置与数据划分的底层测量。

```text
vlm-compression-agent compare --seed 41 --search-samples 24 --validation-samples 24 --max-rounds 7 --output-dir results/agent_comparison/seed-41
vlm-compression-agent compare --seed 62 --search-samples 24 --validation-samples 24 --max-rounds 7 --output-dir results/agent_comparison/seed-62
python experiments/audit_agent_comparison.py results/agent_comparison/seed-41 results/agent_comparison/seed-62
```

## 本机实测

两次运行共做了 17 轮不重复的 GPU 测量，每轮各评测 24 道 POPE 和 24 道 TextVQA。审核脚本从逐题记录重新计算得分、峰值显存、TTFT 和推荐结果，并验证图片划分与各策略预算。每个策略的耗时按其独立运行所需的测量计算，包含加载模型和评测。

- **Seed 41** 的自适应配置为 AWQ 加 12.5% VisionZip，复测显存 **3.77 GiB**、POPE 与 TextVQA 相对 BF16 均为 **0.00** 个百分点。TTFT 为 **383.5 ms**，固定网格和纯 AWQ 的通过配置均为 **398.4 ms、3.35 GiB**。四个策略分别用了 7、6、4、4 轮，自适应用时 12.35 分钟，固定网格 10.00 分钟，纯 AWQ 6.92 分钟。
- **Seed 62** 的自适应策略从 25% 提高到 37.5% 探索，复测时 25% 的候选在 TextVQA 上掉 **7.92** 个百分点。自适应、网格和固定 AWQ 均通过纯 AWQ，**3.35 GiB**、**671.5 ms**，POPE 持平，TextVQA 高 **0.42** 个百分点。三者分别用 7、7、4 轮。
- **固定 25% VisionZip** 在两组 TextVQA 复测分别掉 **4.17**、**7.92** 个百分点。此前每任务每划分 120 题的实验中，纯 AWQ 和 25% VisionZip 也都超出该任务的 2 点限制。固定 AWQ 适合作为低成本首选基线，用户可以用自己的数据复测其准确性。

这两次对照支持两个观察。搜索策略会随样本改变，独立复测对筛出不合格候选有实际作用。自适应搜索多花了评测轮数，只有一组得到较低的 TTFT，且显存更高。每组样本较少，单题就能改变 POPE 得分 4.17 个百分点，单次测得的延迟差也可能受到 GPU 运行状态影响。这些结果适合验证流程和展示取舍，普遍准确率和加速效果还需要更大的测试集与重复计时。

关联记录见[联合审核汇总](results/search-comparison.json)。复现命令会在本地 `results/agent_comparison/` 中生成各轮和逐题记录。

## 相关工具的公开功能

- **NVIDIA ModelOpt AutoQuantize**按层搜索混合精度，使用有效位宽目标与量化敏感度估计，提供量化 checkpoint 的导出和多运行时部署路径。Qwen2.5-VL 的公开支持矩阵列出了 INT4 AWQ。若做实测，可将导出的模型接到相同的 POPE/TextVQA 评测，并记录校准、导出与运行时的成本。[AutoQuantize](https://nvidia.github.io/Model-Optimizer/announcements/20260824-autoquantize.html) · [HF PTQ 示例及支持矩阵](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/hf_ptq/README.md)
- **Microsoft Olive**面向 ONNX Runtime，提供硬件目标、指标目标、Sequential/Random/TPE 搜索和自定义 evaluator。ONNX Runtime GenAI 的 Model Builder 支持 Qwen2.5-VL。针对该 VLM 的实测需要导出 ONNX 图，并给图片和文本接上相同评测。[Olive 选项](https://microsoft.github.io/Olive/reference/options.html) · [评测配置](https://microsoft.github.io/Olive/how-to/configure-workflows/metrics-configuration.html) · [ONNX Runtime GenAI 发行记录](https://github.com/microsoft/onnxruntime-genai/releases)
- **LightCompress / LLMC+**提供视觉语言模型的压缩与评测，覆盖语言模块、视觉编码器及多种量化和稀疏化方法。可以用它生成多种压缩候选，再在统一的运行时和任务样本上验收。[LightCompress 代码](https://github.com/ModelTC/LightCompress) · [LLMC+ 论文](https://arxiv.org/abs/2508.09981)

ModelOpt 和 LightCompress 使用公开资料做功能对照。外部论文和官方表格的分数使用不同协议，本报告的数值来自本机保存的逐题测量。

## Olive 的同机部署对照

实测使用 Olive 0.13.0、Mobius 0.1.0、ONNX Runtime GPU 1.32.0 与 GenAI CUDA 0.20.0。`olive optimize` 的该版本仅接受文本模型。本次使用支持多模态的 Olive/Mobius 导出路径，从同一官方 BF16 权重生成 FP16 图，并用 Olive 的 `OnnxBlockWiseRtnQuantization` 处理解码器。最终图包含 253 个 INT4 MatMulNBits 节点，量化块大小 128。视觉编码器与嵌入仍用 FP16。已有 AWQ 文件直接转图时因量化 scale 的布局不匹配而失败。原始图的本地 tokenizer 打包问题用原 checkpoint 的文件补齐。[Olive 导出文档](https://microsoft.github.io/Olive/reference/cli.html#capture-onnx-graph) · [Mobius 支持的模型](https://github.com/onnxruntime/mobius)

下面命令在隔离的 Olive 环境中执行，CUDA 13 DLL 和 cuDNN 需能被找到。Windows 上可将 `ORT_CUDA_DLL_DIR` 指向 CUDA Toolkit 中包含 `cublas64_13.dll` 的目录。

```text
python -m experiments.capture_olive_qwen25_vl -m models/Qwen2.5-VL-3B-Instruct -t image-text-to-text -o results/olive-output --use_mobius_builder --precision fp16 --disable_telemetry
python -m experiments.quantize_olive_decoder <包含 genai_config.json 的导出目录>
python -m experiments.benchmark_olive_qwen25_vl --model-dir <导出目录> --comparison-dir results/agent_comparison/seed-41 --output results/olive-seed-41.json --match-hf-vision
```

评测复用相同图片、问题、评分函数、视觉 token 上限和生成长度。诊断中发现 ONNX 原生图像预处理产生的张量与同一张图的 Hugging Face 张量差异很大，尽管每张图的视觉 token 数相同。12 题 FP16 ONNX 抽检中，改用相同的 Hugging Face 图像张量后，POPE 从 83.33% 恢复到 100%，TextVQA 从 0 恢复到 83.33%，与 PyTorch BF16 的同题得分一致。正式对照因此在两套运行时中使用统一的图像张量。

ONNX Runtime 的首 token 时间包含创建生成器和绑定输入，PyTorch 从 GPU 输入就绪开始计时。此处只比较两套实际部署方案的观察结果。PyTorch 的已分配显存指标无法直接用于 ONNX Runtime，故没有跨运行时显存排名。

- **Seed 41** 的 Olive INT4 RTN 在统一输入下测得 POPE **62.50%**、TextVQA **18.75%**，分别比 PyTorch BF16 低 **33.33**、**58.75** 个百分点，两任务首 token 延迟中位数的平均值为 **243.4 ms**。同题的 Agent 推荐配置为 AWQ 加 12.5% VisionZip，两项得分为 **95.83%**、**77.50%**，首 token 延迟为 **383.5 ms**。
- **Seed 62** 的 Olive INT4 RTN 得分为 **54.17%**、**12.50%**，分别比 PyTorch BF16 低 **41.67**、**74.17** 个百分点，首 token 延迟为 **262.6 ms**。同题的 Agent 推荐配置为纯 AWQ，两项得分为 **95.83%**、**87.08%**，首 token 延迟为 **671.5 ms**。

本次 Olive 配置的首 token 先到，两组的掉分都远超 2 点上限。结果覆盖这条固定 INT4 RTN 部署路径和本机环境。[Olive 审核汇总](results/olive-comparison.json)
