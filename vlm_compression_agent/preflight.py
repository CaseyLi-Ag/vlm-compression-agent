"""Check local inputs and the CUDA runtime before an experiment."""

import io
import json
import os
import shutil
from contextlib import redirect_stdout
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version


GPU_PACKAGES = ("torch", "torchvision", "transformers", "accelerate", "GPTQModel",
                "angelslim", "pyarrow", "Pillow", "PyYAML")


def input_issues(args):
    issues = []
    for model, quantized in ((args.bf16_model, False), (args.awq_model, True)):
        try:
            config = json.loads((model / "config.json").read_text(encoding="utf-8"))
            if config.get("model_type") != "qwen2_5_vl":
                issues.append(f"{model} is not a Qwen2.5-VL checkpoint")
            quantization = config.get("quantization_config") or {}
            if (quantized and (quantization.get("quant_method"), quantization.get("bits")) != ("awq", 4)
                    or not quantized and quantization):
                issues.append(f"{model} must contain {'INT4 AWQ' if quantized else 'unquantized'} weights")
            index = model / "model.safetensors.index.json"
            shards = (set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
                      if index.is_file() else {"model.safetensors"})
            missing = sorted(shard for shard in shards if not (model / shard).is_file())
            if missing:
                issues.append(f"{model} is missing weight files: {', '.join(missing)}")
        except (OSError, ValueError, KeyError, AttributeError, TypeError) as error:
            issues.append(f"Cannot inspect checkpoint {model}: {error}")

    for task, split in (("pope", "test"), ("textvqa", "validation")):
        folder = args.datasets_dir / task
        if not next(folder.rglob(f"{split}*.parquet"), None):
            issues.append(f"{folder} needs {split}*.parquet files")
    return issues


def doctor(args):
    issues = input_issues(args)
    versions = {}
    for package in GPU_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            issues.append(f"Python package missing: {package}")

    if "triton" in args.awq_backend:
        triton_package = "triton-windows" if os.name == "nt" else "triton"
        try:
            versions[triton_package] = version(triton_package)
        except PackageNotFoundError:
            issues.append(f"Python package missing for {args.awq_backend}: {triton_package}")
        if os.name == "nt" and not shutil.which(os.environ.get("CC", "cl.exe")):
            issues.append("Triton needs a C++ compiler. Run this command in x64 Visual Studio Developer PowerShell.")

    device = None
    memory = None
    if "torch" in versions:
        try:
            import torch

            if not torch.version.cuda:
                issues.append("The installed PyTorch build does not support NVIDIA CUDA")
            elif not 0 <= args.cuda_device < torch.cuda.device_count():
                issues.append(f"CUDA device {args.cuda_device} is unavailable among the visible devices")
            else:
                with torch.cuda.device(args.cuda_device):
                    device = torch.cuda.get_device_name()
                    free, total = torch.cuda.mem_get_info()
                    memory = dict(free_mib=free // 1024**2, total_mib=total // 1024**2)
                    if not torch.cuda.is_bf16_supported():
                        issues.append(f"{device} does not support BF16")
        except (ImportError, OSError, RuntimeError) as error:
            issues.append(f"Could not initialize CUDA: {error}")

    if device and not issues:
        imports = [
            ("transformers", "AutoModelForMultimodalLM"),
            ("torchvision", "transforms"),
            ("angelslim.compressor.token_compressor.adapter", "UniversalPruningAdapter"),
            ("angelslim.compressor.token_compressor.base.config", "TokenCompressorConfig"),
        ]
        if "triton" in args.awq_backend:
            imports.append(("triton", "jit"))
        for module, api in imports:
            try:
                with redirect_stdout(io.StringIO()):
                    getattr(import_module(module), api)
            except (ImportError, OSError, RuntimeError, AttributeError) as error:
                issues.append(f"Cannot load {module}.{api}: {error}")

    return dict(status="ready" if not issues else "needs_setup", cuda_device=device, cuda_memory=memory,
                python_packages=versions, issues=issues)
