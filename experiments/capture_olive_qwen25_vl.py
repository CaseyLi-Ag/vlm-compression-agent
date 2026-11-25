"""Run Olive's multimodal capture with CUDA and CPU providers on Windows."""

import argparse
import os
import shutil
import sys
from pathlib import Path

import onnxruntime as ort
from mobius import build
from mobius.integrations.ort_genai import auto_export
from olive.cli.launcher import main


def route_local_tokenizer_files():
    copy_from_hub = auto_export._copy_tokenizer_files
    copy_local = auto_export._copy_tokenizer_files_from_local

    def copy_tokenizer(model_id, output_dir):
        if Path(model_id).is_dir():
            return copy_local(model_id, output_dir)
        return copy_from_hub(model_id, output_dir)

    auto_export._copy_tokenizer_files = copy_tokenizer


def recover_graphs():
    parser = argparse.ArgumentParser(description="Complete an Olive export that failed while copying local files")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("saved_graphs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(sys.argv[2:])
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("decoder", "embedding", "vision_encoder"):
        (args.output / name).mkdir()
        for filename in ("model.onnx", "model.onnx.data"):
            shutil.copy2(args.saved_graphs / name / filename, args.output / name / filename)
    package = build(str(args.checkpoint), dtype="f16", execution_provider="cuda", load_weights=False)
    files = auto_export.write_ort_genai_config(package, str(args.output),
                                               hf_model_id=str(args.checkpoint), ep="cuda")
    print(f"Recovered Olive model at {args.output}, runtime files: {', '.join(files)}", flush=True)


if __name__ == "__main__":
    route_local_tokenizer_files()
    if sys.argv[1:2] == ["recover"]:
        recover_graphs()
        raise SystemExit(0)
    cuda_dll_dir = os.environ.get("ORT_CUDA_DLL_DIR")
    if cuda_dll_dir:
        ort.preload_dlls(cuda=True, cudnn=False, directory=cuda_dll_dir)
        ort.preload_dlls(cuda=False, cudnn=True)
    else:
        ort.preload_dlls()
    if sys.platform == "win32":
        # Olive 0.13 registers every installed provider during startup. For a
        # CUDA capture avoid loading the unrelated TensorRT DLL and SDK.
        providers = [name for name in ort.get_available_providers()
                     if name in {"CUDAExecutionProvider", "CPUExecutionProvider"}]
        ort.get_available_providers = lambda: providers.copy()
        print(f"Olive capture providers: {', '.join(providers)}", flush=True)
    sys.argv.insert(1, "capture-onnx-graph")
    main()
