"""Evaluate an Olive/Mobius Qwen2.5-VL export on an agent run's held-out images."""

import argparse
import io
import json
import os
import re
import time
from importlib.metadata import version
from pathlib import Path
from statistics import mean, median

import onnx
import onnxruntime as ort
import onnxruntime_genai as genai
import numpy as np
from PIL import Image

from vlm_compression_agent.benchmark_qwen25_vl import AutoProcessor, consensus_score, load_grouped_splits
from vlm_compression_agent.controller import MODEL, TASKS, make_sample_manifest, write_json


def apply_image_budget(model_dir, image_size):
    config = json.loads((model_dir / "genai_config.json").read_text(encoding="utf-8"))
    processor_path = model_dir / config["model"]["vision"]["config_filename"]
    content = json.loads(processor_path.read_text(encoding="utf-8"))
    resize = [item["operation"]["attrs"] for item in content["processor"]["transforms"]
              if item["operation"].get("attrs", {}).get("smart_resize")]
    if len(resize) != 1:
        raise ValueError("Expected one Qwen2.5-VL smart resize operation")
    resize[0]["min_pixels"] = image_size["shortest_edge"]
    resize[0]["max_pixels"] = image_size["longest_edge"]
    write_json(processor_path, content)


def evaluate(model, processor, hf_processor, vision_ranks, sample, task, inspect_pixels=False,
             use_hf_pixels=False):
    instruction = "Answer with Yes or No only." if task == "pope" else "Give a short answer only."
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": f"{sample['question']}\n{instruction}"},
    ]}]
    prompt = hf_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image = genai.Images.open_bytes(sample["image"]["bytes"])
    inputs = processor(prompt, images=image)
    for name, rank in vision_ranks.items():
        tensor = inputs[name]
        shape = tensor.shape()
        if len(shape) == rank + 1 and shape[0] == 1:
            inputs[name] = tensor.as_numpy()[0]
        elif len(shape) != rank:
            raise ValueError(f"Vision input {name} has shape {shape}, expected rank {rank}")
    grid = inputs["image_grid_thw"].as_numpy() if "image_grid_thw" in inputs.keys() else None
    visual_tokens = sum(int(t) * int(h) * int(w) // 4 for t, h, w in grid.reshape(-1, 3)) if grid is not None else None
    pixel_parity = {}
    if inspect_pixels or use_hf_pixels:
        with Image.open(io.BytesIO(sample["image"]["bytes"])) as source:
            hf_vision = hf_processor.image_processor(images=[source.convert("RGB")], return_tensors="np")
        hf_pixels = hf_vision["pixel_values"]
        ort_pixels = inputs["pixel_values"].as_numpy()
        pixel_parity["hf_pixel_shape"] = list(hf_pixels.shape)
        pixel_parity["ort_pixel_shape"] = list(ort_pixels.shape)
        if inspect_pixels and hf_pixels.shape == ort_pixels.shape:
            hf_values = hf_pixels.astype(np.float32).ravel()
            ort_values = ort_pixels.astype(np.float32).ravel()
            pixel_parity["pixel_cosine"] = float(np.dot(hf_values, ort_values) /
                                                 (np.linalg.norm(hf_values) * np.linalg.norm(ort_values)))
        if use_hf_pixels:
            if hf_pixels.shape != ort_pixels.shape or not np.array_equal(hf_vision["image_grid_thw"], grid):
                raise ValueError("ORT and Hugging Face vision grid shapes do not agree")
            inputs["pixel_values"] = hf_pixels.astype(ort_pixels.dtype, copy=False)
    params = genai.GeneratorParams(model)
    params.set_search_options(max_length=2048, do_sample=False, batch_size=1)
    start = time.perf_counter()
    generator = genai.Generator(model, params)
    generator.set_inputs(inputs)
    generated = []
    for _ in range(6 if task == "pope" else 16):
        if generator.is_done():
            break
        generator.generate_next_token()
        generated.append(int(generator.get_next_tokens()[0]))
        if len(generated) == 1:
            ttft_ms = (time.perf_counter() - start) * 1000
    if not generated:
        raise RuntimeError("ONNX Runtime generated no tokens")
    prediction = hf_processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
    if task == "pope":
        match = re.match(r"^\s*(yes|no)\b", prediction.lower())
        score = float(bool(match) and match[1] == sample["answer"])
    else:
        score = consensus_score(prediction, sample["answers"])
    row = dict(question_id=sample["question_id"], prediction=prediction, score=score, ttft_ms=ttft_ms,
               generated_tokens=len(generated), visual_tokens=visual_tokens)
    if "category" in sample:
        row["category"] = sample["category"]
    if inspect_pixels:
        row["vision_diagnostic"] = pixel_parity
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Olive export containing genai_config.json")
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fp16-reference", action="store_true", help="Run the original ONNX decoder as a parity check")
    parser.add_argument("--probe-per-task", type=int, help="Inspect preprocessing on this many diagnostic questions")
    parser.add_argument("--match-hf-vision", action="store_true", help="Feed the same image tensors used by PyTorch")
    args = parser.parse_args()
    if args.probe_per_task is not None and args.probe_per_task < 1:
        parser.error("--probe-per-task must be positive")
    previous = json.loads((args.comparison_dir / "comparison.json").read_text(encoding="utf-8"))
    manifest = json.loads((args.comparison_dir / previous["sample_manifest"]).read_text(encoding="utf-8"))
    bf16_model = args.project_dir / "models" / MODEL
    hf_processor = AutoProcessor.from_pretrained(bf16_model, local_files_only=True,
                                                max_pixels=previous["max_image_tokens"] * 28**2)
    apply_image_budget(args.model_dir, hf_processor.image_processor.size)
    genai_config = json.loads((args.model_dir / "genai_config.json").read_text(encoding="utf-8"))
    vision_path = args.model_dir / genai_config["model"]["vision"]["filename"]
    vision_graph = onnx.load(vision_path, load_external_data=False).graph
    vision_ranks = {value.name: len(value.type.tensor_type.shape.dim) for value in vision_graph.input}
    packed_nodes = {}
    for graph in sorted(args.model_dir.rglob("*.onnx")):
        model_graph = onnx.load(graph, load_external_data=False)
        count = sum(node.op_type == "MatMulNBits" for node in model_graph.graph.node)
        packed_nodes[str(graph.relative_to(args.model_dir))] = count
    cuda_dll_dir = os.environ.get("ORT_CUDA_DLL_DIR")
    if cuda_dll_dir:
        ort.preload_dlls(cuda=True, cudnn=False, directory=cuda_dll_dir)
        ort.preload_dlls(cuda=False, cudnn=True)
    else:
        ort.preload_dlls()
    config = genai.Config(str(args.model_dir))
    source_config = json.loads((bf16_model / "config.json").read_text(encoding="utf-8"))
    runtime_settings = {"vision_start_token_id": source_config["vision_start_token_id"]}
    if args.fp16_reference:
        reference_config = json.loads((args.model_dir / "genai_config.fp16.json").read_text(encoding="utf-8"))
        runtime_settings["decoder"] = reference_config["model"]["decoder"]
    config.overlay(json.dumps({"model": runtime_settings}))
    model = genai.Model(config)
    if model.device_type.lower() != "cuda":
        raise RuntimeError(f"Expected CUDA, got {model.device_type}")
    processor = model.create_multimodal_processor()
    print(f"CUDA model {model.type}, packed INT4 nodes: {packed_nodes}", flush=True)
    report = dict(
        olive=version("olive-ai"), onnxruntime=ort.__version__, genai=genai.__version__, seed=previous["seed"],
        device=model.device_type, model_dir=str(args.model_dir), comparison_dir=str(args.comparison_dir),
        decoder_precision="fp16" if args.fp16_reference else "rtn_int4",
        vision_preprocessor="huggingface" if args.match_hf_vision else "ort_genai",
        diagnostic=args.probe_per_task is not None,
        max_image_tokens=previous["max_image_tokens"], matmul_nbits_nodes=packed_nodes,
        ttft_scope="warm generation including generator creation and ONNX Runtime input binding",
        memory_scope="PyTorch allocated bytes are unavailable for this runtime",
        tasks={},
    )
    for task in TASKS:
        splits, files, image_key = load_grouped_splits(task, previous["search_samples_per_task"],
                                                     previous["validation_samples_per_task"],
                                                     previous["seed"], args.project_dir / "datasets")
        if make_sample_manifest(splits, image_key, files) != manifest[task]:
            raise ValueError(f"Samples no longer match the paired PyTorch run for {task}")
        samples = splits["validation"]
        if args.probe_per_task is not None:
            samples = samples[:args.probe_per_task]
        evaluate(model, processor, hf_processor, vision_ranks, samples[0], task,
                 use_hf_pixels=args.match_hf_vision)
        rows = []
        for index, sample in enumerate(samples, start=1):
            rows.append(evaluate(model, processor, hf_processor, vision_ranks, sample, task,
                                 inspect_pixels=args.probe_per_task is not None,
                                 use_hf_pixels=args.match_hf_vision))
            if index % 6 == 0 or index == len(samples):
                print(f"{task}: {index}/{len(samples)}", flush=True)
        report["tasks"][task] = dict(rows=rows, score=mean(row["score"] for row in rows),
                                     median_ttft_ms=median(row["ttft_ms"] for row in rows))
        write_json(args.output, report)


if __name__ == "__main__":
    main()
