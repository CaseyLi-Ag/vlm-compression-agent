"""Benchmark a BF16 or INT4 AWQ Qwen2.5-VL checkpoint with and without VisionZip."""

import argparse
import io
import json
import random
import re
import statistics
import time
from importlib.resources import files
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import pyarrow.parquet as parquet
import torch
import yaml
from PIL import Image
from transformers import AutoConfig, AutoModelForMultimodalLM, AutoProcessor

from angelslim.compressor.token_compressor.adapter import UniversalPruningAdapter
from angelslim.compressor.token_compressor.base.config import TokenCompressorConfig


VISIONZIP = files("vlm_compression_agent").joinpath("visionzip.yaml")
NUMBER_WORDS = dict(zip("zero one two three four five six seven eight nine ten".split(), map(str, range(11))))


def normalize_vqa(answer):
    answer = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", answer.lower().replace("’", "'"))
    answer = re.sub(r"(?<=\d),(?=\d)", "", answer)
    answer = re.sub(r"[^a-z0-9.'\s]", " ", answer)
    return " ".join(NUMBER_WORDS.get(word, word) for word in answer.split() if word not in {"a", "an", "the"})


def consensus_score(prediction, answers):
    prediction = normalize_vqa(prediction)
    matches = [normalize_vqa(answer) == prediction for answer in answers]
    total = sum(matches)
    return statistics.mean(min((total - match) / 3, 1) for match in matches)


def read_sample_table(dataset, include_image_id=False, datasets_dir=None):
    datasets_dir = datasets_dir or Path.cwd() / "datasets"
    split = "test" if dataset == "pope" else "validation"
    files = sorted((datasets_dir / dataset).rglob(f"{split}*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {split} parquet files for {dataset}")
    columns = ["question_id", "question", "image"]
    columns += ["answer", "category"] if dataset == "pope" else ["answers"]
    if include_image_id:
        columns.append("image_source" if dataset == "pope" else "image_id")
    return parquet.read_table(files, columns=columns), [str(path.relative_to(datasets_dir.parent)) for path in files]


def sample_rows(data, dataset, limit, rng, pool):
    if dataset == "pope" and limit >= 6:
        groups = defaultdict(list)
        categories = data["category"].to_pylist()
        answers = data["answer"].to_pylist()
        for index in pool:
            groups[categories[index], answers[index]].append(index)
        if len(groups) != 6:
            raise ValueError(f"Expected six POPE category/label groups, found {list(groups)}")
        base, extra = divmod(limit, len(groups))
        indices = [index for group_index, key in enumerate(sorted(groups))
                   for index in rng.sample(groups[key], base + (group_index < extra))]
        rng.shuffle(indices)
    else:
        indices = rng.sample(pool, limit)
    return data.take(indices).to_pylist()


def load_samples(dataset, limit, seed):
    data, files = read_sample_table(dataset)
    return sample_rows(data, dataset, limit, random.Random(seed), range(len(data))), files


def load_grouped_splits(dataset, search_limit, validation_limit, seed, datasets_dir=None):
    data, files = read_sample_table(dataset, include_image_id=True, datasets_dir=datasets_dir)
    image_key = "image_source" if dataset == "pope" else "image_id"
    image_ids = data[image_key].to_pylist()
    if any(image_id is None for image_id in image_ids):
        raise ValueError(f"Cannot split {dataset} by image because an image ID is missing")
    rng = random.Random(f"{dataset}:{seed}")
    images = sorted(set(image_ids))
    rng.shuffle(images)
    search_images = set(images[:len(images) // 2])
    pools = {"search": [], "validation": []}
    for index, image_id in enumerate(image_ids):
        pools["search" if image_id in search_images else "validation"].append(index)
    return {
        split: sample_rows(data, dataset, limit, rng, pools[split])
        for split, limit in (("search", search_limit), ("validation", validation_limit))
    }, files, image_key


class FirstTokenClock:
    def __init__(self):
        self.prompt_seen = False
        self.first_token_time = None

    def put(self, _token):
        if not self.prompt_seen:
            self.prompt_seen = True
        elif self.first_token_time is None:
            torch.cuda.synchronize()
            self.first_token_time = time.perf_counter()

    def end(self):
        pass


def measure(model, processor, sample, dataset, max_new_tokens):
    prompt = "Answer with Yes or No only." if dataset == "pope" else "Give a short answer only."
    with Image.open(io.BytesIO(sample["image"]["bytes"])) as image:
        messages = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": f"{sample['question']}\n{prompt}"},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image.convert("RGB")], return_tensors="pt").to("cuda")

    prompt_tokens = inputs.input_ids.shape[-1]
    original_visual_tokens = int((inputs.input_ids == model.config.image_token_id).sum().item())
    decoder_lengths = []

    def capture_decoder_length(_module, args, kwargs):
        if not decoder_lengths:
            hidden_states = args[0] if args else kwargs["hidden_states"]
            decoder_lengths.append(hidden_states.shape[-2])

    hook = model.model.language_model.layers[0].register_forward_pre_hook(capture_decoder_length, with_kwargs=True)
    clock = FirstTokenClock()
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        # AngelSlim prints intermediate shapes to stdout on every request.
        with torch.inference_mode(), redirect_stdout(io.StringIO()):
            output_ids = model.generate(**inputs, do_sample=False, use_cache=True,
                                        max_new_tokens=max_new_tokens, streamer=clock)
        torch.cuda.synchronize()
        end = time.perf_counter()
        peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
    finally:
        hook.remove()

    if clock.first_token_time is None or not decoder_lengths:
        raise RuntimeError("Generation did not report a first token and decoder prefill length")
    retained = original_visual_tokens + decoder_lengths[0] - prompt_tokens
    if not 0 < retained <= original_visual_tokens:
        raise ValueError(f"Invalid visual token count {retained}/{original_visual_tokens}")
    suffix = output_ids[:, prompt_tokens:]
    prediction = processor.batch_decode(suffix, skip_special_tokens=True)[0].strip()
    row = dict(question_id=sample["question_id"], question=sample["question"], prediction=prediction,
               original_visual_tokens=original_visual_tokens, retained_visual_tokens=retained,
               ttft_ms=1000 * (clock.first_token_time - start), generate_ms=1000 * (end - start),
               peak_allocated_mib=peak_mib, generated_tokens=suffix.shape[-1])
    if dataset == "pope":
        match = re.match(r"^\s*(yes|no)\b", prediction.lower())
        label = match[1] if match else None
        row.update(answer=sample["answer"], category=sample["category"], label=label,
                   score=float(label == sample["answer"]))
    else:
        row.update(answers=sample["answers"], score=consensus_score(prediction, sample["answers"]))
    return row


def summarize(rows, dataset):
    summary = dict(samples=len(rows), score=statistics.mean(row["score"] for row in rows),
                   median_ttft_ms=statistics.median(row["ttft_ms"] for row in rows),
                   median_generate_ms=statistics.median(row["generate_ms"] for row in rows),
                   peak_allocated_mib=max(row["peak_allocated_mib"] for row in rows),
                   mean_generated_tokens=statistics.mean(row["generated_tokens"] for row in rows),
                   visual_retention=sum(row["retained_visual_tokens"] for row in rows)
                   / sum(row["original_visual_tokens"] for row in rows))
    if dataset == "pope":
        tp = sum(row["label"] == row["answer"] == "yes" for row in rows)
        predicted_yes = sum(row["label"] == "yes" for row in rows)
        actual_yes = sum(row["answer"] == "yes" for row in rows)
        summary["f1"] = 2 * tp / (predicted_yes + actual_yes) if predicted_yes + actual_yes else 0.0
        summary["invalid_answers"] = sum(row["label"] is None for row in rows)
    return summary


def load_benchmark_model(model_path, processor_path, awq_backend, max_image_tokens, processor=None):
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    quantization = getattr(config, "quantization_config", None) or {}
    is_awq = quantization.get("quant_method") == "awq" and quantization.get("bits") == 4
    if (quantization or awq_backend) and not is_awq:
        raise ValueError("Quantized runs require an INT4 AWQ checkpoint")
    if is_awq and "visual" in (quantization.get("modules_to_not_convert") or []):
        # Transformers nests the vision tower under model, unlike the original checkpoint.
        quantization["modules_to_not_convert"] = [*quantization["modules_to_not_convert"], "model.visual"]
    if awq_backend:
        quantization["backend"] = awq_backend
    dtype = torch.float16 if is_awq else torch.bfloat16
    model = AutoModelForMultimodalLM.from_pretrained(model_path, config=config, dtype=dtype,
                                                    device_map={"": torch.cuda.current_device()},
                                                    attn_implementation="sdpa",
                                                    local_files_only=True).eval()
    if processor is None:
        processor = AutoProcessor.from_pretrained(processor_path, max_pixels=max_image_tokens * 28**2,
                                                  local_files_only=True)
    packed = [module for module in model.modules()
              if isinstance(getattr(module, "qweight", None), torch.Tensor)
              and module.qweight.dtype == torch.int32 and module.qweight.is_cuda]
    if is_awq and not packed:
        raise RuntimeError("The AWQ checkpoint did not load packed INT4 weights onto CUDA")
    backend_classes = sorted({type(module).__name__ for module in packed})
    if is_awq:
        print(f"AWQ CUDA modules: {len(packed)}, kernels: {', '.join(backend_classes)}", flush=True)
    return model, processor, dict(model=str(model_path), processor=str(processor_path),
                                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                                  dtype=str(dtype).removeprefix("torch."), batch_size=1,
                                  weight_format="int4_awq" if is_awq else "bf16",
                                  awq_backend_classes=backend_classes, packed_cuda_modules=len(packed),
                                  max_image_tokens=max_image_tokens)


def evaluate_configuration(model, processor, samples, dataset, ratio, progress=None):
    max_new_tokens = 6 if dataset == "pope" else 16
    adapter = None
    try:
        if ratio:
            mapping = yaml.safe_load(VISIONZIP.read_text(encoding="utf-8"))["model_mapping"]
            config = TokenCompressorConfig.from_yaml(str(VISIONZIP))
            config.strategies["global"].params["ratio"] = ratio
            adapter = UniversalPruningAdapter(model, config, mapping)
            adapter.wrap_model()
        measure(model, processor, samples[0], dataset, max_new_tokens)
        rows = []
        for sample in samples:
            rows.append(measure(model, processor, sample, dataset, max_new_tokens))
            if progress is not None:
                progress(len(rows), len(samples))
        return dict(summary=summarize(rows, dataset), rows=rows)
    finally:
        if adapter is not None:
            adapter.unwrap_model()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path.cwd() / "models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--processor", type=Path, help="Use the same processor across checkpoints")
    parser.add_argument("--awq-backend", help="Override the Transformers AWQ kernel for a packed checkpoint")
    parser.add_argument("--dataset", choices=["textvqa", "pope"], required=True)
    parser.add_argument("--limit", type=int, default=48)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--ratios", type=float, nargs="*", default=[0.5, 0.75, 0.9])
    parser.add_argument("--baseline-last", action="store_true", help="Measure the baseline after the pruning runs")
    parser.add_argument("--max-image-tokens", type=int, default=768)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit < 1 or any(not 0 < ratio < 1 for ratio in args.ratios):
        parser.error("limit must be positive and compression ratios must be between zero and one")
    samples, source_files = load_samples(args.dataset, args.limit, args.seed)
    model, processor, details = load_benchmark_model(args.model, args.processor or args.model,
                                                     args.awq_backend, args.max_image_tokens)
    max_new_tokens = 6 if args.dataset == "pope" else 16
    report = dict(**details, dataset=args.dataset, seed=args.seed, sources=source_files,
                  score_definition="POPE yes/no accuracy" if args.dataset == "pope" else
                  "VQA annotator consensus with lightweight answer normalization",
                  max_new_tokens=max_new_tokens, configurations={})
    args.output.parent.mkdir(exist_ok=True, parents=True)
    ratios = [*args.ratios, 0.0] if args.baseline_last else [0.0, *args.ratios]
    for ratio in ratios:
        name = f"visionzip_{ratio:g}" if ratio else "baseline"
        def progress(done, total):
            if done % 12 == 0:
                print(f"{name}: {done}/{total}", flush=True)

        result = evaluate_configuration(model, processor, samples, args.dataset, ratio, progress)
        report["configurations"][name] = result
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(name, json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
