"""Search local Qwen2.5-VL compression configurations under a GPU and score budget."""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean


TASKS = ("pope", "textvqa")
MODEL = "Qwen2.5-VL-3B-Instruct"
POLICIES = ("adaptive", "grid", "fixed-awq", "fixed-visionzip")
BASELINE_FAILURES = ("baseline_failed", "validation_baseline_failed")


@dataclass(frozen=True)
class Candidate:
    weight_format: str
    compression_ratio: float


@dataclass(frozen=True)
class Goal:
    max_vram_gib: float
    max_score_drop_pp: float
    max_rounds: int

    def __post_init__(self):
        if not math.isfinite(self.max_vram_gib) or self.max_vram_gib <= 0:
            raise ValueError("--max-vram-gib must be positive and finite")
        if not 0 <= self.max_score_drop_pp <= 100:
            raise ValueError("--max-score-drop-pp must be between 0 and 100")
        if self.max_rounds < 5:
            raise ValueError("--max-rounds must allow at least 3 search and 2 validation rounds")


BF16 = Candidate("bf16", 0)
AWQ = Candidate("awq", 0)
FIXED_VISIONZIP = Candidate("awq", 0.25)
GRID_RATIOS = (0.25, 0.5, *(i / 2**level for level in range(3, 7)
                           for i in range(1, 2 ** (level - 1), 2)))


def mean_ttft(metrics):
    return mean(metrics[task]["median_ttft_ms"] for task in TASKS)


def assess(metrics, baseline, goal):
    violations = []
    deltas = {}
    for task in TASKS:
        score = metrics[task]["score"]
        peak = metrics[task]["peak_allocated_mib"]
        if not 0 <= score <= 1 or not math.isfinite(peak) or peak <= 0:
            raise ValueError(f"{task} returned an invalid score or peak memory")
        ttft = metrics[task]["median_ttft_ms"]
        if not math.isfinite(ttft) or ttft <= 0:
            raise ValueError(f"{task} returned an invalid TTFT")
        difference = (score - baseline[task]["score"]) * 100
        deltas[task] = difference
        if peak >= goal.max_vram_gib * 1024:
            violations.append(f"{task} peak memory {peak / 1024:.2f} GiB reached the {goal.max_vram_gib:g} GiB limit")
        if difference < -goal.max_score_drop_pp - 1e-9:
            violations.append(f"{task} score dropped {-difference:.2f} percentage points")
    return deltas, violations


def run_search(goal, evaluate, persist, *, policy="adaptive"):
    """Execute at most one two-task evaluation per round, including validation and failures."""
    reserve = 3 if goal.max_rounds >= 6 else 2
    search_limit = goal.max_rounds - reserve
    if policy not in POLICIES:
        raise ValueError(f"Unknown search policy {policy}")
    fixed = {"fixed-awq": AWQ, "fixed-visionzip": FIXED_VISIONZIP}.get(policy)
    report = dict(goal=asdict(goal), policy=policy, rounds_used=0, status="running", recommendation=None,
                  search_round_limit=search_limit, validation_rounds_reserved=reserve, rounds=[])

    def trial(candidate, split, reason, baseline=None):
        number = report["rounds_used"] + 1
        if number > goal.max_rounds:
            raise RuntimeError("Round budget exhausted")
        entry = dict(round=number, split=split, candidate=asdict(candidate), reason=reason, status="running")
        report["rounds_used"] = number
        report["rounds"].append(entry)
        persist(report)
        start = time.monotonic()
        try:
            metrics = evaluate(candidate, split, number)
            deltas, violations = assess(metrics, baseline if baseline is not None else metrics, goal)
        except Exception as error:
            entry.update(status="failed", error=f"{type(error).__name__}: {error}")
        else:
            entry.update(status="measured", metrics=metrics, score_delta_pp=deltas,
                         violations=violations, eligible=not violations, mean_ttft_ms=mean_ttft(metrics))
        entry["elapsed_seconds"] = round(time.monotonic() - start, 3)
        persist(report)
        return entry

    def finish(status, reason):
        report.update(status=status, stop_reason=reason)
        persist(report)
        return report

    search_baseline = trial(BF16, "search", "Measure BF16 as the score reference for the search images.")
    if search_baseline["status"] == "failed":
        return finish("baseline_failed", "A valid BF16 search score reference is required.")
    reference = search_baseline["metrics"]
    initial = fixed or AWQ
    initial_reason = ("Measure the fixed configuration on the search images." if fixed else
                      "Measure weight quantization alone before adjusting the visual token count.")
    awq = trial(initial, "search", initial_reason, reference)
    eligible = {candidate: entry for candidate, entry in ((BF16, search_baseline), (initial, awq))
                if entry.get("eligible")}
    best_awq = awq["mean_ttft_ms"] if awq.get("eligible") else math.inf
    low, high = 0.0, 0.5
    reason = "Start at 25% token removal within the range from 0% to 50%."
    for grid_ratio in GRID_RATIOS[:search_limit - 2]:
        if fixed or (policy == "adaptive" and high - low <= 0.015625):
            break
        ratio = grid_ratio if policy == "grid" else (low + high) / 2
        candidate = Candidate("awq", ratio)
        entry = trial(candidate, "search", "Measure the next predetermined grid point." if policy == "grid"
                      else reason, reference)
        if entry.get("eligible"):
            eligible[candidate] = entry
        if policy == "grid":
            continue
        if entry.get("eligible") and entry["mean_ttft_ms"] < best_awq:
            best_awq = entry["mean_ttft_ms"]
            low = ratio
            reason = f"{ratio:.1%} met both constraints and improved AWQ latency. Try removing more tokens."
        else:
            high = ratio
            finding = entry.get("error") or ", ".join(entry.get("violations", [])) or "TTFT did not improve"
            reason = f"At {ratio:.1%} removal, {finding}. Try retaining more tokens."

    if not eligible and not fixed:
        return finish("no_feasible_search_candidate", "No measured search configuration met both limits.")
    ranked = [fixed] if fixed else sorted(eligible, key=lambda candidate: eligible[candidate]["mean_ttft_ms"])
    validation_baseline = trial(BF16, "validation", "Measure the BF16 score reference on held-out images.")
    if validation_baseline["status"] == "failed":
        return finish("validation_baseline_failed", "A valid BF16 holdout reference is required.")
    verified = [validation_baseline] if not fixed and BF16 in eligible and validation_baseline["eligible"] else []
    finalists = [candidate for candidate in ranked if candidate != BF16][:reserve - 1]
    for candidate in finalists:
        entry = trial(candidate, "validation", "Recheck a search finalist on held-out images.",
                      validation_baseline["metrics"])
        if entry.get("eligible"):
            verified.append(entry)
    if not verified:
        return finish("no_verified_candidate", "No finalist met both limits on held-out images.")
    best = min(verified, key=lambda item: item["mean_ttft_ms"])
    report["recommendation"] = {key: best[key] for key in
                                ("candidate", "round", "metrics", "score_delta_pp", "mean_ttft_ms")}
    return finish("recommended", "Round budget reached." if report["rounds_used"] == goal.max_rounds
                  else "All shortlisted configurations were checked.")


def write_json(path, contents):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(contents, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def make_sample_manifest(rows_by_split, image_key, sources):
    manifest = {"sources": sources}
    for split, rows in rows_by_split.items():
        manifest[split] = []
        for row in rows:
            item = {"question_id": row["question_id"], "image_id": row[image_key]}
            if "category" in row:
                item["category"] = row["category"]
            manifest[split].append(item)
    return manifest


class LocalEvaluator:
    def __init__(self, args, output, cached=None):
        self.output = output
        self.args = args
        self.cached = cached or {}
        self.gpus = {metric["gpu"] for metrics in self.cached.values() for metric in metrics.values()}

    def __call__(self, candidate, split, number):
        key = (split, candidate)
        if key in self.cached:
            return self.cached[key]
        request = dict(args=vars(self.args), output=self.output, candidate=asdict(candidate),
                       split=split, number=number)
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        env.pop("DEEPSEEK_API_KEY", None)
        result = subprocess.run([sys.executable, "-u", "-m", "vlm_compression_agent", "--worker"],
                                input=json.dumps(request, default=str), encoding="utf-8", env=env,
                                cwd=Path(__file__).resolve().parents[1])
        if result.returncode:
            error_file = self.output / f"measurements/round-{number:02d}-error.json"
            error = json.loads(error_file.read_text(encoding="utf-8"))["error"] if error_file.exists() else \
                f"GPU worker exited with code {result.returncode}"
            raise RuntimeError(error)
        metrics = {}
        for task in TASKS:
            artifact = f"measurements/round-{number:02d}-{task}.json"
            measurement = json.loads((self.output / artifact).read_text(encoding="utf-8"))
            if self.gpus and measurement["gpu"] not in self.gpus:
                raise ValueError("The GPU differs from the other measurements in this run")
            self.gpus.add(measurement["gpu"])
            metrics[task] = dict(**measurement["summary"], report=artifact, gpu=measurement["gpu"])
        return metrics


def run_worker():
    request = json.load(sys.stdin)
    args = argparse.Namespace(**request["args"])
    output = Path(request["output"])
    candidate = Candidate(**request["candidate"])
    split, number = request["split"], request["number"]
    try:
        from . import benchmark_qwen25_vl as benchmark

        benchmark.torch.cuda.set_device(args.cuda_device)
        samples, manifest = {}, {}
        for task in TASKS:
            phases, sources, image_key = benchmark.load_grouped_splits(
                task, args.search_samples, args.validation_samples, args.seed, Path(args.datasets_dir))
            manifest[task] = make_sample_manifest(phases, image_key, sources)
            samples[task] = phases[split]
        sample_path = output / "samples.json"
        if sample_path.exists():
            if json.loads(sample_path.read_text(encoding="utf-8")) != manifest:
                raise RuntimeError("The local dataset no longer matches the run's sample manifest")
        else:
            write_json(sample_path, manifest)

        print(f"Round {number} {split} {candidate.weight_format} {candidate.compression_ratio:.2%} token removal",
              flush=True)
        is_awq = candidate.weight_format == "awq"
        model, processor, details = benchmark.load_benchmark_model(
            Path(args.awq_model if is_awq else args.bf16_model), Path(args.bf16_model),
            args.awq_backend if is_awq else None, args.max_image_tokens)
        for task in TASKS:
            def progress(done, total):
                if done % 30 == 0 or done == total:
                    print(f"  {task}: {done}/{total}", flush=True)

            result = benchmark.evaluate_configuration(model, processor, samples[task], task,
                                                      candidate.compression_ratio, progress)
            write_json(output / f"measurements/round-{number:02d}-{task}.json",
                       dict(**details, dataset=task, split=split, candidate=asdict(candidate), **result))
    except Exception as error:
        write_json(output / f"measurements/round-{number:02d}-error.json",
                   dict(error=f"{type(error).__name__}: {str(error).partition(chr(10))[0]}", trace=traceback.format_exc()))
        raise


def reuse_search_baselines(source, output, args):
    previous = json.loads((source / "report.json").read_text(encoding="utf-8"))
    settings = dict(seed=args.seed, search_samples_per_task=args.search_samples,
                    validation_samples_per_task=args.validation_samples, max_image_tokens=args.max_image_tokens)
    if any(previous[key] != value for key, value in settings.items()):
        raise ValueError("Cached search baselines were measured with different sample or image settings")
    copied = {}
    devices = set()
    for index, candidate in enumerate((BF16, AWQ), 1):
        entry = previous["rounds"][index - 1]
        if entry["status"] != "measured" or entry["split"] != "search" or entry["candidate"] != asdict(candidate):
            raise ValueError("Both original search baselines must have completed")
        metrics = {}
        for task in TASKS:
            filename = f"measurements/round-{index:02d}-{task}.json"
            measurement = json.loads((source / filename).read_text(encoding="utf-8"))
            expected_model = args.bf16_model if candidate == BF16 else args.awq_model
            if Path(measurement["model"]).resolve() != expected_model.resolve():
                raise ValueError("Cached search baselines used a different checkpoint")
            if candidate == AWQ and (args.awq_backend != "gemm_triton" or
                                     measurement["awq_backend_classes"] != ["AwqGEMMTritonLinear"]):
                raise ValueError("Cached AWQ search baseline used a different backend")
            if measurement["summary"] != {k: v for k, v in entry["metrics"][task].items()
                                           if k not in {"report", "gpu", "reused_from"}}:
                raise ValueError("Cached measurement and run summary disagree")
            devices.add(measurement["gpu"])
            shutil.copyfile(source / filename, output / filename)
            metrics[task] = dict(**measurement["summary"], report=filename, gpu=measurement["gpu"],
                                 reused_from=str((source / filename).resolve()))
        copied["search", candidate] = metrics
    if len(devices) != 1:
        raise ValueError("Cached search baselines came from different GPUs")
    shutil.copyfile(source / "samples.json", output / "samples.json")
    return copied


def demo(args, goal):
    def evaluate(candidate, split, _number):
        ratio = candidate.compression_ratio
        baseline = candidate == BF16
        return {
            task: {"score": 0.86 if baseline else 0.855 - (ratio / 20 if task == "textvqa" else 0),
                   "median_ttft_ms": 360 if baseline else 440 - 200 * ratio,
                   "peak_allocated_mib": 7300 if baseline else 3450 - 140 * ratio}
            for task in TASKS
        }

    print(json.dumps(dict(simulated=True, **run_search(goal, evaluate, lambda _: None,
                                                       policy=args.policy)), indent=2))


def main():
    if sys.argv[1:] == ["--worker"]:
        run_worker()
        return
    from . import __version__
    from .preflight import doctor

    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog="Commands: run performs a search; compare evaluates four policies; "
                                     "doctor checks a local GPU setup; demo uses simulated measurements; "
                                     "chat builds a local evaluation plan from a DeepSeek conversation.")
    parser.add_argument("command", nargs="?", choices=("run", "compare", "doctor", "demo", "chat"), default="run")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--project-dir", type=Path, default=Path.cwd(),
                        help="Directory containing models/ and datasets/ (default: current directory)")
    parser.add_argument("--datasets-dir", type=Path, help="Override the project datasets directory")
    parser.add_argument("--cuda-device", type=int, default=0,
                        help="Index within CUDA_VISIBLE_DEVICES (default: 0)")
    parser.add_argument("--policy", choices=POLICIES, default="adaptive", help="Search policy for run or demo")
    parser.add_argument("--max-vram-gib", type=float, default=4)
    parser.add_argument("--max-score-drop-pp", type=float, default=2)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--search-samples", type=int, default=120)
    parser.add_argument("--validation-samples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--max-image-tokens", type=int, default=768)
    parser.add_argument("--bf16-model", type=Path, help="Override the local unquantized checkpoint directory")
    parser.add_argument("--awq-model", type=Path, help="Override the local INT4 AWQ checkpoint directory")
    parser.add_argument("--awq-backend", default="gemm_triton")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reuse-search-baselines", type=Path,
                        help="Reuse BF16 and AWQ search measurements from this machine after an interrupted run")
    args = parser.parse_args()
    args.project_dir = args.project_dir.resolve()
    args.datasets_dir = (args.datasets_dir or args.project_dir / "datasets").resolve()
    args.bf16_model = (args.bf16_model or args.project_dir / "models" / MODEL).resolve()
    args.awq_model = (args.awq_model or args.project_dir / "models" / f"{MODEL}-AWQ").resolve()
    if args.command == "doctor":
        info = doctor(args)
        print(json.dumps(info, indent=2))
        raise SystemExit(0 if info["status"] == "ready" else 1)
    try:
        goal = Goal(args.max_vram_gib, args.max_score_drop_pp, args.max_rounds)
        if min(args.search_samples, args.validation_samples) < 6 or args.max_image_tokens < 1:
            raise ValueError("Use at least 6 questions per split and a positive image-token limit")
    except ValueError as error:
        parser.error(str(error))
    if args.command == "demo":
        demo(args, goal)
        return
    if args.cuda_device < 0:
        parser.error("--cuda-device must be nonnegative")
    if args.reuse_search_baselines and (args.command != "run" or args.policy not in {"adaptive", "grid"}):
        parser.error("--reuse-search-baselines is available for adaptive or grid run only")
    if args.command == "chat":
        from .chat import run_chat

        raise SystemExit(run_chat(args, goal, lambda confirmed: run_evaluation(args, confirmed, parser)))
    report, path = run_evaluation(args, goal, parser)
    if args.command == "compare":
        for policy, result in report["policies"].items():
            print(f"{policy}: {result['status']} after {result['rounds_used']} rounds")
        print(path)
        if all(result["status"] in BASELINE_FAILURES for result in report["policies"].values()):
            raise SystemExit(1)
        return
    print(f"{report['status']} after {report['rounds_used']} rounds", flush=True)
    if report["recommendation"]:
        best = report["recommendation"]
        print(f"Recommended {best['candidate']} with mean TTFT {best['mean_ttft_ms']:.1f} ms", flush=True)
    print(path, flush=True)
    if report["status"] in BASELINE_FAILURES:
        raise SystemExit(1)


def run_evaluation(args, goal, parser):
    from . import __version__
    from .preflight import input_issues

    issues = input_issues(args)
    if issues:
        parser.error("\n".join(issues))

    out = (args.output_dir or args.project_dir / "results" / "compression_agent"
           / datetime.now().strftime("%Y%m%d-%H%M%S-%f")).resolve()
    try:
        out.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        parser.error(str(error))
    (out / "measurements").mkdir()
    print(f"Reports: {out.resolve()}", flush=True)
    cached = reuse_search_baselines(args.reuse_search_baselines, out, args) if args.reuse_search_baselines else None
    evaluator = LocalEvaluator(args, out.resolve(), cached)
    context = dict(seed=args.seed, search_samples_per_task=args.search_samples,
                   validation_samples_per_task=args.validation_samples, max_image_tokens=args.max_image_tokens,
                   cuda_device_index=args.cuda_device, awq_backend=args.awq_backend, agent_version=__version__,
                   evaluation_process="one fresh GPU process per measured round", sample_manifest="samples.json")
    report_path = out / ("comparison.json" if args.command == "compare" else "report.json")

    def persist(report):
        write_json(report_path, dict(**context, **report))

    if args.command == "compare":
        from .comparison import compare_policies

        return compare_policies(goal, evaluator, persist), report_path
    return run_search(goal, evaluator, persist, policy=args.policy), report_path


if __name__ == "__main__":
    main()
