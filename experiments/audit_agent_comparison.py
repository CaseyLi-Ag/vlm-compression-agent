"""Recompute paired metrics from the comparison command's question-level files."""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def round_metrics(artifact, samples, reported):
    rows = artifact["rows"]
    row_keys = [(row["question_id"], row.get("category")) for row in rows]
    sample_keys = [(row["question_id"], row.get("category")) for row in samples]
    require(row_keys == sample_keys and len(set(row_keys)) == len(rows), "Question order or identities differ")
    score = mean(row["score"] for row in rows)
    ttft_ms = median(row["ttft_ms"] for row in rows)
    peak_mib = max(row["peak_allocated_mib"] for row in rows)
    require(len(rows) == reported["samples"] and artifact["summary"]["score"] == score
            and math.isclose(reported["score"], score)
            and math.isclose(reported["median_ttft_ms"], ttft_ms)
            and math.isclose(reported["peak_allocated_mib"], peak_mib), "Summary differs from raw rows")
    return rows, dict(score_percent=score * 100, peak_allocated_gib=peak_mib / 1024, median_ttft_ms=ttft_ms)


def paired_image_ci(samples, baseline, candidate, seed):
    # Resample complete image groups because multiple questions may share one image.
    groups = defaultdict(list)
    for sample, reference, row in zip(samples, baseline, candidate, strict=True):
        groups[sample["image_id"]].append(row["score"] - reference["score"])
    clusters = [(sum(values), len(values)) for values in groups.values()]
    rng = random.Random(seed)
    draws = []
    for _ in range(2000):
        bootstrap = rng.choices(clusters, k=len(clusters))
        draws.append(100 * sum(total for total, _ in bootstrap) / sum(count for _, count in bootstrap))
    draws.sort()
    return [draws[49], draws[1949]]


def audit(path):
    report = read_json(path / "comparison.json")
    require(report["status"] == "completed", f"Comparison is incomplete: {path}")
    manifest = read_json(path / report["sample_manifest"])
    for task, samples in manifest.items():
        require({row["image_id"] for row in samples["search"]}.isdisjoint(
                row["image_id"] for row in samples["validation"]), f"{task}: overlap across splits")
        for split in ("search", "validation"):
            require(len(samples[split]) == report[f"{split}_samples_per_task"], f"{task}: unexpected sample count")
    goal = report["goal"]
    reference = {}
    checked = {}
    for observation in report["measurements"]:
        number = observation["measurement"]
        candidate, split = observation["candidate"], observation["split"]
        data = dict(candidate=candidate, split=split, status=observation["status"])
        if observation["status"] == "measured":
            data["tasks"] = {}
            for task, metrics in observation["metrics"].items():
                artifact = read_json(path / metrics["report"])
                require((artifact["candidate"], artifact["split"], artifact["dataset"]) ==
                        (candidate, split, task), f"Measurement {number}: wrong case or dataset")
                rows, summary = round_metrics(artifact, manifest[task][split], metrics)
                key = (split, task)
                if candidate["weight_format"] == "bf16":
                    reference[key] = rows, summary
                bf16_rows, bf16 = reference[key]
                summary["score_delta_pp"] = summary["score_percent"] - bf16["score_percent"]
                if split == "validation" and candidate["weight_format"] != "bf16":
                    summary["score_delta_image_bootstrap_95_pp"] = paired_image_ci(
                        manifest[task][split], bf16_rows, rows, report["seed"])
                data["tasks"][task] = summary
            data["eligible"] = all(
                m["peak_allocated_gib"] < goal["max_vram_gib"]
                and m["score_delta_pp"] >= -goal["max_score_drop_pp"] - 1e-9 for m in data["tasks"].values())
            data["mean_task_median_ttft_ms"] = mean(m["median_ttft_ms"] for m in data["tasks"].values())
            data["peak_allocated_gib"] = max(m["peak_allocated_gib"] for m in data["tasks"].values())
        else:
            data["error"] = observation["error"]
        checked[number] = data
    policies = {}
    for name, policy in report["policies"].items():
        rounds = policy["rounds"]
        require(len(rounds) == policy["rounds_used"] <= goal["max_rounds"], f"{name}: budget mismatch")
        baseline_eligible = name in {"adaptive", "grid"} and rounds[0].get("eligible")
        valid = []
        for round_ in rounds:
            observed = checked[round_["measurement"]]
            require(round_["candidate"] == observed["candidate"] and round_["split"] == observed["split"],
                    f"{name}: measurement reference mismatch")
            require(round_["status"] == observed["status"], f"{name}: measurement status mismatch")
            if round_["status"] == "measured":
                require(round_["eligible"] == observed["eligible"], f"{name}: eligibility mismatch")
                require(math.isclose(round_["mean_ttft_ms"], observed["mean_task_median_ttft_ms"]),
                        f"{name}: TTFT mismatch")
                is_candidate = round_["candidate"]["weight_format"] != "bf16" or baseline_eligible
                if observed["eligible"] and is_candidate and round_["split"] == "validation":
                    valid.append(round_)
        recommended = policy["recommendation"]
        best = min(valid, key=lambda item: checked[item["measurement"]]["mean_task_median_ttft_ms"]) if valid else None
        require((recommended and recommended["round"]) == (best and best["round"]),
                f"{name}: recommendation mismatch")
        policies[name] = dict(status=policy["status"], rounds=policy["rounds_used"],
                             measured_seconds=policy["measured_seconds"],
                             recommendation=best["measurement"] if best else None,
                             validation=[r["measurement"] for r in rounds if r["split"] == "validation"])
    return dict(seed=report["seed"], goal=goal, policies=policies, measurements=checked,
                search_samples_per_task=report["search_samples_per_task"],
                validation_samples_per_task=report["validation_samples_per_task"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([audit(path) for path in args.runs], indent=2))


if __name__ == "__main__":
    main()
