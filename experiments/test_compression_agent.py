import json
import unittest

from experiments.compression_agent import Candidate, Goal, make_sample_manifest, run_search


BF16 = Candidate("bf16", 0)
AWQ = Candidate("awq", 0)


def metrics(score=0.9, memory=3400, latency=400, text_score=None):
    return {
        task: {
            "score": score if task == "pope" or text_score is None else text_score,
            "peak_allocated_mib": memory,
            "median_ttft_ms": latency,
        }
        for task in ("pope", "textvqa")
    }


class SearchTests(unittest.TestCase):
    def test_pope_manifest_distinguishes_reused_question_ids(self):
        rows = {"search": [{"question_id": "1", "image_source": "image.jpg", "category": group}
                           for group in ("random", "popular")]}
        manifest = make_sample_manifest(rows, "image_source", ["pope.parquet"])
        self.assertEqual([item["category"] for item in manifest["search"]], ["random", "popular"])

    def run_fixture(self, evaluate):
        reports = []
        result = run_search(Goal(4, 2, 8), evaluate, lambda data: reports.append(json.loads(json.dumps(data))))
        self.assertEqual(reports[-1], result)
        self.assertLessEqual(result["rounds_used"], 8)
        return result

    def test_score_loss_changes_next_ratio_and_holdout_can_reject_top_choice(self):
        calls = []

        def evaluate(candidate, split, number):
            calls.append((candidate, split, number))
            if candidate == BF16:
                return metrics(memory=7300)
            if candidate == AWQ:
                return metrics()
            if candidate.compression_ratio == 0.25:
                return metrics(text_score=0.86, latency=200)
            if candidate.compression_ratio == 0.125:
                return metrics(text_score=0.86 if split == "validation" else 0.89, latency=300)
            return metrics(text_score=0.87, latency=250)

        result = self.run_fixture(evaluate)
        self.assertEqual([candidate.compression_ratio for candidate, _, _ in calls[:4]], [0, 0, 0.25, 0.125])
        self.assertEqual([number for _, _, number in calls], list(range(1, len(calls) + 1)))
        self.assertEqual([candidate for candidate, split, _ in calls if split == "validation"],
                         [BF16, Candidate("awq", 0.125), AWQ])
        self.assertEqual(result["recommendation"]["candidate"], {"weight_format": "awq", "compression_ratio": 0})

    def test_feasible_speedup_explores_more_compression(self):
        calls = []

        def evaluate(candidate, split, _number):
            if split == "search":
                calls.append(candidate)
            if candidate == BF16:
                return metrics(memory=7300)
            return metrics(latency=400 - 100 * candidate.compression_ratio)

        self.run_fixture(evaluate)
        self.assertEqual([candidate.compression_ratio for candidate in calls[:4]], [0, 0, 0.25, 0.375])

    def test_memory_limit_is_strict_and_requires_validation(self):
        def evaluate(candidate, split, _number):
            if candidate == BF16:
                return metrics(memory=7300)
            return metrics(memory=4095 if split == "search" else 4096)

        result = self.run_fixture(evaluate)
        self.assertEqual(result["status"], "no_verified_candidate")
        self.assertIsNone(result["recommendation"])
        self.assertTrue(any("peak memory" in issue for round_ in result["rounds"]
                            for issue in round_.get("violations", [])))

    def test_failed_evaluation_counts_against_budget(self):
        def evaluate(candidate, _split, _number):
            if candidate == BF16:
                return metrics(memory=7300)
            raise RuntimeError("synthetic CUDA failure")

        result = self.run_fixture(evaluate)
        self.assertEqual(result["status"], "no_feasible_search_candidate")
        self.assertEqual(result["rounds_used"], 5)
        self.assertEqual(len([r for r in result["rounds"] if r["status"] == "failed"]), 4)


if __name__ == "__main__":
    unittest.main()
