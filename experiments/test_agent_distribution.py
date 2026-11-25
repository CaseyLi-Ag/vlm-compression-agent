import unittest

from experiments.test_compression_agent import metrics
from vlm_compression_agent.controller import BF16, AWQ, Candidate, Goal, run_search
from vlm_compression_agent.comparison import compare_policies


class DistributionTests(unittest.TestCase):
    def test_grid_precommits_probes_while_adaptive_uses_feedback(self):
        def evaluate(candidate, split, number):
            return metrics(memory=7300 if candidate == BF16 else 3400,
                           latency=400 + candidate.compression_ratio * 100)

        def ratios(policy):
            report = run_search(Goal(4, 2, 7), evaluate, lambda _: None, policy=policy)
            return [item['candidate']['compression_ratio'] for item in report['rounds']
                    if item['split'] == 'search']

        self.assertEqual(ratios('adaptive'), [0, 0, 0.25, 0.125])
        self.assertEqual(ratios('grid'), [0, 0, 0.25, 0.5])

    def test_fixed_preset_has_independent_held_out_evaluation(self):
        def evaluate(candidate, split, number):
            if candidate == BF16:
                return metrics(memory=7300)
            return metrics(text_score=0.86 if split == 'search' else 0.90)

        report = run_search(Goal(4, 2, 7), evaluate, lambda _: None, policy='fixed-awq')
        self.assertEqual(report['rounds_used'], 4)
        self.assertEqual(report['recommendation']['candidate']['weight_format'], 'awq')

    def test_larger_budget_keeps_holdout_shortlist_at_two(self):
        def evaluate(candidate, split, number):
            return metrics(memory=7300 if candidate == BF16 else 3400,
                           latency=400 + candidate.compression_ratio * 100)

        report = run_search(Goal(4, 2, 12), evaluate, lambda _: None)
        attempts = [round_ for round_ in report["rounds"] if round_["split"] == "validation"]
        self.assertEqual(len(attempts), 3)

    def test_comparison_counts_rounds_per_policy_and_never_repeats_a_measurement(self):
        calls = []

        def evaluate(candidate, split, number):
            calls.append((candidate, split))
            if candidate == Candidate('awq', 0.25):
                raise RuntimeError('synthetic kernel failure')
            return metrics(memory=7300 if candidate == BF16 else 3400)

        report = compare_policies(Goal(4, 2, 7), evaluate, lambda _: None)
        self.assertEqual(len(calls), len(set(calls)))
        self.assertIn((AWQ, 'validation'), calls)
        self.assertEqual(set(report['policies']), {'adaptive', 'grid', 'fixed-awq', 'fixed-visionzip'})
        for policy in report['policies'].values():
            self.assertLessEqual(policy['rounds_used'], 7)
        self.assertEqual(report['policies']['fixed-visionzip']['status'], 'no_verified_candidate')
