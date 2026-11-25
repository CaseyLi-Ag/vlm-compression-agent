"""Run several policies against the same paired measurements."""

import time
from dataclasses import asdict

from .controller import POLICIES, Candidate, run_search


def compare_policies(goal, evaluate, persist):
    report = dict(status="running", goal=asdict(goal), policies={}, measurements=[])
    measurements = {}

    def measure_once(candidate, split, number):
        key = (candidate, split)
        if key not in measurements:
            entry = dict(measurement=len(measurements) + 1, split=split, candidate=asdict(candidate))
            start = time.monotonic()
            try:
                entry.update(status="measured", metrics=evaluate(candidate, split, entry["measurement"]))
            except Exception as error:
                entry.update(status="failed", error=f"{type(error).__name__}: {error}")
            entry["elapsed_seconds"] = round(time.monotonic() - start, 3)
            measurements[key] = entry
            report["measurements"].append(entry)
            persist(report)
        entry = measurements[key]
        if entry["status"] == "failed":
            raise RuntimeError(entry["error"])
        return entry["metrics"]

    for policy in POLICIES:
        def save_policy(trial_report):
            report["policies"][policy] = trial_report
            persist(report)

        result = run_search(goal, measure_once, save_policy, policy=policy)
        # Report the cost of running this policy by itself, including observations
        # shared with another policy in this comparison.
        for trial in result["rounds"]:
            observation = measurements[Candidate(**trial["candidate"]), trial["split"]]
            trial["measurement"] = observation["measurement"]
            trial["measured_seconds"] = observation["elapsed_seconds"]
        result["measured_seconds"] = round(sum(trial["measured_seconds"] for trial in result["rounds"]), 3)
        save_policy(result)
    report["status"] = "completed"
    persist(report)
    return report
