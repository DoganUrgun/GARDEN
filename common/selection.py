"""Pre-registered sparse-return aggregation and deterministic selection."""

from __future__ import annotations

from typing import Any

import numpy as np


def score_candidate(
    candidate_returns: dict[str, float],
    baseline_returns: dict[str, float],
) -> dict[str, Any]:
    if set(candidate_returns) != set(baseline_returns):
        raise ValueError("Candidate and baseline scenario sets differ")
    normalized = {
        scenario: (float(candidate_returns[scenario]) - float(baseline_returns[scenario]))
        / max(abs(float(baseline_returns[scenario])), 1.0)
        for scenario in baseline_returns
    }
    return {
        "scenario_sparse_returns": {scenario: float(candidate_returns[scenario]) for scenario in baseline_returns},
        "baseline_sparse_returns": {scenario: float(baseline_returns[scenario]) for scenario in baseline_returns},
        "scenario_normalized_improvements": normalized,
        "aggregate_score": float(np.mean(list(normalized.values()))),
        "worst_scenario_improvement": float(min(normalized.values())),
        "mean_sparse_return": float(np.mean(list(candidate_returns.values()))),
    }


def rank_candidates(
    candidates: dict[str, dict[str, float]],
    baseline: dict[str, float],
) -> list[dict[str, Any]]:
    rows = [
        {"candidate": candidate, **score_candidate(returns, baseline)}
        for candidate, returns in candidates.items()
    ]
    rows.sort(key=lambda row: (-row["aggregate_score"], -row["worst_scenario_improvement"], -row["mean_sparse_return"], row["candidate"]))
    for position, row in enumerate(rows, start=1):
        row["rank"] = position
        row["selected"] = position == 1
    return rows
