"""Fixed evidence interfaces for the two controls (never used to guide GARDEN).

The schema and source packet are registered before candidate generation. No
trajectories, checkpoints, shell tools or ad-hoc requests reach either control.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
import statistics
from pathlib import Path

import config
from common.records import file_hash, read_json, write_json
from common.selection import rank_candidates
from common.workspace import _copy_reference


SCHEMA = {
    "version": 2, "learning_curve_bins": 5,
    "fixed_fields": ["sparse_task_return", "learning_curve", "episode_length",
                     "policy_loss", "value_loss", "entropy", "generated_shaping_reward"],
    "selection_metadata": ["aggregate_score", "scenario_normalized_improvements"],
    "shaping_population": "all training rollout agent-transitions, not selected final traces",
    "eureka_extra": "per-component raw reward statistics in each learning-curve bin",
}
SOURCE_FILES = (
    "PROTOCOL.md", "PUBLIC_FAMILY_CONFIG.json", "config.py", "validate_reward.py",
    "SMAX/adapter.py", "SMAX/networks.py", "common/reward.py", "common/mappo.py",
    "common/networks.py", "runtime/jaxmarl/environments/multi_agent_env.py",
    "runtime/jaxmarl/environments/smax/smax_env.py",
    "runtime/jaxmarl/environments/smax/heuristic_enemy_smax_env.py",
    "runtime/jaxmarl/environments/smax/heuristic_enemy.py",
    "runtime/jaxmarl/environments/smax/distributions.py",
)


def reference_packet(run, family) -> str:
    """Materialize one immutable, development-only source packet per search."""
    folder = run.folder / "fixed_reference"
    packet = folder / "packet.md"
    manifest = folder / "manifest.json"
    if manifest.exists():
        registered = read_json(manifest, {})
        if not packet.is_file() or file_hash(packet) != registered.get("packet_hash"):
            raise RuntimeError("Registered fixed source packet changed")
        return packet.read_text(encoding="utf-8")
    # No model request can precede this function returning its registered packet.
    # It is therefore safe to rebuild setup interrupted before manifest creation.
    # Missing registration *after* a request instead means corrupted provenance.
    if any((run.folder / "control_drafts").glob("*/request_*.json")):
        raise RuntimeError("Fixed source manifest is missing after model requests; existing evidence cannot be reconstructed silently")
    if folder.exists():
        shutil.rmtree(folder)
    reference = folder / "source"
    _copy_reference(reference, family)
    parts = ["# Fixed development source packet"]
    for name in SOURCE_FILES:
        path = reference / name
        parts.append(f"## {name}\n```text\n{path.read_text(encoding='utf-8')}\n```")
    text = "\n\n".join(parts)
    if any(name in text for name in family.held_out):
        raise RuntimeError("Held-out definition leaked into the fixed packet")
    packet.write_text(text, encoding="utf-8")
    write_json(manifest, {
        "packet_hash": file_hash(packet),
        "files": {name: file_hash(reference / name) for name in SOURCE_FILES},
        "schema": SCHEMA,
    })
    return text


def number(row, name):
    value = row.get(name)
    if value in (None, "", "None"):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def average(rows, name):
    values = [value for row in rows if (value := number(row, name)) is not None]
    return statistics.fmean(values) if values else None


def episode_average(rows, name):
    """Aggregate episode means by episode count, not by rollout count."""
    total, count = 0.0, 0.0
    for row in rows:
        value = number(row, name)
        episodes = number(row, "completed_episodes") or 0.0
        if value is not None and episodes > 0:
            total += value * episodes
            count += episodes
    return total / count if count else None


def shaping_summary(rows, prefix="rollout_llm_reward"):
    # Each recorded rollout contains the same number of transitions. Combine
    # second moments; averaging standard deviations would be incorrect.
    values = [(number(row, f"{prefix}_mean"),
               number(row, f"{prefix}_std")) for row in rows]
    values = [(mean, std) for mean, std in values if mean is not None and std is not None]
    if not values:
        return {key: None for key in ("mean", "std", "min", "max")}
    mean = statistics.fmean(item[0] for item in values)
    variance = statistics.fmean(std * std + value * value for value, std in values) - mean * mean
    low = [value for row in rows if (value := number(row, f"{prefix}_min")) is not None]
    high = [value for row in rows if (value := number(row, f"{prefix}_max")) is not None]
    return {"mean": mean, "std": math.sqrt(max(0, variance)),
            "min": min(low) if low else None, "max": max(high) if high else None}


def diagnostic(job: Path, components: bool = False):
    result = read_json(job / "result.json", {})
    if not result:
        raise RuntimeError(f"Feedback requires a completed development job: {job}")
    with (job / "training_curve.csv").open(encoding="utf-8") as stream:
        curve = list(csv.DictReader(stream))
    bins = []
    for index in range(5):
        rows = curve[index * len(curve) // 5:(index + 1) * len(curve) // 5]
        weighted, count = 0.0, 0.0
        for row in rows:
            episodes = number(row, "completed_episodes") or 0
            value = number(row, "completed_task_return_mean")
            if value is not None:
                weighted += episodes * value
                count += episodes
        item = {"fraction": [(index / 5), ((index + 1) / 5)],
                "sparse_episode_return": weighted / count if count else None,
                "completed_episodes": count}
        if components:
            # Pool component moments exactly like total shaping; averaging
            # standard deviations/minima would misdescribe the bin's values.
            names = sorted({key[len("raw_component_"):-len("_mean")]
                            for row in rows for key in row
                            if key.startswith("raw_component_") and key.endswith("_mean")})
            item["reward_components"] = {
                name: shaping_summary(rows, f"raw_component_{name}") for name in names
            }
        bins.append(item)
    return {
        "scenario": result["scenario"], "seed": result["seed"],
        "sparse_task_return": result["evaluation"]["task_mean"],
        "learning_curve": bins,
        "episode_length": episode_average(curve, "completed_episode_length_mean"),
        **{metric: {"mean": average(curve, metric),
                    "last_fifth_mean": average(curve[max(0, len(curve) * 4 // 5):], metric)}
           for metric in ("policy_loss", "value_loss", "entropy")},
        "generated_shaping_reward": shaping_summary(curve),
    }


def measured_report(run, family, candidate, *, components=False):
    # This is the already registered selection objective, not a new diagnostic.
    # Every report has the same keys; native is the zero-improvement reference.
    if candidate == "native":
        score = 0.0
        normalized = {scenario: 0.0 for scenario in family.development}
    else:
        summary = read_json(run.folder / "family" / "candidates" / candidate / "summary.json", {})
        if "aggregate_score" not in summary or "scenario_normalized_improvements" not in summary:
            raise RuntimeError(f"Completed mechanical selection summary is missing for {candidate}")
        score = summary["aggregate_score"]
        normalized = summary["scenario_normalized_improvements"]
    report = {"schema": SCHEMA, "candidate": candidate,
              "aggregate_score": score, "scenario_normalized_improvements": normalized,
              "jobs": [
        diagnostic(run.folder / "family" / "development" / candidate / scenario / f"seed_{seed}", components)
        for scenario in family.development for seed in config.TRAIN_SEEDS
    ]}
    write_json(run.folder / "family" / "diagnostics" / f"{candidate}.json", report)
    return report


def design_text(run, candidate):
    folder = run.folder / "family" / "candidates" / candidate
    return (f"## Reward {candidate}\n```python\n{(folder / 'reward.py').read_text(encoding='utf-8')}\n```\n"
            f"## Design report\n{(folder / 'report.md').read_text(encoding='utf-8')}")


def learning_probe_history(run, candidate, *, components=False):
    """Prior private experiments, projected onto the registered control schema.

    Only control_context chooses whose history is permitted. No checkpoint,
    trajectory, shell output or controller path is included in this projection.
    """
    root = run.folder / "learning_probes" / candidate
    ledger = read_json(root / "ledger.json", {})
    history = []
    for attempt in ledger.get("attempts", []):
        archive = root / attempt["id"]
        for relative, digest in attempt.get("artifact_hashes", {}).items():
            path = archive / relative
            if not path.is_file() or file_hash(path) != digest:
                raise RuntimeError("Registered learning-probe history changed")
        source = archive / "reward.py"
        item = {key: attempt.get(key) for key in (
            "id", "status", "scenario", "seed", "hypothesis", "reward_hash")}
        item["reward_source"] = source.read_text(encoding="utf-8") if source.is_file() else None
        item["diagnostic"] = diagnostic(archive / "training", components=components) if attempt["status"] == "complete" else None
        # Runtime internals are retained in the archive, not sent to controls.
        item["error"] = None if attempt["status"] == "complete" else "The registered training attempt did not complete."
        history.append(item)
    return history


def control_context(run, family, generation, position):
    """Both siblings have identical inputs; only earlier generations are visible."""
    parts = [reference_packet(run, family), "## Fixed evidence schema\n" + json.dumps(SCHEMA)]
    visible = []
    if config.EXPERIMENT == "exp2":
        parts.append("## Native development baseline\n" + json.dumps(measured_report(run, family, "native")))
        visible = [candidate for pair in config.GENERATIONS[:generation - 1] for candidate in pair]
    elif generation > 1:
        pair = config.GENERATIONS[generation - 2]
        baseline = read_json(run.folder / "family" / "baseline_summary.json")["scenario_sparse_returns"]
        scores = {candidate: read_json(run.folder / "family" / "candidates" / candidate / "summary.json")["scenario_sparse_returns"] for candidate in pair}
        visible = [rank_candidates(scores, baseline)[0]["candidate"]]
    for candidate in visible:
        parts.append(design_text(run, candidate))
        parts.append("## Measured development feedback\n" + json.dumps(
            measured_report(run, family, candidate, components=config.EXPERIMENT == "exp3")))
        parts.append("## Earlier learning-probe feedback\n" + json.dumps(
            learning_probe_history(run, candidate, components=config.EXPERIMENT == "exp3")))
    return "\n\n".join(parts)
