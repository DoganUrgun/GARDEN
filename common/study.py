"""Controller for one independently selected environment-family study."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import config
from common.hardware import training_worker_environment
from common.records import Run, event, file_hash, now, read_json, suite_held_out_barrier, update, write_json
from common.reward import normalized_program_hash
from common.selection import rank_candidates


def design_candidate(*args):
    # Import the SDK only when actual design begins; plans and tests are offline.
    from common.designer import design_candidate as design
    return design(*args)


def _design_complete(folder: Path) -> bool:
    design = read_json(folder / "design.json", {})
    source, report = folder / "reward.py", folder / "report.md"
    if not (design and source.is_file() and report.is_file()):
        return False
    source_text = source.read_text(encoding="utf-8")
    return bool(
        design.get("reward_hash") == file_hash(source)
        and design.get("report_hash") == file_hash(report)
        and design.get("normalized_program_hash") == normalized_program_hash(source_text)
    )


def _llm_calls_after_lock(run: Run, family_name: str) -> int:
    locked, calls = False, 0
    if not run.events_file.is_file():
        return 0
    with run.events_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("family") != family_name:
                continue
            if row.get("kind") == "reward_locked":
                locked = True
            elif locked and str(row.get("kind", "")).startswith("llm_"):
                calls += 1
    return calls


def _job_folder(run, family, phase, method, scenario, seed):
    return run.folder / "family" / phase / method / scenario / f"seed_{seed}"


def _job_matches(result, family, scenario, seed, mode, reward_hash, accelerator):
    worker_accelerator = result.get("accelerator", {})
    return (
        result.get("family") == family.name
        and result.get("scenario") == scenario
        and result.get("seed") == seed
        and result.get("mode") == mode
        and result.get("reward_hash") == reward_hash
        and result.get("requested_timesteps") == family.timesteps
        and result.get("actual_timesteps") == (family.timesteps // (family.learner.num_envs * family.learner.rollout_steps)) * (family.learner.num_envs * family.learner.rollout_steps)
        and result.get("learner") == family.learner.to_dict()
        and result.get("evaluation_episodes") == family.evaluation_episodes
        and result.get("trace_episodes") == family.trace_episodes
        and result.get("snapshot_episodes") == config.SNAPSHOT_EPISODES
        and result.get("snapshot_fractions") == (list(config.SNAPSHOT_FRACTIONS) if config.SNAPSHOT_EPISODES else [])
        and result.get("snapshot_seed") == seed + 300_000
        and result.get("evaluation_greedy") == config.EVALUATION_GREEDY
        and result.get("reward_weight") == (family.reward_weight if mode == "candidate" else None)
        and all(
            worker_accelerator.get(name) == accelerator.get(name)
            for name in ("backend", "devices", "process_count")
        )
    )


def _run_training_job(run: Run, family, phase, method, scenario, seed, mode, reward_path):
    folder = _job_folder(run, family, phase, method, scenario, seed)
    reward_hash = file_hash(reward_path) if reward_path else None
    accelerator = read_json(run.folder / "manifest.json", {}).get("accelerator", {})
    existing = read_json(folder / "result.json", {})
    if existing:
        if not _job_matches(existing, family, scenario, seed, mode, reward_hash, accelerator):
            raise RuntimeError(f"Registered job result is incompatible: {folder}")
        event(run, "training_job_reused", f"Reused completed {family.name} {method} on {scenario}, seed {seed}", family=family.name, method=method, scenario=scenario, seed=seed)
        print(f"Reused {phase} {family.name}/{method}/{scenario}, seed {seed}: sparse mean={existing['evaluation']['task_mean']:.6g}", flush=True)
        return existing
    folder.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(config.ROOT / "training.py"), "--family", family.name,
        "--scenario", scenario, "--seed", str(seed), "--mode", mode,
        "--output", str(folder), "--timesteps", str(family.timesteps),
        "--eval-episodes", str(family.evaluation_episodes), "--trace-episodes", str(family.trace_episodes),
        "--snapshot-episodes", str(config.SNAPSHOT_EPISODES),
    ]
    if reward_path:
        command.extend(("--reward", str(reward_path)))
    write_json(folder / "controller_job.json", {
        "command": command, "family": family.name, "phase": phase, "method": method,
        "scenario": scenario, "seed": seed, "mode": mode, "reward_hash": reward_hash, "created_at": now(),
    })
    update(run, family=family.name, candidate=None if method == "native" else method, phase=phase, activity=f"Training {family.name} {method} on {scenario}, seed {seed}. The policy starts from scratch.", current_job=str(folder.relative_to(run.folder)))
    event(run, "training_job_started", f"Started {family.name} {method} on {scenario}, seed {seed}", family=family.name, phase=phase, method=method, scenario=scenario, seed=seed, reward_hash=reward_hash)
    environment = training_worker_environment()
    last_error = ""
    for attempt in range(1, config.TRAINING_JOB_RETRIES + 2):
        print(f"Training {phase} {family.name}/{method}/{scenario}, seed {seed}, attempt {attempt}: {family.timesteps:,} requested steps", flush=True)
        with (folder / "worker.log").open("a", encoding="utf-8") as log:
            log.write(f"\n[{now()}] attempt {attempt}\n")
            log.flush()
            creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            process = subprocess.Popen(command, cwd=config.ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, creationflags=creation_flags)
            started = time.monotonic()
            last_heartbeat = 0.0
            last_progress = 0.0
            write_json(
                folder / "heartbeat.json",
                {"status": "running", "worker_pid": process.pid, "updated_at": now()},
            )
            try:
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    if elapsed - last_heartbeat >= 15:
                        write_json(
                            folder / "heartbeat.json",
                            {
                                "status": "running",
                                "worker_pid": process.pid,
                                "elapsed_seconds": int(elapsed),
                                "updated_at": now(),
                            },
                        )
                        last_heartbeat = elapsed
                    if elapsed - last_progress >= 60:
                        progress = read_json(folder / "progress.json", {})
                        print(
                            f"  {family.name}/{method}/{scenario}, seed {seed}: "
                            f"{progress.get('status', 'starting')}, "
                            f"update {progress.get('update', 0)}/{progress.get('updates', '?')}, "
                            f"{progress.get('timesteps', 0):,} steps, {int(elapsed)}s elapsed",
                            flush=True,
                        )
                        last_progress = elapsed
                    if elapsed > config.TRAINING_JOB_TIMEOUT_SECONDS:
                        process.kill()
                        process.wait()
                        last_error = "registered training timeout"
                        write_json(
                            folder / "heartbeat.json",
                            {"status": "timed_out", "worker_pid": process.pid, "updated_at": now()},
                        )
                        break
                    time.sleep(2)
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                event(
                    run,
                    "training_job_interrupted",
                    f"Interrupted {family.name} {method} on {scenario}, seed {seed}; worker stopped",
                    family=family.name,
                    method=method,
                    scenario=scenario,
                    seed=seed,
                )
                write_json(
                    folder / "progress.json",
                    {
                        "status": "interrupted",
                        "message": "Training worker stopped safely before completing this job.",
                        "updated_at": now(),
                    },
                )
                write_json(
                    folder / "heartbeat.json",
                    {"status": "interrupted", "worker_pid": process.pid, "updated_at": now()},
                )
                raise
            if process.returncode == 0 and (folder / "result.json").is_file():
                result = read_json(folder / "result.json", {})
                if not _job_matches(result, family, scenario, seed, mode, reward_hash, accelerator):
                    raise RuntimeError(f"Worker produced an incompatible result: {folder}")
                write_json(
                    folder / "heartbeat.json",
                    {"status": "complete", "worker_pid": process.pid, "updated_at": now()},
                )
                event(run, "training_job_completed", f"Completed {family.name} {method} on {scenario}, seed {seed}", family=family.name, phase=phase, method=method, scenario=scenario, seed=seed, sparse_return=result["evaluation"]["task_mean"])
                print(f"Completed {phase} {family.name}/{method}/{scenario}, seed {seed}: sparse mean={result['evaluation']['task_mean']:.6g}", flush=True)
                return result
            last_error = last_error or f"worker exit code {process.returncode}"
            write_json(
                folder / "heartbeat.json",
                {
                    "status": "failed",
                    "worker_pid": process.pid,
                    "error": last_error,
                    "updated_at": now(),
                },
            )
        event(run, "training_job_retry", f"{family.name} {method} on {scenario} failed: {last_error}", family=family.name, method=method, scenario=scenario, seed=seed, attempt=attempt)
        print(f"Training attempt {attempt} failed for {family.name}/{method}/{scenario}, seed {seed}: {last_error}; log: {folder / 'worker.log'}", flush=True)
    raise RuntimeError(f"Training failed for {family.name} {method} on {scenario}, seed {seed}: {last_error}. See {folder / 'worker.log'}")


def _phase_seeds(phase):
    return config.HELD_OUT_SEEDS if phase == "held_out" else config.TRAIN_SEEDS


def _method_returns(run, family, phase, method, scenarios):
    return {
        scenario: float(np.mean([
            float(read_json(_job_folder(run, family, phase, method, scenario, seed) / "result.json", {})["evaluation"]["task_mean"])
            for seed in _phase_seeds(phase)
        ]))
        for scenario in scenarios
    }


def _available_returns(run, family, phase, method, scenarios):
    values = {}
    for scenario in scenarios:
        completed = []
        for seed in _phase_seeds(phase):
            result = read_json(_job_folder(run, family, phase, method, scenario, seed) / "result.json", {})
            if result:
                completed.append(float(result["evaluation"]["task_mean"]))
        values[scenario] = float(np.mean(completed)) if len(completed) == len(_phase_seeds(phase)) else None
    return values


def _write_development_table(run: Run, family, ranking=None):
    root = run.folder / "family"
    baseline = _available_returns(run, family, "development", "native", family.development)
    baseline_complete = all(value is not None for value in baseline.values())
    rows = [{
        "method": "native", **baseline,
        "mean_sparse_return": float(np.mean(list(baseline.values()))) if baseline_complete else None,
        "aggregate_score": None,
        "status": "complete" if baseline_complete else "training",
    }]
    scores = {row["candidate"]: row for row in ranking or []}
    for candidate in config.CANDIDATES:
        summary = read_json(root / "candidates" / candidate / "summary.json", {})
        candidate_returns = summary.get("scenario_sparse_returns") or _available_returns(run, family, "development", candidate, family.development)
        measured = sum(value is not None for value in candidate_returns.values())
        designed = _design_complete(root / "candidates" / candidate)
        rows.append({
            "method": candidate,
            **{scenario: candidate_returns.get(scenario) for scenario in family.development},
            "mean_sparse_return": summary.get("mean_sparse_return"),
            "aggregate_score": scores.get(candidate, summary).get("aggregate_score"),
            "status": "complete" if summary else "training" if measured else "designed" if designed else "waiting",
        })
    write_json(root / "development_table.json", {"rows": rows, "updated_at": now()})
    fields = ["method", *family.development, "mean_sparse_return", "aggregate_score", "status"]
    with (root / "development_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_development_baseline(run, family):
    if config.EXPERIMENT != "exp1":
        from common.comparison import reuse_development_baseline
        reuse_development_baseline(run, family)
    for scenario in family.development:
        for seed in config.TRAIN_SEEDS:
            _run_training_job(run, family, "development", "native", scenario, seed, "native", None)
            _write_development_table(run, family)
    baseline = _method_returns(run, family, "development", "native", family.development)
    write_json(run.folder / "family" / "baseline_summary.json", {
        "family": family.name, "scenario_sparse_returns": baseline,
        "mean_sparse_return": float(np.mean(list(baseline.values()))), "seeds": list(config.TRAIN_SEEDS),
    })
    _write_development_table(run, family)
    return baseline


def _train_candidate(run, family, candidate, baseline):
    source = run.folder / "family" / "candidates" / candidate / "reward.py"
    if not source.is_file():
        raise RuntimeError(f"Candidate source is missing: {source}")
    expected_hash = read_json(source.parent / "design.json", {}).get("reward_hash")
    if not expected_hash or file_hash(source) != expected_hash:
        raise RuntimeError(f"Registered candidate source changed: {source}")
    for scenario in family.development:
        for seed in config.TRAIN_SEEDS:
            from common.learning_probes import reuse_learning_probe
            reuse_learning_probe(run, family, candidate, scenario, seed, source,
                                 _job_folder(run, family, "development", candidate, scenario, seed))
            _run_training_job(run, family, "development", candidate, scenario, seed, "candidate", source)
            if file_hash(source) != expected_hash:
                raise RuntimeError(f"Candidate source changed during training: {source}")
            _write_development_table(run, family)
    returns = _method_returns(run, family, "development", candidate, family.development)
    summary = {
        **rank_candidates({candidate: returns}, baseline)[0],
        "family": family.name, "reward_hash": expected_hash,
        "seeds": list(config.TRAIN_SEEDS), "completed_at": now(),
    }
    write_json(source.parent / "summary.json", summary)
    _write_development_table(run, family)
    return summary


def _select_and_lock(run, family, baseline):
    print(f"Selecting {family.name} reward from development-only scores", flush=True)
    root = run.folder / "family"
    designs = [read_json(root / "candidates" / candidate / "design.json", {}) for candidate in config.CANDIDATES]
    source_hashes = [design.get("reward_hash") for design in designs]
    program_hashes = [design.get("normalized_program_hash") for design in designs]
    if None in source_hashes or len(set(source_hashes)) != len(config.CANDIDATES):
        raise RuntimeError("Final selection requires six exact-source-unique rewards")
    if None in program_hashes or len(set(program_hashes)) != len(config.CANDIDATES):
        raise RuntimeError("Final selection requires six normalized-program-unique rewards")
    candidates = {candidate: read_json(root / "candidates" / candidate / "summary.json", {})["scenario_sparse_returns"] for candidate in config.CANDIDATES}
    ranking = rank_candidates(candidates, baseline)
    selected = ranking[0]["candidate"]
    source = root / "candidates" / selected / "reward.py"
    expected = read_json(source.parent / "summary.json", {}).get("reward_hash")
    if not expected or file_hash(source) != expected:
        raise RuntimeError(f"Selected candidate source changed after evaluation: {source}")
    frozen = root / "frozen_reward.py"
    shutil.copy2(source, frozen)
    digest = file_hash(frozen)
    write_json(root / "selection.json", {
        "family": family.name, "selected_candidate": selected, "selected_reward_hash": digest,
        "ranking": ranking, "selection_uses_held_out": False, "selected_at": now(),
    })
    _write_development_table(run, family, ranking)
    update(
        run,
        reward_locked=True,
        held_out_revealed=False,
        phase="reward_locked",
        activity=f"Locked {family.name} {selected}; no further reward design is permitted",
        candidate=selected,
        current_job=None,
    )
    event(run, "reward_locked", f"Mechanically selected and froze {family.name} {selected}", family=family.name, candidate=selected, reward_hash=digest)
    print(f"Selected and froze {family.name}/{selected}: {digest}", flush=True)
    return selected, frozen


def _per_seed_returns(run, family, method):
    values = {}
    for scenario in family.held_out:
        values[scenario] = []
        for seed in config.HELD_OUT_SEEDS:
            result = read_json(_job_folder(run, family, "held_out", method, scenario, seed) / "result.json", {})
            if result:
                values[scenario].append({"seed": seed, "task_mean": result["evaluation"]["task_mean"]})
    return values


def _write_held_out_table(run, family, selected, expected_hash, reference=None, complete=False):
    """Publish partial cells after every job; means require all registered seeds."""
    candidate = _available_returns(run, family, "held_out", selected, family.held_out)
    per_seed = {selected: _per_seed_returns(run, family, selected)}
    if reference:
        native = reference["native_sparse_returns"]
        garden = reference["selected_sparse_returns"]
        old_selected = reference["selected_candidate"]
        per_seed["native"] = reference.get("per_seed", {}).get("native", {})
        per_seed["garden_reference"] = reference.get("per_seed", {}).get(old_selected, {})
        rows = [{"method": "native", **native}, {"method": "garden_reference", **garden}, {"method": selected, **candidate}]
    else:
        native = _available_returns(run, family, "held_out", "native", family.held_out)
        per_seed["native"] = _per_seed_returns(run, family, "native")
        rows = [{"method": "native", **native}, {"method": selected, **candidate}]
    summary = {
        "family": family.name, "experiment": config.EXPERIMENT,
        "status": "complete" if complete else "running",
        "selected_candidate": selected, "frozen_reward_hash": expected_hash,
        "evaluation_seeds": list(config.HELD_OUT_SEEDS), "per_seed": per_seed,
        "native_sparse_returns": native, "selected_sparse_returns": candidate,
        "paired_improvements": {
            scenario: candidate[scenario] - native[scenario] if candidate[scenario] is not None and native[scenario] is not None else None
            for scenario in family.held_out
        },
        "rows": rows, "llm_calls_after_lock": _llm_calls_after_lock(run, family.name),
        "updated_at": now(),
    }
    if reference:
        summary["garden_reference_sparse_returns"] = garden
        summary["garden_reference_selected_candidate"] = old_selected
    if complete:
        summary["completed_at"] = now()
    write_json(run.folder / "family" / "held_out_table.json", summary)
    return summary


def _held_out(run, family, selected, frozen):
    suite_held_out_barrier(run)
    root = run.folder / "family"
    expected_hash = read_json(root / "selection.json", {})["selected_reward_hash"]
    if file_hash(frozen) != expected_hash:
        raise RuntimeError(f"Frozen reward hash changed for {family.name}")
    update(run, held_out_revealed=True, phase="held_out", activity=f"Evaluating frozen {family.name} reward with fresh held-out policies")
    print(f"Held-out evaluation started: {family.name}/{selected}, frozen reward {expected_hash}", flush=True)
    reference = None
    if config.EXPERIMENT != "exp1":
        from common.comparison import reference_held_out
        reference = reference_held_out(run, family)
    _write_held_out_table(run, family, selected, expected_hash, reference)
    for scenario in family.held_out:
        for seed in config.HELD_OUT_SEEDS:
            if config.EXPERIMENT == "exp1":
                _run_training_job(run, family, "held_out", "native", scenario, seed, "native", None)
                _write_held_out_table(run, family, selected, expected_hash, reference)
            if file_hash(frozen) != expected_hash:
                raise RuntimeError("Frozen reward changed during held-out evaluation")
            _run_training_job(run, family, "held_out", selected, scenario, seed, "candidate", frozen)
            if file_hash(frozen) != expected_hash:
                raise RuntimeError("Frozen reward changed during held-out evaluation")
            _write_held_out_table(run, family, selected, expected_hash, reference)
    llm_calls = _llm_calls_after_lock(run, family.name)
    if llm_calls:
        raise RuntimeError(f"Information-boundary violation: {llm_calls} LLM events after {family.name} lock")
    summary = _write_held_out_table(run, family, selected, expected_hash, reference, complete=True)
    event(run, "held_out_completed", f"Completed native and frozen-reward held-out evaluation for {family.name}", family=family.name, selected=selected)
    print(f"Held-out evaluation complete: {family.name}/{selected}; results: {root / 'held_out_table.json'}", flush=True)
    return summary


def run_family(run, family, phase="study"):
    if phase not in {"study", "search", "evaluate"}:
        raise ValueError("phase must be study, search, or evaluate")
    root = run.folder / "family"
    root.mkdir(parents=True, exist_ok=True)
    selection = read_json(root / "selection.json", {})
    if selection:
        # A resumed held-out phase must never re-enter the reward designer.
        baseline = read_json(root / "baseline_summary.json", {})["scenario_sparse_returns"]
        selected, frozen = selection["selected_candidate"], root / "frozen_reward.py"
        if not frozen.is_file() or file_hash(frozen) != selection["selected_reward_hash"]:
            raise RuntimeError("Frozen reward is missing or changed; design cannot resume after selection.")
        update(run, reward_locked=True)
        print(f"Using frozen {family.name}/{selected}; reward design remains locked", flush=True)
    else:
        if phase == "evaluate":
            raise RuntimeError("Evaluation requires an already selected and frozen reward; design will not run.")
        update(run, family=family.name, candidate=None, phase="development_baseline", activity=f"Preparing {family.name} native development baselines", current_job=None)
        print(f"Preparing {family.name} native development baselines", flush=True)
        baseline = _run_development_baseline(run, family)
        for generation, pair in enumerate(config.GENERATIONS, start=1):
            # Commit both programs before formal assessment. Design-time probes
            # are private to each candidate; workspace construction hides them
            # from its same-generation sibling.
            for candidate, position in zip(pair, ("a", "b")):
                if not _design_complete(root / "candidates" / candidate):
                    print(f"Design started: {family.name}/{candidate}, generation {generation}", flush=True)
                    design_candidate(run, family, candidate, generation, position)
                    _write_development_table(run, family)
                    print(f"Design complete: {family.name}/{candidate}", flush=True)
            print(f"Evaluating generation {generation}: {family.name}, {', '.join(pair)}", flush=True)
            for candidate in pair:
                _train_candidate(run, family, candidate, baseline)
            print(f"Generation {generation} evaluation complete: {family.name}", flush=True)
        selected, frozen = _select_and_lock(run, family, baseline)
    if phase == "search":
        write_json(root / "search_summary.json", {
            "family": family.name, "experiment": config.EXPERIMENT, "status": "search_complete",
            "selection": read_json(root / "selection.json", {}), "baseline": baseline,
            "llm_usage": _llm_usage(run), "training_job_counts": _training_counts(run), "completed_at": now(),
        })
        return
    suite_held_out_barrier(run)
    held_out = read_json(root / "held_out_table.json", {})
    if held_out.get("status") != "complete":
        held_out = _held_out(run, family, selected, frozen)
    write_json(root / "family_summary.json", {
        "family": family.name, "status": "complete", "baseline": baseline,
        "experiment": config.EXPERIMENT, "search_replicate": config.SEARCH_REPLICATE,
        "selection": read_json(root / "selection.json", {}), "held_out": held_out,
        "llm_usage": _llm_usage(run), "training_job_counts": _training_counts(run), "completed_at": now(),
    })
    event(run, "family_completed", f"Completed the registered {family.name} study", family=family.name)


def run_study(run: Run, family: config.FamilyConfig, phase="study"):
    if run.family_name != family.name:
        raise RuntimeError("Run family does not match the requested study family")
    if not (run.folder / "family" / "family_summary.json").is_file():
        run_family(run, family, phase)
    if phase == "search":
        update(run, status="search_complete", phase="search_complete", reward_locked=True,
               activity=f"{family.name} {config.EXPERIMENT} search complete; frozen reward awaits evaluation",
               current_job=None, search_completed_at=now())
        event(run, "search_completed", "Reward selected and frozen; held-out results remain unrevealed", family=family.name)
        return
    update(
        run,
        status="complete",
        phase="complete",
        activity=f"{family.name} {config.EXPERIMENT} reward search and held-out comparison are complete",
        candidate=None,
        current_job=None,
        completed_at=now(),
    )
    event(run, "study_completed", f"Completed the registered {family.name} study", family=family.name)


def _llm_usage(run):
    totals = {"requests": 0, "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "tool_calls": 0,
              "learning_probe_requests": 0, "unreadable_event_lines": 0}
    if run.events_file.is_file():
        for line in run.events_file.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A power cut may leave a partial append. Keep usable evidence
                # and disclose incomplete accounting instead of losing results.
                totals["unreadable_event_lines"] += 1
                continue
            for key, value in row.get("usage", {}).items():
                if key in totals:
                    totals[key] += int(value or 0)
            if row.get("kind") in {"llm_shell_command", "llm_learning_probe_call", "llm_submit_candidate_call"}:
                totals["tool_calls"] += 1
            if row.get("kind") == "llm_learning_probe_call":
                totals["learning_probe_requests"] += 1
    return totals


def _training_counts(run):
    counts = {"development": 0, "held_out": 0, "reused_native_development": 0,
              "development_assessments_reused_from_probes": 0, "charged_learning_probes": 0}
    for phase in ("development", "held_out"):
        for result in (run.folder / "family" / phase).glob("*/*/seed_*/result.json"):
            if config.EXPERIMENT != "exp1" and phase == "development" and result.parents[2].name == "native":
                counts["reused_native_development"] += 1
            elif phase == "development" and (result.parent / "probe_reuse.json").is_file():
                counts["development_assessments_reused_from_probes"] += 1
            else:
                counts[phase] += 1
    for ledger in (run.folder / "learning_probes").glob("*/ledger.json"):
        counts["charged_learning_probes"] += len(read_json(ledger, {}).get("attempts", []))
    counts["total_new_trainings"] = counts["development"] + counts["held_out"] + counts["charged_learning_probes"]
    return counts
