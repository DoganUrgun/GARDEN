"""Bounded, development-only learning experiments shared by all three methods.

The engineer chooses a standalone reward and a development scenario. Everything
else is the registered learner protocol. These jobs supply evidence, never a
selection score. An exact, unchanged probe may later serve as its candidate's
development assessment; that is reused computation, not independent evidence.
"""

from __future__ import annotations

import asyncio
import csv
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

import config
from common.hardware import training_worker_environment
from common.records import (Run, dependency_versions, event, file_hash, manifest_hash,
                            now, read_json, scientific_source_manifest, update, write_json)
from common.workspace import _redact_controller_paths, protected_manifest


def _validate(path: Path, family_name: str) -> dict:
    # Import lazily: controller inspection and mocked tests need no JAX backend.
    from validate_reward import validate
    return validate(path, family_name)


def _hashes(folder: Path) -> dict[str, str]:
    return {path.relative_to(folder).as_posix(): file_hash(path)
            for path in sorted(folder.rglob("*")) if path.is_file()}


def _source_signature() -> dict:
    """A probe is reusable only under the same scientific code and packages."""
    return {"source_hash": manifest_hash(scientific_source_manifest()),
            "dependencies": dependency_versions()}


def _network_config(family, scenario):
    # Keep controller startup and offline protocol tests free of JAX imports.
    import bootstrap  # noqa: F401
    from common.families import adapter_for
    adapter = adapter_for(family.name, family)
    return adapter.network_config(adapter.make_environment(scenario))


def _verify_training(folder: Path, family, scenario, seed, reward_hash, accelerator) -> dict:
    """Verify settings and the actual evidence before accepting or reusing a job."""
    from common.study import _job_matches
    result = read_json(folder / "result.json", {})
    if not _job_matches(result, family, scenario, seed, "candidate", reward_hash, accelerator):
        raise RuntimeError("Probe worker result does not match the registered training protocol")
    batch = family.learner.num_envs * family.learner.rollout_steps
    if result.get("actual_timesteps") != max(1, family.timesteps // batch) * batch:
        raise RuntimeError("Probe worker did not complete its fixed training budget")
    if result.get("network") != _network_config(family, scenario):
        raise RuntimeError("Probe network differs from the registered environment network")
    evaluation = result.get("evaluation", {})
    if evaluation.get("native_shaping_return_mean") != 0:
        raise RuntimeError("Generated-reward probe must explicitly record zero native shaping")
    task = evaluation.get("task_mean")
    if not isinstance(task, (float, int)) or not math.isfinite(task):
        raise RuntimeError("Probe sparse task return is missing or non-finite")
    with (folder / "evaluation_episodes.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    returns = [float(row["task_return"]) for row in rows]
    if (len(returns) != family.evaluation_episodes
            or any(not math.isfinite(value) for value in returns)
            or not math.isclose(sum(returns) / len(returns), task, abs_tol=1e-12)):
        raise RuntimeError("Probe evaluation episodes disagree with the registered sparse result")
    if family.name == "SMAX" and any(value not in (0.0, 1.0) for value in returns):
        raise RuntimeError("SMAX episode task returns must be binary")
    if any(float(row["native_shaping_return_mean"]) != 0 for row in rows):
        raise RuntimeError("Probe evaluation episodes contain native shaping")
    with (folder / "training_curve.csv").open(newline="", encoding="utf-8") as stream:
        curve = list(csv.DictReader(stream))
    if not curve or int(float(curve[-1]["timesteps"])) != result["actual_timesteps"]:
        raise RuntimeError("Probe learning curve is incomplete")
    for network in ("actor", "critic"):
        checkpoint = folder / "checkpoint" / f"{network}.msgpack"
        if (not checkpoint.is_file()
                or file_hash(checkpoint) != result.get("checkpoint_hashes", {}).get(network)):
            raise RuntimeError(f"Probe {network} checkpoint is missing or changed")
    if not (folder / "trajectories.jsonl").is_file():
        raise RuntimeError("Probe final trajectories are missing")
    # Snapshot contents are covered by the immutable artifact ledger. Require the
    # declared records too, so an incomplete worker cannot claim full evidence.
    snapshots = result.get("training_snapshots", [])
    reached = {value for snapshot in snapshots for value in snapshot.get("requested_fractions", [])}
    expected = set(config.SNAPSHOT_FRACTIONS) if config.SNAPSHOT_EPISODES else set()
    if reached != expected:
        raise RuntimeError("Probe intermediate snapshots are incomplete")
    for snapshot in snapshots:
        saved = (folder / snapshot["folder"]).resolve()
        if not saved.is_relative_to(folder.resolve()):
            raise RuntimeError("Probe snapshot path escapes its job")
        for name in ("metadata.json", "trajectories.jsonl", "checkpoint/actor.msgpack", "checkpoint/critic.msgpack"):
            if not (saved / name).is_file():
                raise RuntimeError(f"Probe snapshot artifact is missing: {name}")
    return result


def reuse_learning_probe(run: Run, family, candidate: str, scenario: str,
                         seed: int, reward_path: Path, destination: Path) -> bool:
    """Copy one exact successful probe into its final development assessment.

    This is deliberately candidate-local and development-only. It never resumes
    a checkpoint, pools observations, or transfers a trained policy to held-out
    scenarios. A provenance marker prevents reused jobs being counted twice.
    """
    expected = run.folder / "family" / "development" / candidate / scenario / f"seed_{seed}"
    if (scenario not in family.development or destination.resolve() != expected.resolve()
            or not destination.resolve().is_relative_to(run.folder.resolve())):
        raise ValueError("Probe reuse is restricted to its candidate's development job")
    root = run.folder / "learning_probes" / candidate
    ledger = read_json(root / "ledger.json", {})
    if not ledger:
        return False
    digest = file_hash(reward_path)
    matches = [row for row in ledger.get("attempts", [])
               if row.get("status") == "complete" and row.get("scenario") == scenario
               and row.get("seed") == seed and row.get("reward_hash") == digest]
    if not matches:
        return False
    settings = ledger.get("settings", {})
    accelerator = read_json(run.folder / "manifest.json", {}).get("accelerator", {})
    if (settings.get("version") != 2 or settings.get("candidate") != candidate
            or settings.get("family") != family.public_dict()
            or settings.get("source_signature") != _source_signature()
            or settings.get("accelerator") != accelerator):
        raise RuntimeError("Matching probe was produced under a different registered protocol")
    attempt = matches[-1]
    archive = root / attempt["id"]
    if not archive.resolve().is_relative_to(root.resolve()):
        raise RuntimeError("Probe ledger path escapes its candidate archive")
    if _hashes(archive) != attempt.get("artifact_hashes") or file_hash(archive / "reward.py") != digest:
        raise RuntimeError("Canonical probe artifacts changed; refusing reuse")
    training = archive / "training"
    _verify_training(training, family, scenario, seed, digest, accelerator)
    expected_files = _hashes(training)
    provenance = {
        "kind": "exact_learning_probe_reuse", "candidate": candidate,
        "probe": attempt["id"], "source": str(training.relative_to(run.folder)).replace("\\", "/"),
        "scenario": scenario, "seed": seed, "reward_hash": digest,
        "artifact_hashes": expected_files,
        "independent_confirmation": False, "additional_training_jobs": 0,
    }
    if destination.exists():
        marker = read_json(destination / "probe_reuse.json", {})
        if not marker:
            # A normally trained assessment is valid too; never overwrite it.
            return False
        observed = _hashes(destination)
        observed.pop("probe_reuse.json", None)
        if marker != provenance or observed != expected_files:
            raise RuntimeError("Reused probe assessment changed or has conflicting provenance")
        return True
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".probe-reuse-{uuid.uuid4().hex}"
    try:
        shutil.copytree(training, staging)
        if _hashes(staging) != expected_files:
            raise RuntimeError("Probe artifact copy failed verification")
        write_json(staging / "probe_reuse.json", provenance)
        staging.replace(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)  # Only this call's uniquely named staging directory.
    event(run, "training_job_reused", f"Reused exact {candidate} probe on {scenario}, seed {seed}",
          family=family.name, method=candidate, scenario=scenario, seed=seed,
          source_kind="learning_probe", probe=attempt["id"], additional_training_jobs=0)
    return True


class LearningProbes:
    """A persistent budget of fresh-policy jobs, with no configurable tool knobs."""

    def __init__(self, run: Run, family: config.FamilyConfig, candidate: str,
                 workspace: Path, *, max_jobs: int | None = None, seed: int | None = None,
                 timeout_seconds: float | None = None, evidence_mode: str = "autonomous"):
        max_jobs = config.LEARNING_PROBE_JOBS if max_jobs is None else max_jobs
        seed = config.TRAIN_SEEDS[0] if seed is None else seed
        timeout_seconds = config.LEARNING_PROBE_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        if not re.fullmatch(r"[A-Za-z0-9_-]+", candidate):
            raise ValueError("Candidate must be a simple identifier")
        if max_jobs < 1 or seed < 0 or timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
            raise ValueError("Probe budget, seed, or timeout is invalid")
        if run.family_name != family.name:
            raise ValueError("Probe family differs from its registered run")
        if evidence_mode not in {"autonomous", "fixed", "eureka"}:
            raise ValueError("Unknown registered probe evidence interface")
        self.study, self.family, self.candidate = run, family, candidate
        self.workspace = workspace.resolve(strict=True)
        self.root = run.folder / "learning_probes" / candidate
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.resolve().is_relative_to(run.folder.resolve()):
            raise ValueError("Probe archive escapes its run")
        self.ledger_path = self.root / "ledger.json"
        self.timeout_seconds = timeout_seconds
        self.seed = seed
        self.evidence_mode = evidence_mode
        self.expected = protected_manifest(self.workspace)
        registered = read_json(self.workspace / "workspace_manifest.json", {})
        if registered and registered.get("protected_files") != self.expected:
            raise RuntimeError("Protected reference or evidence changed")
        self.settings = {
            "version": 2, "evidence_mode": evidence_mode,
            "family": family.public_dict(), "candidate": candidate,
            "max_jobs": max_jobs, "seed": seed, "timeout_seconds": timeout_seconds,
            "snapshot_episodes": config.SNAPSHOT_EPISODES,
            "snapshot_fractions": list(config.SNAPSHOT_FRACTIONS) if config.SNAPSHOT_EPISODES else [],
            "evaluation_greedy": config.EVALUATION_GREEDY,
            "protected_files": self.expected,
            "accelerator": read_json(run.folder / "manifest.json", {}).get("accelerator", {}),
            "source_signature": _source_signature(),
        }
        if self.ledger_path.exists():
            self._ledger()
        else:
            write_json(self.ledger_path, {"settings": self.settings, "attempts": [], "created_at": now()})

    def _ledger(self) -> dict:
        ledger = read_json(self.ledger_path, {})
        if ledger.get("settings") != self.settings:
            raise RuntimeError("Registered probe settings changed; refusing to reset the budget")
        return ledger

    def usage(self) -> dict:
        attempts = self._ledger()["attempts"]
        calls = 0
        if self.study.events_file.is_file():
            for line in self.study.events_file.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                calls += row.get("kind") == "llm_learning_probe_call" and row.get("candidate") == self.candidate
        return {
            "max_jobs": self.settings["max_jobs"], "charged_jobs": len(attempts),
            "remaining_jobs": self.settings["max_jobs"] - len(attempts),
            "completed_jobs": sum(row["status"] == "complete" for row in attempts),
            "requested_timesteps_per_job": self.family.timesteps,
            "charged_requested_timesteps": len(attempts) * self.family.timesteps,
            "worker_seconds": sum(row.get("elapsed_seconds", 0) for row in attempts),
            "tool_calls": calls,
        }

    def instructions(self) -> str:
        evidence = ("complete raw training artifacts under scratch/learning_probes/"
                    if self.evidence_mode == "autonomous"
                    else "only the registered fixed diagnostic report, with no filesystem access")
        request = ("Choose a permitted standalone reward file inside this workspace and call run_learning_probe "
                   if self.evidence_mode == "autonomous"
                   else "Return action=probe with complete standalone source, a development scenario and a hypothesis ")
        return (
            f"Optional learning probes: at most {self.settings['max_jobs']} fresh-policy "
            f"development training jobs, each with the fixed {self.family.timesteps:,}-step "
            f"budget and seed {self.seed}. {request}"
            "to test your investigation hypothesis. Decide yourself whether and how to use this capability. The tool "
            f"returns {evidence}. "
            "The learner, evaluation, seed and budget cannot be changed. Failed or "
            "interrupted launched jobs still consume a slot. Repeating the exact reward "
            "and scenario returns the existing record, not another training. Probes are "
            "development evidence only. If your final source is byte-identical, its exact "
            "scenario/seed probe may supply that final development assessment without "
            "retraining; it is not an independent confirmation. Do not launch policy "
            "training through shell scripts or otherwise bypass this registered budget. "
            "Tool runtime counts toward the overall design time budget; it does not "
            "grant additional reasoning time."
        )

    def _check_boundary(self) -> None:
        state = read_json(self.study.state_file, {})
        if state.get("reward_locked") or state.get("held_out_revealed"):
            raise RuntimeError("Reward design is closed; learning probes are disabled")
        if protected_manifest(self.workspace) != self.expected:
            raise RuntimeError("Protected reference or evidence changed")
        if _source_signature() != self.settings["source_signature"]:
            raise RuntimeError("Scientific source or packages changed during probe design")

    def _reward_path(self, name: str) -> Path:
        relative = Path(name)
        if (relative.is_absolute() or PureWindowsPath(name).drive
                or ".." in relative.parts or ".." in PureWindowsPath(name).parts):
            raise ValueError("Reward must be a workspace-relative path without parent traversal")
        path = (self.workspace / relative).resolve(strict=True)
        if not path.is_relative_to(self.workspace) or not path.is_file() or path.suffix != ".py":
            raise ValueError("Reward must be a .py file inside this workspace")
        return path

    def _publish(self, attempt: dict) -> str | None:
        archive = self.root / attempt["id"]
        if not archive.is_dir():
            return None
        destination = self.workspace / "scratch" / "learning_probes" / attempt["id"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.resolve().is_relative_to(self.workspace):
            raise RuntimeError("Probe evidence destination escapes the workspace")
        if not destination.exists():
            shutil.copytree(archive, destination)
            _redact_controller_paths(destination, self.study)
        return destination.relative_to(self.workspace).as_posix()

    def _response(self, attempt: dict, *, reused: bool = False) -> str:
        payload = {
            "probe": attempt["id"], "status": attempt["status"], "reused": reused,
            "scenario": attempt["scenario"], "reward_hash": attempt["reward_hash"],
            "sparse_return": attempt.get("sparse_return"), "error": attempt.get("error"),
            "usage": self.usage(),
        }
        if self.evidence_mode == "autonomous":
            payload["artifacts"] = self._publish(attempt)
        elif attempt["status"] == "complete":
            from common.feedback import diagnostic
            payload["diagnostic"] = diagnostic(self.root / attempt["id"] / "training",
                                                components=self.evidence_mode == "eureka")
        else:
            # Controls can correct their reward and choose another permitted
            # probe, but do not gain arbitrary worker logs or artifact paths.
            payload["error"] = "Registered training did not complete; this job consumed one probe slot."
        response = json.dumps(payload, allow_nan=False)
        for path, label in ((self.study.folder, "<registered_run>"), (config.ROOT, "<project_root>")):
            for spelling in (str(path), path.as_posix()):
                response = response.replace(json.dumps(spelling)[1:-1], label)
        return response

    async def _stop(self, process) -> None:
        """Reap the worker and its group before returning control to the engineer."""
        if process.returncode is not None:
            await process.wait()
            return
        if os.name == "nt":
            # The production execution is WSL; taskkill also covers Windows children.
            killer = await asyncio.create_subprocess_exec(
                "taskkill", "/PID", str(process.pid), "/T", "/F",
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            await killer.wait()
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        try:
            await asyncio.wait_for(process.wait(), timeout=15)
        except asyncio.TimeoutError:
            if os.name == "nt":
                process.kill()
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            await process.wait()

    async def run(self, reward_file: str, scenario: str, hypothesis: str) -> str:
        """Execute one registered probe; cancellation propagates after safe cleanup."""
        event(self.study, "llm_learning_probe_call", "Reward engineer requested a learning probe",
              family=self.family.name, candidate=self.candidate,
              call_id=uuid.uuid4().hex, call_index=self.usage()["tool_calls"] + 1,
              reward_file=reward_file, scenario=scenario, hypothesis=hypothesis)
        # One outstanding call per candidate, including across controller processes.
        lock = self.root / ".active.lock"
        self._recover_dead_lock(lock)
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            return json.dumps({"status": "rejected", "error": "A probe is already active; inspect its record before retrying"})
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"controller_pid": os.getpid()}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            return await self._run(reward_file, scenario, hypothesis)
        finally:
            lock.unlink(missing_ok=True)

    def _recover_dead_lock(self, lock: Path) -> None:
        """Recover only provably dead POSIX processes; never refund a job slot."""
        if not lock.exists() or os.name == "nt":
            return
        try:
            owner = read_json(lock, {})
        except (json.JSONDecodeError, OSError):
            return  # An unknown owner is not permission to start another worker.

        def dead(pid):
            if not isinstance(pid, int) or pid <= 0:
                return False
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                pass
            return False

        if not dead(owner.get("controller_pid")):
            return
        ledger = self._ledger()
        active = [row for row in ledger["attempts"] if row["status"] == "running"]
        if any(not dead(row.get("worker_pid")) for row in active):
            return
        for attempt in active:
            attempt.update(status="interrupted", error="Controller and worker ended before result acceptance",
                           completed_at=now())
            archive = self.root / attempt["id"]
            if archive.is_dir():
                write_json(archive / "outcome.json", {key: value for key, value in attempt.items()
                                                       if key != "artifact_hashes"})
            attempt["artifact_hashes"] = _hashes(archive)
        write_json(self.ledger_path, ledger)
        lock.unlink()

    async def _run(self, reward_file: str, scenario: str, hypothesis: str) -> str:
        self._check_boundary()
        ledger = self._ledger()
        try:
            if scenario not in self.family.development:
                raise ValueError("Only this family's registered development scenarios are allowed")
            if not hypothesis.strip():
                raise ValueError("Provide a concise investigation hypothesis")
            source = self._reward_path(reward_file)
            digest = file_hash(source)
            for previous in ledger["attempts"]:
                if (previous["reward_hash"], previous["scenario"]) == (digest, scenario):
                    if previous["status"] == "running":
                        return json.dumps({"status": "rejected", "error": "An interrupted probe reservation already exists; no duplicate will be launched", "usage": self.usage()})
                    if _hashes(self.root / previous["id"]) != previous["artifact_hashes"]:
                        raise RuntimeError("Canonical probe artifacts changed")
                    return self._response(previous, reused=True)
            if len(ledger["attempts"]) >= self.settings["max_jobs"]:
                raise ValueError("The registered learning-probe budget is exhausted")
            try:
                validation = _validate(source, self.family.name)
            except Exception as error:
                self._check_boundary()
                raise ValueError(f"Reward validation failed: {type(error).__name__}: {error}") from error
            if not validation.get("valid") or file_hash(source) != digest:
                raise ValueError("Reward validation failed or its source changed during validation")
            if self.evidence_mode == "eureka" and not validation.get("has_reward_components"):
                raise ValueError("Eureka-inspired probes require additive reward_components for component feedback")
            self._check_boundary()
        except (ValueError, FileNotFoundError) as error:
            return json.dumps({"status": "rejected", "error": str(error), "usage": self.usage()})

        identifier = f"probe_{len(ledger['attempts']) + 1:03d}"
        if (self.root / identifier).exists():
            raise RuntimeError("Unregistered probe artifacts exist; refusing to overwrite them")
        attempt = {
            "id": identifier, "status": "running",
            "scenario": scenario, "reward_file": reward_file, "reward_hash": digest,
            "hypothesis": hypothesis, "started_at": now(), "seed": self.seed,
        }
        # Reserve durably BEFORE any worker launch. An exception never refunds it.
        ledger["attempts"].append(attempt)
        write_json(self.ledger_path, ledger)
        archive = self.root / attempt["id"]
        process = None
        started = time.monotonic()
        try:
            archive.mkdir(exist_ok=False)
            snapshot = archive / "reward.py"
            shutil.copy2(source, snapshot)
            if file_hash(snapshot) != digest:
                raise RuntimeError("Reward source changed before its snapshot was frozen")
            write_json(archive / "validation.json", validation)
            training = archive / "training"
            training.mkdir()
            command = [
                sys.executable, str(config.ROOT / "training.py"), "--family", self.family.name,
                "--scenario", scenario, "--seed", str(self.seed), "--mode", "candidate",
                "--reward", str(snapshot), "--output", str(training),
                "--timesteps", str(self.family.timesteps),
                "--eval-episodes", str(self.family.evaluation_episodes),
                "--trace-episodes", str(self.family.trace_episodes),
                "--snapshot-episodes", str(config.SNAPSHOT_EPISODES),
            ]
            write_json(archive / "request.json", {**attempt, "command": command, "settings": self.settings})
            update(self.study, phase="learning_probe", candidate=self.candidate,
                   current_job=str(training.relative_to(self.study.folder)),
                   activity=f"Reward engineer is testing {self.candidate} {attempt['id']} on {scenario}; fresh policy, fixed learner.")
            event(self.study, "llm_learning_probe_started", f"Started {attempt['id']} on {scenario}",
                  family=self.family.name, candidate=self.candidate, **attempt)
            with (training / "worker.log").open("x", encoding="utf-8") as log:
                options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
                launching = asyncio.create_task(asyncio.create_subprocess_exec(
                    *command, cwd=config.ROOT, env=training_worker_environment(),
                    stdout=log, stderr=asyncio.subprocess.STDOUT, **options,
                ))
                try:
                    process = await asyncio.shield(launching)
                except asyncio.CancelledError:
                    # Creation can complete after cancellation; obtain the PID so
                    # the outer cleanup cannot leave an untracked GPU worker.
                    process = await launching
                    raise
                attempt["worker_pid"] = process.pid
                write_json(self.ledger_path, ledger)
                write_json(training / "heartbeat.json", {"status": "running", "worker_pid": process.pid, "updated_at": now()})
                await asyncio.wait_for(process.wait(), timeout=self.timeout_seconds)
            if process.returncode != 0:
                raise RuntimeError(f"Training worker exited with code {process.returncode}; inspect training/worker.log")
            result = _verify_training(training, self.family, scenario, self.seed, digest, self.settings["accelerator"])
            task = result["evaluation"]["task_mean"]
            if file_hash(snapshot) != digest:
                raise RuntimeError("Frozen probe source changed during training")
            self._check_boundary()
            attempt.update(status="complete", sparse_return=task, actual_timesteps=result["actual_timesteps"])
        except BaseException as error:
            if process is not None:
                await self._stop(process)
            attempt.update(status="cancelled" if isinstance(error, asyncio.CancelledError) else "failed",
                           error="Training timed out" if isinstance(error, asyncio.TimeoutError) else f"{type(error).__name__}: {error}")
            if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
                raise
        finally:
            attempt.update(elapsed_seconds=time.monotonic() - started, completed_at=now())
            if protected_manifest(self.workspace) != self.expected:
                attempt.update(status="failed", error="Protected reference or evidence changed")
            if archive.is_dir():
                write_json(archive / "outcome.json", attempt)
                heartbeat = archive / "training" / "heartbeat.json"
                if heartbeat.parent.is_dir():
                    write_json(heartbeat, {"status": attempt["status"], "updated_at": now()})
            attempt["artifact_hashes"] = _hashes(archive)
            write_json(self.ledger_path, ledger)
            update(self.study, phase="reward_design", current_job=None,
                   activity=f"{self.candidate} learning probe {attempt['status']}; reward engineering may continue.")
            event(self.study, "llm_learning_probe_completed" if attempt["status"] == "complete" else "llm_learning_probe_failed",
                  f"{attempt['id']} {attempt['status']}", family=self.family.name,
                  candidate=self.candidate, **attempt)
        return self._response(attempt)
