"""Run the controlled-study sequence with explicit reference bindings.

Each child executes one stage of its registered run. The queue adds ordering,
provenance and a barrier that freezes all three SMAX winners before testing.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

import bootstrap  # noqa: F401
import config
from common.records import (
    Run, _run_id, compatible, create_run, dependency_versions, file_hash,
    manifest_hash, now, planned_jobs, read_json, scientific_source_manifest,
    suite_held_out_barrier, write_json,
)


MEMBERS = (
    ("smax_exp1", "exp1", "SMAX"),
    ("smax_exp2", "exp2", "SMAX"),
    ("smax_exp3", "exp3", "SMAX"),
    ("overcooked_exp1", "exp1", "Overcooked"),
)
STAGES = (
    ("smax_exp1_search", "smax_exp1", "search"),
    ("smax_exp2_search", "smax_exp2", "search"),
    ("smax_exp3_search", "smax_exp3", "search"),
    ("smax_exp1_evaluate", "smax_exp1", "evaluate"),
    ("smax_exp2_evaluate", "smax_exp2", "evaluate"),
    ("smax_exp3_evaluate", "smax_exp3", "evaluate"),
    ("overcooked_exp1_study", "overcooked_exp1", "study"),
)

CONDITION_NAMES = {
    "exp1": "GARDEN + native baseline",
    "exp2": "Fixed-diagnostics control",
    "exp3": "Eureka-inspired control",
}

STAGE_NAMES = {
    "smax_exp1_search": "SMAX: native baseline and GARDEN reward search",
    "smax_exp2_search": "SMAX: fixed-diagnostics reward search",
    "smax_exp3_search": "SMAX: Eureka-inspired reward search",
    "smax_exp1_evaluate": "SMAX: frozen GARDEN and native held-out evaluation",
    "smax_exp2_evaluate": "SMAX: frozen fixed-diagnostics held-out evaluation",
    "smax_exp3_evaluate": "SMAX: frozen Eureka-inspired held-out evaluation",
    "overcooked_exp1_study": "Overcooked: native baseline, GARDEN search and held-out evaluation",
}


@contextmanager
def experiment_mode(experiment):
    """Registration uses the same dynamic protocol as each subprocess."""
    previous = config.EXPERIMENT
    config.EXPERIMENT = experiment
    try:
        yield
    finally:
        config.EXPERIMENT = previous


def family_plan(family):
    """Count training jobs for the currently selected condition; run nothing."""
    development_jobs = len(config.CANDIDATES) * len(family.development) * len(config.TRAIN_SEEDS)
    baseline_jobs = len(family.development) * len(config.TRAIN_SEEDS) if config.EXPERIMENT == "exp1" else 0
    held_out_jobs = len(family.held_out) * len(config.HELD_OUT_SEEDS) * (2 if config.EXPERIMENT == "exp1" else 1)
    batch = family.learner.num_envs * family.learner.rollout_steps
    return {
        "experiment": config.EXPERIMENT, "family": family.name, "search_replicate": config.SEARCH_REPLICATE,
        "model": config.MODEL, "candidates": len(config.CANDIDATES),
        "development_scenarios": list(family.development), "held_out_scenarios": list(family.held_out),
        "search_training_seeds": list(config.TRAIN_SEEDS), "held_out_training_seeds": list(config.HELD_OUT_SEEDS),
        "baseline_development_jobs": baseline_jobs, "candidate_development_jobs": development_jobs,
        "held_out_jobs": held_out_jobs, "total_new_training_jobs": planned_jobs(family),
        "maximum_optional_probe_jobs": len(config.CANDIDATES) * config.LEARNING_PROBE_JOBS,
        "maximum_new_training_jobs": planned_jobs(family) + len(config.CANDIDATES) * config.LEARNING_PROBE_JOBS,
        "actual_steps_per_job": (family.timesteps // batch) * batch,
        "reuses_frozen_exp1_native_reference": config.EXPERIMENT != "exp1",
        "output_root": str(config.ROOT / "results" / config.EXPERIMENT / family.cli_name / f"replicate_{config.SEARCH_REPLICATE}"),
    }


def plan():
    members = []
    for key, experiment, family_name in MEMBERS:
        family = config.FAMILY_BY_NAME[family_name]
        with experiment_mode(experiment):
            details = family_plan(family)
        assessments = details["total_new_training_jobs"]
        probes = details["maximum_optional_probe_jobs"]
        members.append({"key": key, "assessment_jobs": assessments,
                        "maximum_probe_jobs": probes, "maximum_jobs": assessments + probes})
    return {
        "members": members, "maximum_training_jobs": sum(item["maximum_jobs"] for item in members),
        "stages": [{"id": key, "member": member, "phase": phase} for key, member, phase in STAGES],
        "same_reward_exact_probe_reuse": True, "automatic_failed_stage_retries": 0,
        "search_training_seeds": list(config.TRAIN_SEEDS), "held_out_training_seeds": list(config.HELD_OUT_SEEDS),
    }


def register_suite():
    """Register every run before execution; never adopt an old latest run."""
    identifier = _run_id()
    folder = config.ROOT / "results" / "suites" / identifier
    folder.mkdir(parents=True)
    members = []
    for key, experiment, family_name in MEMBERS:
        family = config.FAMILY_BY_NAME[family_name]
        with experiment_mode(experiment):
            run = create_run(family, suite={"suite_id": identifier, "folder": str(folder), "member": key})
        manifest = read_json(run.folder / "manifest.json")
        members.append({"key": key, "experiment": experiment, "family": family_name,
                        "run_id": run.id, "run_folder": str(run.folder),
                        "manifest_hash": file_hash(run.folder / "manifest.json"),
                        "protocol_hash": manifest["protocol_hash"]})
    registration = {
        "members": members, "stages": plan()["stages"], "plan": plan(),
        "scientific_source_hash": manifest_hash(scientific_source_manifest()),
        "dependencies": dependency_versions(),
    }
    write_json(folder / "suite.json", {
        "schema_version": 1, "suite_id": identifier, "created_at": now(), "updated_at": now(),
        "registration": registration, "registration_hash": manifest_hash(registration),
        "status": "registered", "current_stage": None, "current_member": None,
        "stages_status": {stage["id"]: {"status": "pending"} for stage in registration["stages"]},
        "frozen_rewards": {}, "automatic_retries": 0,
    })
    return folder


def load_suite(folder):
    path = folder / "suite.json"
    suite = read_json(path, {})
    registration = suite.get("registration", {})
    if not suite or manifest_hash(registration) != suite.get("registration_hash"):
        raise RuntimeError("Suite registration is missing or changed.")
    if registration.get("scientific_source_hash") != manifest_hash(scientific_source_manifest()):
        raise RuntimeError("Registered scientific source changed; the suite will not silently restart.")
    if registration.get("dependencies") != dependency_versions():
        raise RuntimeError("Registered dependencies changed.")
    for member in registration["members"]:
        run = Run(member["run_id"], Path(member["run_folder"]), member["family"])
        manifest = read_json(run.folder / "manifest.json", {})
        if file_hash(run.folder / "manifest.json") != member["manifest_hash"]:
            raise RuntimeError(f"Member manifest changed: {member['key']}")
        if manifest.get("suite") != {"suite_id": suite["suite_id"], "folder": str(folder), "member": member["key"]}:
            raise RuntimeError(f"Member suite binding changed: {member['key']}")
        with experiment_mode(member["experiment"]):
            if not compatible(run):
                raise RuntimeError(f"Member cannot resume its registered configuration: {member['key']}")
    return suite


def _freeze_record(member):
    root = Path(member["run_folder"]) / "family"
    selection = read_json(root / "selection.json", {})
    frozen = root / "frozen_reward.py"
    if not frozen.is_file() or file_hash(frozen) != selection.get("selected_reward_hash"):
        raise RuntimeError(f"Selected reward is missing or changed: {member['key']}")
    if selection.get("selection_uses_held_out") is not False:
        raise RuntimeError("Selection must explicitly exclude held-out evidence.")
    return {"candidate": selection["selected_candidate"], "reward_hash": file_hash(frozen),
            "selection_hash": file_hash(root / "selection.json")}


def _save(folder, suite, **changes):
    suite.update(changes)
    suite["updated_at"] = now()
    write_json(folder / "suite.json", suite)


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def run_suite(folder, *, resume=False):
    suite = load_suite(folder)
    if suite["status"] == "complete":
        print(f"Suite already complete: {folder}")
        return
    if suite["status"] != "registered" and not resume:
        raise RuntimeError("This queue was already started. Inspect its records and use --resume explicitly.")
    active = suite.get("stages_status", {}).get(suite.get("current_stage"), {})
    if active.get("worker_pid") and active.get("status") != "complete":
        if suite.get("controller_host") != socket.gethostname():
            raise RuntimeError("The last worker belongs to another host; inspect it before resuming.")
        if _pid_alive(active["worker_pid"]):
            raise RuntimeError(f"A previously launched stage worker is still running: PID {active['worker_pid']}. No duplicate will be started.")
    # The directory lock is held only by the controller. A crashed controller
    # leaves an exact PID record; explicit resume may remove that stale lock.
    lock = folder / "controller.lock"
    if lock.exists():
        owner = read_json(lock, {})
        if owner.get("host") != socket.gethostname():
            raise RuntimeError("Suite lock belongs to another host; inspect it before resuming.")
        if not _pid_alive(owner["pid"]):
            if not resume:
                raise RuntimeError("Stale suite lock; explicit --resume is required.")
            lock.unlink()
        else:
            raise RuntimeError(f"Suite controller is already running: PID {owner['pid']}")
    with lock.open("x", encoding="utf-8") as handle:
        import json
        json.dump({"pid": os.getpid(), "host": socket.gethostname()}, handle)
    members = {item["key"]: item for item in suite["registration"]["members"]}
    active_stage = None
    try:
        for stage in suite["registration"]["stages"]:
            member = members[stage["member"]]
            folder_run = Path(member["run_folder"])
            run = Run(member["run_id"], folder_run, member["family"])
            if suite["stages_status"][stage["id"]]["status"] == "complete":
                if suite.get("frozen_rewards", {}).get(member["key"]) != _freeze_record(member):
                    raise RuntimeError("A previously completed stage's frozen evidence changed.")
                continue
            active_stage = stage["id"]
            if stage["phase"] == "evaluate":
                suite_held_out_barrier(run)
            command = [sys.executable, "-B", "-m", "common.worker",
                       "--experiment", member["experiment"], "--family", member["family"],
                       "--phase", stage["phase"], "--run", str(folder_run)]
            stage_state = {"status": "running", "started_at": now(), "command": command}
            suite["stages_status"][active_stage] = stage_state
            _save(folder, suite, status="running", current_stage=active_stage,
                  current_member=member["key"], controller_pid=os.getpid(), controller_host=socket.gethostname(), error=None)
            environment = os.environ.copy()
            environment.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            print(f"Starting {STAGE_NAMES[active_stage]}\nRecords: {folder_run}", flush=True)
            with (folder / f"{active_stage}.log").open("a", encoding="utf-8") as log, \
                    (folder / f"{active_stage}.log").open("r", encoding="utf-8", errors="replace") as output:
                output.seek(0, os.SEEK_END)
                # A terminal Ctrl+C reaches only the queue. It forwards one
                # SIGINT, giving the child time to reap its current worker.
                process_options = ({"creationflags": subprocess.CREATE_NO_WINDOW}
                                   if os.name == "nt" else {"start_new_session": True})
                process = subprocess.Popen(command, cwd=config.ROOT, env=environment,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           **process_options)
                stage_state["worker_pid"] = process.pid
                _save(folder, suite)
                try:
                    # Keep the complete stage log and echo new child output.
                    while True:
                        try:
                            code = process.wait(timeout=1)
                        except subprocess.TimeoutExpired:
                            print(output.read(), end="", flush=True)
                            continue
                        print(output.read(), end="", flush=True)
                        break
                except BaseException:
                    # Let the ordinary controller stop and reap its current
                    # learner/probe before the queue exits (WSL/Linux).
                    if os.name != "nt":
                        process.send_signal(signal.SIGINT)
                    else:
                        process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    print(output.read(), end="", flush=True)
                    raise
            if code:
                raise RuntimeError(f"{active_stage} stopped with exit code {code}; inspect {active_stage}.log. No retry was started.")
            expected = "search_complete" if stage["phase"] == "search" else "complete"
            state = read_json(run.state_file, {})
            if state.get("status") != expected:
                raise RuntimeError(f"{active_stage} exited without its required {expected} record.")
            frozen = _freeze_record(member)
            accepted = suite["frozen_rewards"].get(member["key"])
            if accepted and accepted != frozen:
                raise RuntimeError("Frozen reward or selection changed between search and evaluation.")
            suite["frozen_rewards"][member["key"]] = frozen
            stage_state.update(status="complete", completed_at=now())
            _save(folder, suite)
            print(f"Completed {STAGE_NAMES[active_stage]}", flush=True)
        _save(folder, suite, status="complete", completed_at=now(), current_stage=None, current_member=None)
        print(f"Suite complete: {folder}")
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        if active_stage:
            suite["stages_status"][active_stage].update(status=status, stopped_at=now())
        _save(folder, suite, status=status, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        lock.unlink(missing_ok=True)
