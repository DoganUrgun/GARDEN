"""Append-friendly records for one selected environment-family study."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config
from common.hardware import probe_accelerator


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    """Replace a record atomically so interruption cannot leave partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest() -> dict[str, str]:
    # A source allowlist keeps environments, results and analysis artifacts out
    # of resume hashes. Adding a run must never invalidate a different run.
    roots = ("common", "Overcooked", "SMAX", "runtime", "prompts", "tests")
    files = [p for p in config.ROOT.iterdir() if p.is_file() and (p.suffix in {".py", ".txt"} or p.name == "PROTOCOL.md")]
    for name in roots:
        folder = config.ROOT / name
        if folder.is_dir():
            files.extend(folder.rglob("*"))
    allowed = {".py", ".md", ".txt", ".json"}
    values = {}
    for path in sorted(files):
        relative = path.relative_to(config.ROOT)
        if (
            not path.is_file()
            or any(part in {"__pycache__", ".pytest_cache"} for part in relative.parts)
            or path.suffix.lower() not in allowed
        ):
            continue
        values[relative.as_posix()] = file_hash(path)
    return values


def scientific_source_manifest() -> dict[str, str]:
    return {
        name: digest
        for name, digest in source_manifest().items()
        if name != "README.md" and not name.startswith("tests/")
    }


def manifest_hash(value: dict[str, str]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def dependency_versions() -> dict[str, str]:
    versions = {}
    for name in (
        "jax", "jaxlib", "flax", "optax", "chex", "jaxtyping", "numpy",
        "openai", "openai-agents", "pydantic", "jax-cuda12-plugin", "jax-cuda12-pjrt",
        "nvidia-cudnn-cu12", "nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


@dataclass(frozen=True)
class Run:
    id: str
    folder: Path
    family_name: str

    @property
    def family(self) -> config.FamilyConfig:
        return config.FAMILY_BY_NAME[self.family_name]

    @property
    def state_file(self) -> Path:
        return self.folder / "state.json"

    @property
    def events_file(self) -> Path:
        return self.folder / "events.jsonl"


def _run_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nonce = hashlib.sha256(os.urandom(32)).hexdigest()[:8]
    return f"{stamp}_{nonce}"


def run_root(family: config.FamilyConfig, experiment: str | None = None) -> Path:
    root = config.ROOT / "results" / (experiment or config.EXPERIMENT)
    return root / family.cli_name / f"replicate_{config.SEARCH_REPLICATE}"


def comparison_signature(family: config.FamilyConfig) -> dict[str, Any]:
    """Settings that must match before sharing baseline/reference evidence.

    The agent interface is deliberately the treatment, so experiment mode
    and instruction contents selected by that mode are excluded.
    All instruction *files* are nevertheless covered by the package hash.
    """
    protocol = config.protocol(family)
    return {
        "family": family.private_dict(),
        "train_seeds": list(config.TRAIN_SEEDS),
        "held_out_seeds": list(config.HELD_OUT_SEEDS),
        "candidate_schedule": [list(pair) for pair in config.GENERATIONS],
        "model": protocol["model"],
        "reward_modes": protocol["reward_modes"],
        "evaluation": protocol["evaluation"],
        "training_evidence": protocol["training_evidence"],
        "learning_probe_budget": {key: value for key, value in protocol.get("learning_probes", {}).items() if key != "feedback"},
        "reward_validation": protocol["reward_validation"],
        "execution": protocol["execution"],
        "dependencies": dependency_versions(),
        "scientific_source_hash": manifest_hash(scientific_source_manifest()),
    }


def _registered_manifest(family: config.FamilyConfig) -> dict[str, Any]:
    sources = source_manifest()
    scientific_sources = scientific_source_manifest()
    protocol = config.protocol(family)
    return {
        "created_at": now(),
        "family": family.name,
        "experiment": config.EXPERIMENT,
        "search_replicate": config.SEARCH_REPLICATE,
        "comparison_signature": comparison_signature(family),
        "protocol": protocol,
        "protocol_hash": hashlib.sha256(
            json.dumps(protocol, sort_keys=True).encode()
        ).hexdigest(),
        "source_manifest": sources,
        "source_hash": manifest_hash(sources),
        "scientific_source_manifest": scientific_sources,
        "scientific_source_hash": manifest_hash(scientific_sources),
        "dependencies": dependency_versions(),
        "accelerator": probe_accelerator(),
        "python": sys.version,
        "platform": platform.platform(),
    }


def create_run(family: config.FamilyConfig, *, identifier: str | None = None, suite: dict | None = None) -> Run:
    root = run_root(family)
    root.mkdir(parents=True, exist_ok=True)
    identifier = identifier or _run_id()
    run = Run(identifier, root / identifier, family.name)
    run.folder.mkdir(parents=True)
    manifest = _registered_manifest(family)
    if suite:
        manifest["suite"] = suite
    write_json(run.folder / "manifest.json", manifest)
    # Hashes alone cannot reconstruct a later-edited package. Archive the exact
    # allowlisted source, including prompts, before any model or learner runs.
    for name, digest in manifest["source_manifest"].items():
        target = run.folder / "registered" / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config.ROOT / name, target)
        if file_hash(target) != digest:
            raise RuntimeError(f"Source changed during registration: {name}")
    write_json(run.folder / "registered" / "protocol.json", config.protocol(family))
    write_json(
        run.state_file,
        {
            "run_id": identifier,
            "family": family.name,
            "family_cli": family.cli_name,
            "experiment": config.EXPERIMENT,
            "search_replicate": config.SEARCH_REPLICATE,
            "planned_jobs": planned_jobs(family),
            "planned_assessment_jobs": planned_jobs(family),
            "maximum_probe_jobs": len(config.CANDIDATES) * config.LEARNING_PROBE_JOBS,
            "status": "running",
            "phase": "created",
            "activity": f"Registered the {family.name} experiment",
            "created_at": now(),
            "updated_at": now(),
            "candidate": None,
            "held_out_revealed": False,
            "reward_locked": False,
        },
    )
    (root / "latest.txt").write_text(identifier + "\n", encoding="utf-8")
    event(run, "run_created", f"Created a new {family.name} study", run_id=identifier)
    return run


def compatible(run: Run) -> bool:
    saved = read_json(run.folder / "manifest.json", {})
    current = _registered_manifest(run.family)
    if saved.get("family") != run.family_name:
        return False
    if saved.get("protocol_hash") != current.get("protocol_hash"):
        return False
    if saved.get("dependencies") != current.get("dependencies"):
        return False
    saved_accelerator = saved.get("accelerator", {})
    current_accelerator = current.get("accelerator", {})
    for name in ("backend", "devices", "process_count"):
        if saved_accelerator.get(name) != current_accelerator.get(name):
            return False
    if saved.get("scientific_source_hash") is None:
        return saved.get("source_hash") == current.get("source_hash")
    return saved.get("scientific_source_hash") == current.get("scientific_source_hash")


def planned_jobs(family: config.FamilyConfig) -> int:
    """Required assessments, excluding optional probes and copied references."""
    development = len(family.development) * len(config.TRAIN_SEEDS)
    held_out = len(family.held_out) * len(config.HELD_OUT_SEEDS)
    return development * (len(config.CANDIDATES) + (config.EXPERIMENT == "exp1")) + held_out * (2 if config.EXPERIMENT == "exp1" else 1)


def suite_held_out_barrier(run: Run) -> None:
    """A suite-bound SMAX run cannot bypass the three-winner freeze barrier.

    This check is also called by standalone evaluation, not only the queue.
    Selection records and frozen programs must still match the queue's hashes.
    """
    manifest = read_json(run.folder / "manifest.json", {})
    binding = manifest.get("suite")
    if not binding or run.family_name != "SMAX":
        return
    suite = read_json(Path(binding["folder"]) / "suite.json", {})
    registration = suite.get("registration", {})
    if suite.get("suite_id") != binding["suite_id"] or manifest_hash(registration) != suite.get("registration_hash"):
        raise RuntimeError("Suite registration is missing or changed; held-out evaluation is forbidden.")
    members = registration.get("members", [])
    member = next((item for item in members if item["key"] == binding["member"]), None)
    if not member or Path(member["run_folder"]).resolve() != run.folder.resolve():
        raise RuntimeError("Run is not a registered member of this suite.")
    smax = [item for item in members if item["family"] == "SMAX"]
    if {item["experiment"] for item in smax} != {"exp1", "exp2", "exp3"} or len(smax) != 3:
        raise RuntimeError("Suite must register all three SMAX conditions before held-out evaluation.")
    for item in smax:
        folder = Path(item["run_folder"])
        accepted = suite.get("frozen_rewards", {}).get(item["key"], {})
        selection_path = folder / "family" / "selection.json"
        frozen = folder / "family" / "frozen_reward.py"
        selection = read_json(selection_path, {})
        if (
            file_hash(folder / "manifest.json") != item["manifest_hash"]
            or not accepted
            or not selection_path.is_file()
            or not frozen.is_file()
            or file_hash(selection_path) != accepted.get("selection_hash")
            or file_hash(frozen) != accepted.get("reward_hash")
            or selection.get("selected_reward_hash") != accepted.get("reward_hash")
        ):
            raise RuntimeError("All three SMAX winners must be selected, frozen and accepted before any held-out evaluation.")


def latest_run(family: config.FamilyConfig, experiment: str | None = None) -> Run | None:
    """Choose by recorded creation time, never fall back to an older success."""
    root = run_root(family, experiment)
    rows = []
    if root.is_dir():
        for path in root.glob("*/manifest.json"):
            manifest = read_json(path, {})
            if manifest.get("family") == family.name:
                rows.append((manifest.get("created_at", ""), path.parent.name, path.parent))
    if not rows:
        return None
    _, identifier, folder = max(rows)
    return Run(identifier, folder, family.name)


def update(run: Run, **changes: Any) -> dict[str, Any]:
    state = read_json(run.state_file, {})
    state.update(changes)
    state["updated_at"] = now()
    write_json(run.state_file, state)
    return state


def event(run: Run, kind: str, message: str, **details: Any) -> None:
    row = {"time": now(), "kind": kind, "message": message, **details}
    run.folder.mkdir(parents=True, exist_ok=True)
    with run.events_file.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            + "\n"
        )
        handle.flush()
