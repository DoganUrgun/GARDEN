"""Build auditable development-only workspaces for the reward engineer."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
from pathlib import Path

import config
from common.records import Run, file_hash, read_json, write_json


def _filter_mapping(path: Path, assignment_name: str, allowed: set[str]) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    scenario_variables = set()
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == assignment_name
            and isinstance(node.value, ast.Dict)
        ):
            for value in node.value.values:
                if isinstance(value, ast.Call) and value.args and isinstance(value.args[0], ast.Name):
                    scenario_variables.add(value.args[0].id)
            pairs = [
                (key, value)
                for key, value in zip(node.value.keys, node.value.values)
                if isinstance(key, ast.Constant) and str(key.value) in allowed
            ]
            node.value.keys = [key for key, _ in pairs]
            node.value.values = [value for _, value in pairs]
    if scenario_variables:
        tree.body = [
            node for node in tree.body
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in scenario_variables
                and node.targets[0].id not in allowed
            )
        ]
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree) + "\n"
    compile(source, str(path), "exec")
    path.write_text(source, encoding="utf-8")


def _sanitize_runtime(reference: Path, family: config.FamilyConfig) -> None:
    if family.name == "Overcooked":
        _filter_mapping(
            reference / "runtime/jaxmarl/environments/overcooked/layouts.py",
            "overcooked_layouts",
            set(family.development),
        )
    else:
        _filter_mapping(reference / "runtime/jaxmarl/environments/smax/smax_env.py", "MAP_NAME_TO_SCENARIO", set(family.development))


def _copy_reference(reference: Path, family: config.FamilyConfig) -> None:
    reference.mkdir(parents=True, exist_ok=True)
    shutil.copytree(config.RUNTIME, reference / "runtime", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.md"))
    environments = reference / "runtime/jaxmarl/environments"
    keep = {
        "Overcooked": "overcooked",
        "SMAX": "smax",
    }[family.name]
    for environment in ("overcooked", "smax"):
        if environment != keep and (environments / environment).exists():
            shutil.rmtree(environments / environment)
    if family.name == "Overcooked":
        exports = 'from .environments.overcooked import Overcooked, overcooked_layouts\n\n__all__ = ["Overcooked", "overcooked_layouts"]\n'
        environment_exports = 'from .multi_agent_env import MultiAgentEnv, State\nfrom .overcooked import Overcooked, overcooked_layouts\n\n__all__ = ["MultiAgentEnv", "State", "Overcooked", "overcooked_layouts"]\n'
    else:
        exports = 'from .environments.smax import HeuristicEnemySMAX, SMAX, map_name_to_scenario\n\n__all__ = ["HeuristicEnemySMAX", "SMAX", "map_name_to_scenario"]\n'
        environment_exports = 'from .multi_agent_env import MultiAgentEnv, State\nfrom .smax import HeuristicEnemySMAX, SMAX, map_name_to_scenario\n\n__all__ = ["MultiAgentEnv", "State", "HeuristicEnemySMAX", "SMAX", "map_name_to_scenario"]\n'
    (reference / "runtime/jaxmarl/__init__.py").write_text(exports, encoding="utf-8")
    (environments / "__init__.py").write_text(environment_exports, encoding="utf-8")
    _sanitize_runtime(reference, family)
    for name in ("bootstrap.py", "validate_reward.py", "PROTOCOL.md"):
        shutil.copy2(config.ROOT / name, reference / name)
    # Give the engineer the complete learning/reward machinery, not the private
    # search controller or cross-experiment comparison loader.
    (reference / "common").mkdir()
    for name in ("__init__.py", "mappo.py", "networks.py", "reward.py", "families.py", "records.py", "hardware.py", "evidence.py"):
        source = config.ROOT / "common" / name
        if source.exists():
            shutil.copy2(source, reference / "common" / name)
    shutil.copytree(config.ROOT / family.name, reference / family.name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    redacted = (config.ROOT / "config.py").read_text(encoding="utf-8")
    for index, item in enumerate(family.held_out, start=1):
        redacted = redacted.replace(item, f"concealed_scenario_{index}")
    for other in config.FAMILIES:
        if other.name != family.name:
            for index, item in enumerate(other.development + other.held_out, start=1):
                redacted = redacted.replace(item, f"other_family_concealed_{index}")
    (reference / "config.py").write_text(redacted, encoding="utf-8")
    write_json(reference / "PUBLIC_FAMILY_CONFIG.json", family.public_dict())
    for path in reference.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".py", ".md", ".json", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            leaked = sorted(item for item in family.held_out if item in text)
            if leaked:
                raise RuntimeError(f"Held-out identifier leaked into designer reference {path}: {leaked}")


def _copy_tree(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)


def _redact_controller_paths(root: Path, run: Run) -> None:
    replacements = {
        str(run.folder): "<registered_run>",
        str(config.ROOT): "<project_root>",
        str(run.folder).replace("\\", "/"): "<registered_run>",
        str(config.ROOT).replace("\\", "/"): "<project_root>",
    }
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".csv", ".txt", ".md", ".log"}:
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        redacted = source
        for private, public in replacements.items():
            redacted = redacted.replace(private, public)
        if redacted != source:
            path.write_text(redacted, encoding="utf-8")


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): file_hash(path)
        for path in sorted(root.rglob("*")) if path.is_file()
    } if root.exists() else {}


def protected_manifest(workspace: Path) -> dict[str, str]:
    values = {}
    for folder_name in ("reference", "evidence"):
        for name, digest in _hash_tree(workspace / folder_name).items():
            values[f"{folder_name}/{name}"] = digest
    return values


def _visibility(family: config.FamilyConfig, candidate: str, generation: int, position: str) -> dict:
    visible_measured = [
        earlier_candidate
        for earlier_generation in range(generation - 1)
        for earlier_candidate in config.GENERATIONS[earlier_generation]
    ]
    return {
        "family": family.name,
        "candidate": candidate,
        "generation": generation,
        "visible_measured_candidates": visible_measured,
        "visible_same_generation_without_results": [],
        "same_generation_private": True,
        "held_out_visible": False,
    }


def _workspace_has_progress(workspace: Path) -> bool:
    return any(
        path.is_file()
        for folder in (workspace / "scratch", workspace / "output")
        if folder.is_dir()
        for path in folder.rglob("*")
    )


def _verify_resumable_workspace(workspace: Path, expected_visibility: dict) -> None:
    """Accept unfinished work only when its protected evidence is unchanged."""
    registered = read_json(workspace / "workspace_manifest.json", {})
    expected_files = registered.get("protected_files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise RuntimeError(f"Cannot resume {workspace.name}: workspace manifest is missing")
    expected_hash = hashlib.sha256(
        json.dumps(expected_files, sort_keys=True).encode()
    ).hexdigest()
    if registered.get("manifest_hash") != expected_hash:
        raise RuntimeError(f"Cannot resume {workspace.name}: workspace manifest is invalid")
    if protected_manifest(workspace) != expected_files:
        raise RuntimeError(
            f"Cannot resume {workspace.name}: protected reference or evidence changed"
        )
    if read_json(workspace / "evidence" / "VISIBILITY.json", {}) != expected_visibility:
        raise RuntimeError(f"Cannot resume {workspace.name}: evidence visibility changed")


def prepare_workspace(run: Run, family: config.FamilyConfig, candidate: str, generation: int, position: str) -> Path:
    workspace = run.folder / "workspaces" / candidate
    visibility = _visibility(family, candidate, generation, position)
    if workspace.exists():
        if _workspace_has_progress(workspace):
            _verify_resumable_workspace(workspace, visibility)
            return workspace
        shutil.rmtree(workspace)
    reference, evidence = workspace / "reference", workspace / "evidence"
    for folder in (evidence, workspace / "scratch", workspace / "output"):
        folder.mkdir(parents=True, exist_ok=True)
    _copy_reference(reference, family)
    family_root = run.folder / "family"
    _copy_tree(family_root / "development" / "native", evidence / "native_baseline")
    for earlier_generation in range(generation - 1):
        for earlier_candidate in config.GENERATIONS[earlier_generation]:
            destination = evidence / "candidates" / earlier_candidate
            _copy_tree(family_root / "candidates" / earlier_candidate, destination / "design")
            _copy_tree(family_root / "development" / earlier_candidate, destination / "training")
            _copy_tree(run.folder / "workspaces" / earlier_candidate / "scratch", destination / "autonomous_analysis")
            _copy_tree(run.folder / "llm_tool_outputs" / earlier_candidate, destination / "tool_outputs")
            _copy_tree(run.folder / "learning_probes" / earlier_candidate, destination / "learning_probes")
            # Candidate-local sessions prevent sibling leaks. Make the earlier
            # engineer's recorded reasoning summaries explicitly available here.
            if run.events_file.is_file():
                investigation = [row for line in run.events_file.read_text(encoding="utf-8").splitlines()
                                 if (row := json.loads(line)).get("candidate") == earlier_candidate
                                 and row.get("kind", "").startswith("llm_")]
                write_json(destination / "investigation_log.json", investigation)
    write_json(evidence / "VISIBILITY.json", visibility)
    _redact_controller_paths(evidence, run)
    shell_name = "non-interactive PowerShell" if os.name == "nt" else "a non-interactive POSIX shell"
    (workspace / "WORKSPACE.md").write_text(f"""# Workspace for {candidate}

Environment family: `{family.name}`
Generation: `{generation}`, position: `{position.upper()}`

- `reference/` contains immutable permitted source and the fixed public protocol.
- `evidence/` contains the native development baseline and only results permitted by the generation boundary.
- `scratch/` is writable for autonomous analysis scripts and derived evidence.
- `output/` must contain the final `reward.py` and `report.md`.

Validate with:

```text
python reference/validate_reward.py output/reward.py --family {family.name}
```

Commands run with {shell_name}. Parent paths, absolute paths, credentials,
network services, and held-out definitions are outside the permitted interface.
Do not modify immutable evidence.
""", encoding="utf-8")
    manifest = protected_manifest(workspace)
    write_json(workspace / "workspace_manifest.json", {
        "protected_files": manifest,
        "manifest_hash": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
    })
    return workspace
