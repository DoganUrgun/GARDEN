"""Read-only, provenance-checked sharing of Experiment 1 reference evidence.

Only native development artifacts enter a comparison search. Reference held-out
results are copied for reporting after its selected reward has been frozen.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import config
from common.records import Run, comparison_signature, event, file_hash, latest_run, read_json, write_json


def reference_run(family, run: Run | None = None, *, require_held_out: bool = False) -> Run:
    registered = read_json(run.folder / "reference.json", {}) if run else {}
    if registered:
        folder = Path(registered["folder"])
        reference = Run(folder.name, folder, family.name)
    elif run and read_json(run.folder / "manifest.json", {}).get("suite"):
        binding = read_json(run.folder / "manifest.json")["suite"]
        suite = read_json(Path(binding["folder"]) / "suite.json", {})
        from common.records import manifest_hash
        registration = suite.get("registration", {})
        if suite.get("suite_id") != binding["suite_id"] or manifest_hash(registration) != suite.get("registration_hash"):
            raise RuntimeError("The suite reference registration changed.")
        member = next((item for item in registration["members"] if item["family"] == family.name and item["experiment"] == "exp1"), None)
        if member is None:
            raise RuntimeError("This suite has no matching Experiment 1 reference.")
        folder = Path(member["run_folder"])
        if file_hash(folder / "manifest.json") != member["manifest_hash"]:
            raise RuntimeError("The suite reference manifest changed.")
        reference = Run(member["run_id"], folder, family.name)
    else:
        reference = latest_run(family, experiment="exp1")
    if reference is None:
        raise RuntimeError("No GARDEN/native reference exists for this family/search replicate. Start the complete study with python main.py --new.")
    state = read_json(reference.state_file, {})
    manifest = read_json(reference.folder / "manifest.json", {})
    allowed = {"complete"} if require_held_out else {"search_complete", "complete"}
    if state.get("status") not in allowed:
        raise RuntimeError(f"Latest Experiment 1 reference {reference.id} is not complete. No older run will be substituted.")
    if manifest.get("experiment") != "exp1" or manifest.get("comparison_signature") != comparison_signature(family):
        raise RuntimeError(
            "Experiment 1 reference is incompatible with this package/model/learner/seeds. "
            "Reference and control runs must use the same registered scientific settings."
        )
    if registered and file_hash(reference.folder / "manifest.json") != registered["manifest_hash"]:
        raise RuntimeError("The registered reference manifest changed.")
    selection = read_json(reference.folder / "family" / "selection.json", {})
    frozen = reference.folder / "family" / "frozen_reward.py"
    if not frozen.is_file() or file_hash(frozen) != selection.get("selected_reward_hash"):
        raise RuntimeError("Reference frozen reward is missing or its hash changed.")
    if require_held_out and read_json(reference.folder / "family" / "held_out_table.json", {}).get("status") != "complete":
        raise RuntimeError("Reference run claims completion but held-out results are missing.")
    return reference


def reuse_development_baseline(run: Run, family) -> None:
    """Copy artifacts; never modify a reference or silently train a substitute."""
    reference = reference_run(family, run)
    reference_manifest = read_json(reference.folder / "manifest.json", {})
    target_manifest = read_json(run.folder / "manifest.json", {})
    keys = ("backend", "devices", "process_count")
    if any(reference_manifest.get("accelerator", {}).get(k) != target_manifest.get("accelerator", {}).get(k) for k in keys):
        raise RuntimeError("Reference and comparison accelerator configurations differ.")
    source_root = reference.folder / "family" / "development" / "native"
    target_root = run.folder / "family" / "development" / "native"
    hashes = {}
    for scenario in family.development:
        for seed in config.TRAIN_SEEDS:
            relative = Path(scenario) / f"seed_{seed}"
            source = source_root / relative
            result = source / "result.json"
            if not result.is_file():
                raise RuntimeError(f"Reference native job is missing: {result}")
            hashes[(relative / "result.json").as_posix()] = file_hash(result)
            target = target_root / relative
            if (target / "result.json").is_file():
                if file_hash(target / "result.json") != file_hash(result):
                    raise RuntimeError(f"Copied baseline no longer matches its reference: {target}")
            # Complete any copy interrupted after result.json but before traces
            # or checkpoints. Sources remain read-only and hashes stay auditable.
            shutil.copytree(source, target, dirs_exist_ok=True)
    provenance = {
        "run_id": reference.id,
        "folder": str(reference.folder),
        "manifest_hash": file_hash(reference.folder / "manifest.json"),
        "native_development_result_hashes": hashes,
        "held_out_available_to_designer": False,
    }
    write_json(run.folder / "reference.json", provenance)
    event(run, "reference_baseline_reused", "Copied compatible native development evidence; reference held-out results remain outside the designer workspace", reference_run=reference.id)


def reference_held_out(run: Run, family) -> dict:
    """Expose comparison numbers only after this condition's design is locked."""
    if not read_json(run.state_file, {}).get("reward_locked"):
        raise RuntimeError("Reference held-out results cannot be exposed before reward freeze.")
    from common.records import suite_held_out_barrier
    suite_held_out_barrier(run)
    reference = reference_run(family, run, require_held_out=True)
    table = read_json(reference.folder / "family" / "held_out_table.json", {})
    if table.get("evaluation_seeds") != list(config.HELD_OUT_SEEDS):
        raise RuntimeError("Reference held-out evaluation seeds differ.")
    write_json(run.folder / "family" / "reference_held_out.json", table)
    return table
