"""Execute one registered study stage in an isolated Python process.

This is an internal worker, launched by common/suite.py. Use main.py to start
or resume the study; workers never choose runs or register experiments.
"""

from __future__ import annotations

import argparse
import os
import traceback
from pathlib import Path


def run_stage(family, phase, run_folder):
    """Run only the exact family, condition and folder chosen by the queue."""
    import config
    from common.hardware import require_registered_accelerator
    from common.records import Run, compatible, event, read_json, update

    folder = run_folder.resolve()
    manifest = read_json(folder / "manifest.json", {})
    if manifest.get("family") != family.name or manifest.get("experiment") != config.EXPERIMENT:
        raise RuntimeError("The registered run does not match this condition and family.")
    binding = manifest.get("suite", {})
    if not all(binding.get(key) for key in ("suite_id", "folder", "member")):
        raise RuntimeError("A worker requires a registered suite member; use python main.py --new.")
    run = Run(folder.name, folder, family.name)
    if not compatible(run):
        raise RuntimeError("Cannot resume: registered protocol, source, dependencies or accelerator changed.")

    # Controls reuse the native reference bound to this suite, never another run.
    if config.EXPERIMENT != "exp1":
        from common.comparison import reference_run
        reference_run(family, run)
    state = read_json(run.state_file, {})
    if state.get("status") == "complete":
        print(f"Already complete: {run.folder}")
        return
    if phase == "search" and state.get("status") == "search_complete":
        print(f"Search already complete and frozen: {run.folder}")
        return
    needs_design = phase != "evaluate" and not (run.folder / "family" / "selection.json").is_file()
    if needs_design and not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in this package's private .env file.")
    require_registered_accelerator()
    # The controller validates rewards on CPU; learner workers request CUDA.
    os.environ["JAX_PLATFORMS"] = "cpu"
    from common.study import run_study

    update(run, status="running", error=None, traceback=None)
    print(f"Running {config.EXPERIMENT}/{family.name}, {phase}: {run.id}", flush=True)
    try:
        run_study(run, family, phase)
    except KeyboardInterrupt:
        message = "Interrupted safely; completed jobs are preserved. Resume with python main.py --resume latest."
        update(run, status="interrupted", phase="interrupted", activity=message, error=None)
        event(run, "study_interrupted", message, family=family.name)
        print(message)
        raise SystemExit(130)
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        update(run, status="failed", phase="error", activity=message, error=message, traceback=traceback.format_exc())
        event(run, "study_failed", message, family=family.name)
        raise
    print(f"{phase.capitalize()} complete: {run.folder}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=("exp1", "exp2", "exp3"))
    parser.add_argument("--family", required=True, choices=("Overcooked", "SMAX"))
    parser.add_argument("--phase", required=True, choices=("search", "evaluate", "study"))
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    # Set the condition before configuration or any study module is imported.
    os.environ["REWARD_DESIGN_EXPERIMENT"] = args.experiment
    import bootstrap  # noqa: F401
    import config
    run_stage(config.FAMILY_BY_NAME[args.family], args.phase, args.run)


if __name__ == "__main__":
    main()
