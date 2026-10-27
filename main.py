"""GARDEN: one entry point for the complete paper study.

Start here
----------
    python main.py                 Show help; nothing is trained or generated.
    python main.py plan            Show settings, order and training-job counts.
    python main.py doctor          Check whether CUDA-enabled JAX is available.
    python main.py --new           Start a NEW complete study (uses GPU and API).
    python main.py status          Read the latest study's saved progress.
    python main.py --resume latest Continue that study, keeping completed jobs.

What is being compared?
-----------------------
The paper has TWO main comparisons, not one experiment per Python file:
  1. GARDEN versus native shaping, in Overcooked and SMAX.
  2. Autonomous evidence navigation versus two controls, in SMAX only.

Three reward-design conditions are needed to make these comparisons:
  - GARDEN: the designer chooses what evidence to inspect and how to analyse it.
  - Fixed diagnostics: the designer receives a predefined diagnostic report.
  - Eureka-inspired: the designer receives compact reward-reflection feedback.
Native shaping is the environment's reference reward, not a fourth LLM search.

What does --new run?
-------------------
First, all three SMAX conditions search for rewards and freeze their winners.
Only then are those winners evaluated on held-out SMAX scenarios. Overcooked's
GARDEN/native study follows. This ordering keeps held-out feedback out of search.
Each search designs six rewards in three A/B generations. Held-out policies
are trained from scratch; increasing training seeds does not repeat LLM search.

Settings and files
------------------
Copy .env.example to .env and set your key/settings BEFORE starting. Results
are created under results/ automatically. Do not change scientific settings or
source while resuming a study. --new starts over; --resume preserves progress.

This file handles commands and explanations only. common/suite.py controls the
sequence, common/study.py runs each condition, and common/mappo.py trains policies.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def print_plan():
    """Read configuration and count jobs, without calling the model or JAX."""
    import config
    from common import suite

    overview = suite.plan()
    print("GARDEN study plan (read-only; no API calls or training)")
    print("\nPaper comparisons:")
    print("  1. GARDEN versus native shaping: Overcooked and SMAX.")
    print("  2. GARDEN versus fixed diagnostics and Eureka-inspired control: SMAX.")
    print("\nExecution order:")
    for number, (stage, _, _) in enumerate(suite.STAGES, start=1):
        print(f"  {number}. {suite.STAGE_NAMES[stage]}")
    print("  All three SMAX winners are frozen before step 4. No held-out feedback enters search.")

    print("\nSettings:")
    print(f"  Model: {config.MODEL}")
    print(f"  Candidates per search: {len(config.CANDIDATES)} ({len(config.GENERATIONS)} A/B generations)")
    print(f"  Development training seeds: {list(config.TRAIN_SEEDS)}")
    print(f"  Held-out training seeds: {list(config.HELD_OUT_SEEDS)} (same frozen reward)")
    print("  Four reward searches: three SMAX conditions and one Overcooked condition.")
    for family in config.FAMILIES:
        batch = family.learner.num_envs * family.learner.rollout_steps
        actual_steps = (family.timesteps // batch) * batch
        print(f"\n  {family.name}: {actual_steps:,} training steps per job (complete rollout batches)")
        print(f"    Development: {', '.join(family.development)}")
        print(f"    Held-out:    {', '.join(family.held_out)}")

    print("\nTraining budget:")
    print(f"  {'Condition':<44} {'Assessments':>11} {'Optional probes':>16}")
    for member, counts in zip(suite.MEMBERS, overview["members"]):
        _, experiment, family = member
        label = f"{family}: {suite.CONDITION_NAMES[experiment]}"
        print(f"  {label:<44} {counts['assessment_jobs']:>11} {counts['maximum_probe_jobs']:>16}")
    assessments = sum(item["assessment_jobs"] for item in overview["members"])
    probes = sum(item["maximum_probe_jobs"] for item in overview["members"])
    print(f"  {assessments} assessment jobs + up to {probes} optional probes = at most {overview['maximum_training_jobs']} training jobs.")
    print("  An exact matching probe can replace an assessment, reducing actual jobs.")
    print("  These counts exclude evaluation rollouts, trajectory recording and reward validation.")
    print(f"\nOutputs: {config.ROOT / 'results'}")
    print("  suites/ keeps the overall progress and stage logs.")
    print("  exp1/ keeps GARDEN/native; exp2/ fixed diagnostics; exp3/ Eureka-inspired.")
    print("\nTo start intentionally: python main.py --new")


def latest_suite():
    """Find the newest registered study, not merely the latest successful one."""
    import config
    from common.records import read_json

    records = list((config.ROOT / "results" / "suites").glob("*/suite.json"))
    if not records:
        return None
    return max(records, key=lambda path: (read_json(path, {}).get("created_at", ""), path.parent.name)).parent


def print_status():
    """Show saved progress without launching a worker or contacting the API."""
    from common.records import read_json
    from common.suite import STAGE_NAMES

    folder = latest_suite()
    if folder is None:
        print("No registered study exists. Use python main.py plan to see what would run.")
        return
    record = read_json(folder / "suite.json")
    print(f"Study: {folder.name}\nRecorded status: {record['status']}\nRecords: {folder}")
    print("Saved progress only; this command does not verify that a process is still alive.")
    for stage in record["registration"]["stages"]:
        status = record["stages_status"].get(stage["id"], {}).get("status", "pending")
        print(f"  {status:<12} {STAGE_NAMES.get(stage['id'], stage['id'])}")
    if record.get("error"):
        print(f"Last error: {record['error']}")


def doctor():
    """Probe the training accelerator; do not train policies or call the LLM."""
    import config
    from common.hardware import probe_accelerator

    info = probe_accelerator()
    print(json.dumps(info, indent=2))
    if info.get("gpu"):
        print("PASS: CUDA-enabled JAX is available. This checks execution, not learning quality.")
    elif config.REQUIRE_GPU:
        print("NOT READY: use Linux or Ubuntu/WSL2 with requirements-gpu.txt. See docs/INSTALL.md.")
        raise SystemExit(2)
    else:
        print("GPU not detected. REQUIRE_GPU=0 permits CPU execution; full studies may be very slow.")


def main():
    parser = argparse.ArgumentParser(
        description="GARDEN: run both paper comparisons through one entry point.",
        epilog="With no arguments, only this help is shown. See the comments in main.py and docs/EXPERIMENTS.md.",
    )
    parser.add_argument("command", nargs="?", choices=("plan", "status", "doctor"),
                        help="plan: preview study; status: read latest progress; doctor: check CUDA")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--new", action="store_true", help="start the entire study from scratch (GPU training and paid API calls)")
    action.add_argument("--resume", metavar="FOLDER_OR_LATEST", help="continue a registered study; completed jobs are preserved")
    args = parser.parse_args()
    if args.command and (args.new or args.resume):
        parser.error("Choose a read-only command OR --new/--resume, not both.")
    if not args.command and not args.new and not args.resume:
        parser.print_help()
        return

    # Heavy dependencies are imported only when needed. Help does not even
    # require the study dependencies; plan and status never initialize JAX.
    if args.command == "plan":
        print_plan()
        return
    if args.command == "status":
        print_status()
        return
    if args.command == "doctor":
        doctor()
        return

    import bootstrap  # noqa: F401
    import config  # noqa: F401 -- loads this package's private .env settings
    from common.hardware import require_registered_accelerator
    from common.suite import register_suite, run_suite

    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in .env; no study was started.")
    folder = None
    if args.resume:
        folder = latest_suite() if args.resume == "latest" else Path(args.resume).resolve()
        if folder is None or not (folder / "suite.json").is_file():
            raise RuntimeError("No registered study exists at the requested location; nothing was started.")
    require_registered_accelerator()
    if args.new:
        folder = register_suite()
    print(f"Study records: {folder}", flush=True)
    try:
        run_suite(folder, resume=bool(args.resume))
    except KeyboardInterrupt:
        print("Stopped. Completed jobs are preserved. Continue with python main.py --resume latest.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
