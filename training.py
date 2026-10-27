"""One-process worker for a single immutable scenario/seed training job."""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import config
from common.hardware import configure_training_environment

configure_training_environment()

import bootstrap  # noqa: E402,F401

from common.evidence import record_trajectories
from common.families import adapter_for
from common.hardware import runtime_accelerator
from common.mappo import evaluate_policy, save_checkpoint, train_policy, write_curve, write_evaluation
from common.records import file_hash, now, write_json
from common.reward import load_reward


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--family", required=True, choices=tuple(config.FAMILY_BY_NAME))
    value.add_argument("--scenario", required=True)
    value.add_argument("--seed", required=True, type=int)
    value.add_argument("--mode", required=True, choices=("native", "candidate"))
    value.add_argument("--reward")
    value.add_argument("--output", required=True, type=Path)
    value.add_argument("--timesteps", type=int)
    value.add_argument("--eval-episodes", type=int)
    value.add_argument("--trace-episodes", type=int)
    value.add_argument("--snapshot-episodes", type=int, default=config.SNAPSHOT_EPISODES,
                       help="Raw diagnostic episodes at each intermediate checkpoint; 0 disables snapshots")
    return value


def run(args):
    family = config.FAMILY_BY_NAME[args.family]
    adapter = adapter_for(args.family)
    if args.scenario not in set(family.development) | set(family.held_out):
        raise ValueError(f"Scenario is not registered for {args.family}: {args.scenario}")
    if (args.mode == "candidate") != bool(args.reward):
        raise ValueError("Candidate mode requires --reward and native mode forbids it")
    reward_path = Path(args.reward).resolve() if args.reward else None
    reward = load_reward(reward_path, args.family) if reward_path else None
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    timesteps = args.timesteps or family.timesteps
    eval_episodes = args.eval_episodes or family.evaluation_episodes
    trace_episodes = args.trace_episodes or family.trace_episodes
    snapshot_episodes = getattr(args, "snapshot_episodes", config.SNAPSHOT_EPISODES)
    if snapshot_episodes < 0:
        raise ValueError("--snapshot-episodes must be nonnegative")
    snapshot_fractions = config.SNAPSHOT_FRACTIONS if snapshot_episodes else ()
    snapshot_seed = args.seed + 300_000
    environment = adapter.make_environment(args.scenario)
    spec = {
        "family": args.family,
        "scenario": args.scenario,
        "seed": args.seed,
        "mode": args.mode,
        "reward_source": str(reward_path) if reward_path else None,
        "reward_hash": reward.digest if reward else None,
        "reward_weight": family.reward_weight if reward else None,
        "requested_timesteps": timesteps,
        "evaluation_episodes": eval_episodes,
        "trace_episodes": trace_episodes,
        "snapshot_fractions": list(snapshot_fractions),
        "snapshot_episodes": snapshot_episodes,
        "snapshot_seed": snapshot_seed,
        "evaluation_greedy": config.EVALUATION_GREEDY,
        "learner": family.learner.to_dict(),
        "network": adapter.network_config(environment),
        "accelerator": runtime_accelerator(),
        "started_at": now(),
    }
    write_json(output / "job.json", spec)

    latest_progress = {}

    def progress(row, update_index, updates):
        latest_progress.update({
            "status": "training", "update": update_index, "updates": updates,
            "timesteps": int(row["timesteps"]), "latest": row, "updated_at": now(),
        })
        write_json(output / "progress.json", latest_progress)

    snapshots = []
    pending_fractions = list(snapshot_fractions)

    def snapshot(parameters, update_index, updates, completed_steps):
        # Observe the first EXISTING chunk boundary reaching each target. Never
        # split a PPO chunk to obtain an exact percentage: that could alter JAX
        # compilation and numerical behavior. Report the actual saved position.
        reached = [fraction for fraction in pending_fractions if update_index >= fraction * updates]
        if not reached:
            return
        folder = output / "training_snapshots" / f"update_{update_index:04d}"
        snapshot_metadata = {
            **spec, "update": update_index, "updates": updates,
            "requested_fractions": reached, "actual_fraction": update_index / updates,
            "actual_timesteps": completed_steps, "trace_episodes": snapshot_episodes,
            "trace_seed": snapshot_seed,
        }
        write_json(output / "progress.json", {
            **latest_progress, "status": "snapshots", "updated_at": now(),
            "activity": f"Recording {snapshot_episodes} raw episodes at update {update_index}/{updates}",
        })
        save_checkpoint(folder / "checkpoint", parameters, snapshot_metadata)
        # The same independent seed at both stages gives matched diagnostic
        # resets. It never consumes or changes the learner's random stream.
        record_trajectories(folder, environment, adapter, parameters, args.mode, reward,
                            snapshot_seed, snapshot_episodes, greedy=config.EVALUATION_GREEDY)
        write_json(folder / "metadata.json", snapshot_metadata)
        snapshots.append({
            "folder": str(folder.relative_to(output)).replace("\\", "/"),
            "update": update_index, "updates": updates,
            "requested_fractions": reached, "actual_fraction": update_index / updates,
            "actual_timesteps": completed_steps, "trace_episodes": snapshot_episodes,
            "trace_seed": snapshot_seed,
        })
        write_json(output / "training_snapshots" / "index.json", {
            "description": "Independent raw policy rollouts and checkpoints; not training samples or selection evaluations.",
            "snapshots": snapshots,
        })
        pending_fractions[:] = [fraction for fraction in pending_fractions if fraction not in reached]
        latest_progress["updated_at"] = now()
        write_json(output / "progress.json", latest_progress)

    trained = train_policy(environment, adapter, args.mode, reward, timesteps, args.seed,
                           progress_callback=progress, snapshot_callback=snapshot if snapshot_episodes else None)
    write_curve(output / "training_curve.csv", trained.curve)
    metadata = {**spec, "actual_timesteps": trained.actual_timesteps}
    save_checkpoint(output / "checkpoint", trained.parameters, metadata)
    write_json(output / "progress.json", {
        **latest_progress, "status": "evaluation", "updated_at": now(),
        "activity": f"Evaluating the final policy on {eval_episodes} episodes",
    })
    evaluation = evaluate_policy(environment, adapter, trained.parameters, args.mode, reward, eval_episodes, args.seed + 100_000, greedy=config.EVALUATION_GREEDY)
    write_evaluation(output / "evaluation_episodes.csv", evaluation)
    write_json(output / "progress.json", {
        **latest_progress, "status": "traces", "updated_at": now(),
        "activity": f"Recording {trace_episodes} final-policy raw episodes",
    })
    record_trajectories(output, environment, adapter, trained.parameters, args.mode, reward, args.seed + 200_000, trace_episodes, greedy=config.EVALUATION_GREEDY)
    result = {
        **spec,
        "actual_timesteps": trained.actual_timesteps,
        "training_snapshots": snapshots,
        "evaluation": evaluation.summary(),
        "checkpoint_hashes": {
            "actor": file_hash(output / "checkpoint" / "actor.msgpack"),
            "critic": file_hash(output / "checkpoint" / "critic.msgpack"),
        },
        "completed_at": now(),
    }
    write_json(output / "result.json", result)
    write_json(output / "progress.json", {"status": "complete", "updated_at": now(), "result": result["evaluation"]})


def main():
    args = parser().parse_args()
    try:
        run(args)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "progress.json", {
            "status": "failed", "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(), "updated_at": now(),
        })
        raise


if __name__ == "__main__":
    main()
