"""Raw policy trajectories for autonomous inspection."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from common.mappo import PolicyParameters, actor_step
from common.reward import LoadedReward, candidate_components, initial_batched_memory, native_payoff


def _unbatch_memory(memory: Any) -> Any:
    return jax.tree_util.tree_map(lambda value: value[0], memory)


def record_trajectories(folder: Path, env, adapter, parameters: PolicyParameters, mode: str, reward: LoadedReward | None, seed: int, episodes: int, *, greedy: bool) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    learner = adapter.config.learner
    key = jax.random.PRNGKey(seed)
    episode_rows = []
    raw_folder = folder / "trace_raw"
    raw_folder.mkdir(parents=True, exist_ok=True)
    raw_index = {
        "format": "numeric-npz-v1",
        "family": adapter.family,
        "seed": seed,
        "description": "Complete reward-interface transitions plus policy observations and legal actions. No pickle required.",
        "episodes": [],
    }

    def transition(observations, state, memory, hidden, current_key):
        current_key, action_key, step_key = jax.random.split(current_key, 3)
        hidden, actions, logits = actor_step(
            env,
            adapter,
            parameters.actor,
            observations,
            state,
            hidden,
            action_key,
            greedy=greedy,
        )
        action_dict = {
            agent: actions[index] for index, agent in enumerate(env.agents)
        }
        next_observations, next_state, environment_rewards, dones, info = (
            env.step_env(step_key, state, action_dict)
        )
        task = adapter.task_reward(
            state, next_state, environment_rewards, info, env.agents
        )
        if mode == "native":
            native = adapter.native_shaping(
                environment_rewards, info, task, env.agents
            )
            full = native_payoff(task, native)
            llm = raw_llm = jnp.zeros_like(full)
            next_memory = memory
        else:
            components, next_memory = candidate_components(
                reward,
                adapter.config.reward_weight,
                adapter.reward_state(state),
                actions,
                adapter.reward_state(next_state),
                task,
                memory,
            )
            full = components.full
            native = jnp.zeros_like(full)
            llm, raw_llm = components.llm, components.raw_llm
        return (
            current_key,
            next_observations,
            next_state,
            next_memory,
            hidden,
            actions,
            logits,
            task,
            full,
            native,
            llm,
            raw_llm,
            dones,
        )

    # A Python loop is retained so each decoded state can be written as a
    # readable artifact. The expensive network/environment/reward transition
    # is one compiled device call instead of hundreds of eager JAX operations.
    compiled_transition = jax.jit(transition)
    with (folder / "trajectories.jsonl").open("w", encoding="utf-8") as trajectory_file:
        for episode in range(episodes):
            key, reset_key = jax.random.split(key)
            observations, state = env.reset(reset_key)
            hidden = jnp.zeros((env.num_agents, learner.actor_hidden), dtype=jnp.float32)
            memory = () if reward is None else _unbatch_memory(initial_batched_memory(reward, 1, env.num_agents))
            totals = {"task": 0.0, "training": 0.0, "native": 0.0, "llm": 0.0}
            raw_arrays = {}
            raw_schema = {}
            steps = 0
            for step_index in range(learner.max_steps):
                (
                    key,
                    next_observations,
                    next_state,
                    next_memory,
                    hidden,
                    actions,
                    logits,
                    task,
                    full,
                    native,
                    llm,
                    raw_llm,
                    dones,
                ) = compiled_transition(observations, state, memory, hidden, key)
                # Keep complete legitimate state leaves, not only our human
                # rendering. For example, SMAX cooldowns are absent from the
                # compact JSON view but available here for autonomous analysis.
                transition_trees = {
                    "state": adapter.reward_state(state),
                    "next_state": adapter.reward_state(next_state),
                    "actions": actions,
                    "task_reward": task,
                    "memory": memory,
                    "next_memory": next_memory,
                    "observations": observations,
                    "available_actions": adapter.available_actions(state, env),
                }
                for field, tree in transition_trees.items():
                    leaves, tree_definition = jax.tree_util.tree_flatten_with_path(tree)
                    fields = []
                    for index, (path, leaf) in enumerate(leaves):
                        value = np.asarray(leaf)
                        if value.dtype.hasobject:
                            raise TypeError(f"Raw trace field {field} contains a non-numeric object")
                        name = f"{field}_{index:03d}"
                        raw_arrays.setdefault(name, []).append(value)
                        fields.append({
                            "array": name, "path": jax.tree_util.keystr(path),
                            "shape_per_transition": list(value.shape), "dtype": str(value.dtype),
                        })
                    schema = {"tree_definition": str(tree_definition), "leaves": fields}
                    if field in raw_schema and raw_schema[field] != schema:
                        raise ValueError(f"Raw trace structure changed within an episode: {field}")
                    raw_schema[field] = schema
                task_value = float(np.asarray(task))
                full_values, native_values = np.asarray(full, dtype=np.float64), np.asarray(native, dtype=np.float64)
                llm_values, raw_values = np.asarray(llm, dtype=np.float64), np.asarray(raw_llm, dtype=np.float64)
                row = {
                    "episode": episode,
                    "step": step_index,
                    "state": adapter.decode_state(state),
                    "actions": [int(value) for value in np.asarray(actions)],
                    "action_probabilities": np.asarray(jax.nn.softmax(logits), dtype=np.float64).tolist(),
                    "next_state": adapter.decode_state(next_state),
                    "task_reward": task_value,
                    "training_reward": full_values.tolist(),
                    "native_shaping": native_values.tolist(),
                    "llm_reward": llm_values.tolist(),
                    "raw_llm_reward": raw_values.tolist(),
                    "done": bool(np.asarray(dones["__all__"])),
                }
                trajectory_file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                totals["task"] += task_value
                totals["training"] += float(full_values.mean())
                totals["native"] += float(native_values.mean())
                totals["llm"] += float(llm_values.mean())
                steps += 1
                observations, state, memory = next_observations, next_state, next_memory
                if bool(np.asarray(dones["__all__"])):
                    break
            if not bool(np.asarray(dones["__all__"])):
                raise RuntimeError("Trace horizon ended before episode termination")
            archive_name = f"episode_{episode:03d}.npz"
            np.savez_compressed(
                raw_folder / archive_name,
                **{name: np.stack(values) for name, values in raw_arrays.items()},
            )
            raw_index["episodes"].append({
                "episode": episode, "steps": steps,
                "file": f"trace_raw/{archive_name}", "fields": raw_schema,
            })
            episode_rows.append({
                "episode": episode,
                "steps": steps,
                "task_return": totals["task"],
                "training_return_mean": totals["training"],
                "native_shaping_return_mean": totals["native"],
                "llm_return_mean": totals["llm"],
            })
    fields = list(episode_rows[0]) if episode_rows else ["episode", "task_return"]
    with (folder / "trace_episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(episode_rows)
    (folder / "trajectory_arrays.json").write_text(
        json.dumps(raw_index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
