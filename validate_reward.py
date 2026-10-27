"""Validate a generated reward on real development transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import bootstrap  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np

import config
from common.families import adapter_for
from common.reward import candidate_components, component_values, initial_batched_memory, load_reward, normalized_program_hash


def _tree_signature(value: Any):
    return jax.tree_util.tree_structure(value), [
        (tuple(np.asarray(leaf).shape), str(np.asarray(leaf).dtype))
        for leaf in jax.tree_util.tree_leaves(value)
    ]


def validate(path: Path, family_name: str, steps: int = 8):
    family = config.FAMILY_BY_NAME[family_name]
    adapter = adapter_for(family_name)
    reward = load_reward(path, family_name)
    scenario_results = []
    for scenario_index, scenario in enumerate(family.development):
        env = adapter.make_environment(scenario)
        key = jax.random.PRNGKey(41_000 + scenario_index)
        key, reset_key = jax.random.split(key)
        _, state = env.reset(reset_key)
        memory = reward.initial_memory(env.num_agents)
        initial_batched_memory(reward, 1, env.num_agents)  # Enforce the memory size/type contract too.
        initial_signature = _tree_signature(memory)
        component_fn = jax.jit(
            lambda old, actions, new, task, current_memory: candidate_components(
                reward, family.reward_weight, adapter.reward_state(old), actions,
                adapter.reward_state(new), task, current_memory,
            )
        )
        maxima = []
        term_fn = jax.jit(lambda old, actions, new, task, current: component_values(
            reward, adapter.reward_state(old), actions, adapter.reward_state(new), task, current))
        component_names = None
        for _ in range(steps):
            key, action_key, step_key = jax.random.split(key, 3)
            available = adapter.available_actions(state, env)
            actions = jax.random.categorical(action_key, jnp.where(available, 0.0, -1e10)).astype(jnp.int32)
            action_dict = {agent: actions[index] for index, agent in enumerate(env.agents)}
            _, next_state, environment_rewards, dones, info = env.step_env(
                step_key, state, action_dict
            )
            task = adapter.task_reward(
                state, next_state, environment_rewards, info, env.agents
            )
            components, next_memory = component_fn(state, actions, next_state, task, memory)
            values = np.asarray(components.full)
            if values.shape != (env.num_agents,) or not np.isfinite(values).all():
                raise ValueError(f"Non-finite or invalid reward on {scenario}")
            if _tree_signature(next_memory) != initial_signature:
                raise ValueError(f"Reward memory shape or dtype changed during {scenario}")
            if any(not np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree_util.tree_leaves(next_memory)):
                raise ValueError(f"Non-finite reward memory on {scenario}")
            terms = term_fn(state, actions, next_state, task, memory)
            names = sorted(terms)
            if component_names is not None and names != component_names:
                raise ValueError("Component names must be stable across transitions")
            component_names = names
            if terms:
                term_values = np.stack([np.asarray(terms[name]) for name in names])
                if not np.isfinite(term_values).all():
                    raise ValueError("Reward components must be finite")
                if not np.allclose(term_values.sum(axis=0), np.asarray(components.raw_llm), atol=1e-5, rtol=1e-4):
                    raise ValueError("Reward components must sum to raw reward; diagnostics cannot misrepresent payoff")
            maxima.append(float(np.max(np.abs(np.asarray(components.llm)))))
            state, memory = next_state, next_memory
            if bool(np.asarray(dones["__all__"])):
                break
        scenario_results.append({
            "scenario": scenario,
            "num_agents": env.num_agents,
            "steps": len(maxima),
            "maximum_absolute_weighted_llm_reward": max(maxima, default=0.0),
            "component_names": component_names,
        })
    return {
        "valid": True,
        "family": family_name,
        "reward_hash": reward.digest,
        "normalized_program_hash": normalized_program_hash(reward.source),
        "has_reward_components": reward.reward_components is not None,
        "scenarios": scenario_results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reward", type=Path)
    parser.add_argument("--family", required=True, choices=tuple(config.FAMILY_BY_NAME))
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(validate(args.reward.resolve(), args.family, args.steps), indent=2))


if __name__ == "__main__":
    main()
