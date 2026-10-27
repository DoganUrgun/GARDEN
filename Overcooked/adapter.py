"""Classic Overcooked task/native decomposition and MAPPO feature adapter."""

from __future__ import annotations

import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import config
from Overcooked.networks import ActorCNN, CentralCriticCNN
from jaxmarl.environments.overcooked import Overcooked, overcooked_layouts


class Adapter:
    family = "Overcooked"
    recurrent = False

    def __init__(self, family_config: config.FamilyConfig = config.OVERCOOKED):
        self.config = family_config

    def make_environment(self, scenario: str, *, random_reset: bool = False):
        # This study intentionally uses the classic benchmark for its first,
        # faster family; PROTOCOL.md records the upstream deprecation caveat.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="OvercookedV2 is now released*", category=DeprecationWarning
            )
            return Overcooked(
                layout=overcooked_layouts[scenario],
                random_reset=random_reset,
                max_steps=self.config.learner.max_steps,
            )

    @staticmethod
    def reward_state(state: Any) -> Any:
        return state

    @staticmethod
    def task_reward(previous_state, next_state, environment_rewards, info, agents):
        """Return the shared sparse delivery reward emitted by the environment."""
        del previous_state, next_state, info
        rewards = jnp.stack([environment_rewards[agent] for agent in agents], axis=-1)
        return rewards[..., 0].astype(jnp.float32)

    @staticmethod
    def native_shaping(environment_rewards, info, task_reward, agents) -> jax.Array:
        del environment_rewards, task_reward
        return jnp.stack([info["shaped_reward"][agent] for agent in agents], axis=-1).astype(
            jnp.float32
        )

    @staticmethod
    def actor_inputs(observations, state, env) -> jax.Array:
        del state
        shape = env.observation_space(env.agents[0]).shape
        unbatched = observations[env.agents[0]].ndim == len(shape)
        axis = 0 if unbatched else 1
        values = jnp.stack(
            [observations[agent].astype(jnp.float32) for agent in env.agents], axis=axis
        )
        return values.reshape(values.shape[:2] + (-1,)) if not unbatched else values.reshape((env.num_agents, -1))

    @staticmethod
    def global_inputs(observations, state, env) -> jax.Array:
        del state
        first = observations[env.agents[0]]
        shape = env.observation_space(env.agents[0]).shape
        unbatched = first.ndim == len(shape)
        joint = jnp.concatenate([observations[agent] for agent in env.agents], axis=-1)
        return joint.reshape((-1,)) if unbatched else joint.reshape((joint.shape[0], -1))

    @staticmethod
    def available_actions(state, env) -> jax.Array:
        step = jnp.asarray(state.step)
        shape = (env.num_agents, env.action_space(env.agents[0]).n)
        if step.ndim:
            shape = (step.shape[0],) + shape
        return jnp.ones(shape, dtype=bool)

    @staticmethod
    def done_arrays(dones, env):
        return jnp.stack([dones[agent] for agent in env.agents], axis=-1), jnp.asarray(
            dones["__all__"]
        )

    def build_networks(self, env):
        # Upstream advertises W,H,C, but get_obs actually returns H,W,C.
        # Use the real grid dimensions; swapping a flattened array's dimensions
        # is NOT a transpose and would scramble CNN neighbourhoods.
        actor = ActorCNN(env.action_space(env.agents[0]).n, height=env.height, width=env.width)
        critic = CentralCriticCNN(env.num_agents, height=env.height, width=env.width)
        return actor, critic

    def network_config(self, env):
        return {
            "parameter_sharing": True,
            "actor": "JaxMARL three-layer feed-forward CNN",
            "critic": "centralized feed-forward joint-observation CNN",
            "recurrent": False,
            "cnn_channels": [32, 32, 32],
            "dense_hidden": 64,
            "observation_shape": [env.height, env.width, 26],
            "observation_axis_order": "height,width,channels",
            "num_agents": env.num_agents,
            "action_count": env.action_space(env.agents[0]).n,
        }

    @staticmethod
    def decode_state(state):
        maze = np.asarray(state.maze_map)
        height, width = np.asarray(state.wall_map).shape
        padding = (maze.shape[0] - height) // 2
        visible = maze[padding : padding + height, padding : padding + width]
        cells = [
            {
                "x": int(x),
                "y": int(y),
                "object": int(visible[y, x, 0]),
                "status": int(visible[y, x, 2]),
            }
            for y, x in np.argwhere(visible[:, :, 0] != 1)
        ]
        return {
            "step": int(np.asarray(state.step)),
            "done": bool(np.asarray(state.done)),
            "agents": [
                {
                    "position": [int(value) for value in np.asarray(state.agent_pos[index])],
                    "direction": int(np.asarray(state.agent_dir_idx[index])),
                    "inventory": int(np.asarray(state.agent_inv[index])),
                }
                for index in range(len(state.agent_pos))
            ],
            "cells": cells,
        }

    @staticmethod
    def decode_action(action: int, agent: int, env: Any) -> str:
        del agent, env
        names = ("up", "down", "right", "left", "stay", "interact")
        return names[action] if 0 <= action < len(names) else f"action_{action}"


ADAPTER = Adapter()
