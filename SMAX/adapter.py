"""SMAX binary task objective and recurrent MAPPO feature adapter."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import config
from common.networks import CentralCriticRNN
from SMAX.networks import ActorRNN
from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario


class Adapter:
    family = "SMAX"
    recurrent = True

    def __init__(self, family_config: config.FamilyConfig = config.SMAX):
        self.config = family_config

    def make_environment(self, scenario: str, *, random_reset: bool = False):
        del random_reset
        if self.config.learner.max_steps < 2:
            raise ValueError("SMAX episode budget must be at least two transitions")
        return HeuristicEnemySMAX(
            scenario=map_name_to_scenario(scenario),
            # Upstream tests the OLD step counter before incrementing it. Passing
            # H-1 therefore gives exactly H transitions in training and evaluation.
            max_steps=self.config.learner.max_steps - 1,
            see_enemy_actions=True,
            walls_cause_death=True,
            attack_mode="closest",
            enemy_shoots=True,
            use_self_play_reward=False,
        )

    @staticmethod
    def reward_state(state):
        return state.state

    @staticmethod
    def task_reward(previous_state, next_state, environment_rewards, info, agents):
        del previous_state, environment_rewards, info, agents
        inner = next_state.state
        allied = inner.unit_teams == 0
        enemy = inner.unit_teams == 1
        return jnp.asarray(
            jnp.any(inner.unit_alive & allied) & jnp.all(~inner.unit_alive | ~enemy),
            dtype=jnp.float32,
        )

    @staticmethod
    def native_shaping(environment_rewards, info, task_reward, agents):
        del info
        environment = jnp.stack([environment_rewards[agent] for agent in agents], axis=-1)
        return environment - jnp.asarray(task_reward)[..., None]

    def actor_inputs(self, observations, state, env):
        first = observations[env.agents[0]]
        unbatched = first.ndim == 1
        values = jnp.stack([observations[agent] for agent in env.agents], axis=0 if unbatched else 1)
        if unbatched:
            values = values[None]
            step = jnp.asarray(state.state.step)[None]
        else:
            step = jnp.asarray(state.state.step)
        identity = jnp.broadcast_to(jnp.eye(env.num_agents, dtype=jnp.float32), (values.shape[0], env.num_agents, env.num_agents))
        remaining = ((self.config.learner.max_steps - step.astype(jnp.float32)) / self.config.learner.max_steps)[:, None, None]
        remaining = jnp.broadcast_to(remaining, (values.shape[0], env.num_agents, 1))
        result = jnp.concatenate((values.astype(jnp.float32), identity, remaining), axis=-1)
        return result[0] if unbatched else result

    def global_inputs(self, observations, state, env):
        world = jnp.asarray(observations["world_state"], dtype=jnp.float32)
        unbatched = world.ndim == 1
        if unbatched:
            world = world[None]
            step = jnp.asarray(state.state.step)[None]
        else:
            step = jnp.asarray(state.state.step)
        remaining = ((self.config.learner.max_steps - step.astype(jnp.float32)) / self.config.learner.max_steps)[:, None]
        result = jnp.concatenate((world, remaining), axis=-1)
        return result[0] if unbatched else result

    @staticmethod
    def available_actions(state, env):
        if jnp.asarray(state.state.step).ndim == 0:
            masks = env.get_avail_actions(state)
            return jnp.stack([masks[agent] for agent in env.agents], axis=0).astype(bool)
        masks = jax.vmap(env.get_avail_actions)(state)
        return jnp.stack([masks[agent] for agent in env.agents], axis=1).astype(bool)

    @staticmethod
    def done_arrays(dones, env):
        episode_done = jnp.asarray(dones["__all__"])
        dead = jnp.stack([dones[agent] for agent in env.agents], axis=-1)
        # SMAX per-agent flags mean DEATH, not episode termination. Surviving
        # agents must also reset their recurrent memory after victory or timeout.
        return dead | episode_done[..., None], episode_done

    def build_networks(self, env):
        actor = ActorRNN(env.action_space(env.agents[0]).n, self.config.learner.actor_hidden)
        critic = CentralCriticRNN(env.num_agents, self.config.learner.critic_hidden)
        return actor, critic

    def network_config(self, env):
        return {
            "parameter_sharing": True,
            "recurrent": True,
            "actor": "masked MLP + GRU with agent identity",
            "actor_hidden": self.config.learner.actor_hidden,
            "critic": "centralized world-state GRU",
            "critic_hidden": self.config.learner.critic_hidden,
            "num_agents": env.num_agents,
            "action_count": env.action_space(env.agents[0]).n,
        }

    @staticmethod
    def decode_state(state):
        inner = state.state
        positions = np.asarray(inner.unit_positions)
        health = np.asarray(inner.unit_health)
        alive = np.asarray(inner.unit_alive)
        teams = np.asarray(inner.unit_teams)
        unit_types = np.asarray(inner.unit_types)
        return {
            "step": int(np.asarray(inner.step)),
            "done": bool(np.asarray(inner.done)),
            "units": [
                {
                    "index": index,
                    "team": "ally" if int(teams[index]) == 0 else "enemy",
                    "alive": bool(alive[index]),
                    "health": float(health[index]),
                    "position": [float(v) for v in positions[index]],
                    "unit_type": int(unit_types[index]),
                }
                for index in range(len(alive))
            ],
        }

    @staticmethod
    def decode_action(action: int, agent: int, env: Any) -> str:
        del agent, env
        if action < 4:
            return f"move_direction_{action}"
        if action == 4:
            return "stop"
        return f"attack_enemy_slot_{action - 5}"


ADAPTER = Adapter()
