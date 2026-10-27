"""Fixed shared MAPPO: feed-forward Overcooked and recurrent SMAX."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState

import config
from common.networks import zero_hidden
from common.reward import LoadedReward, candidate_components, component_values, initial_batched_memory, native_payoff, reset_memory


class Transition(NamedTuple):
    actor_inputs: jax.Array
    global_inputs: jax.Array
    available_actions: jax.Array
    actions: jax.Array
    old_log_probabilities: jax.Array
    old_values: jax.Array
    payoffs: jax.Array
    task_reward: jax.Array
    native_shaping: jax.Array
    llm_reward: jax.Array
    raw_llm_reward: jax.Array
    agent_done_before: jax.Array
    global_done_before: jax.Array
    global_done: jax.Array
    completed_task_return: jax.Array
    completed_episode_length: jax.Array
    reward_terms: dict[str, jax.Array]


@dataclass(frozen=True)
class PolicyParameters:
    actor: Any
    critic: Any


@dataclass
class TrainOutput:
    parameters: PolicyParameters
    curve: list[dict[str, float]]
    requested_timesteps: int
    actual_timesteps: int


@dataclass
class Evaluation:
    task_returns: np.ndarray
    training_returns: np.ndarray
    native_shaping_returns: np.ndarray
    llm_returns: np.ndarray
    raw_llm_returns: np.ndarray
    action_counts: np.ndarray
    policy_entropy: float
    episode_lengths: np.ndarray

    def summary(self) -> dict[str, float]:
        task = np.asarray(self.task_returns, dtype=np.float64)
        return {
            "task_mean": float(task.mean()),
            "task_std": float(task.std(ddof=1)) if task.size > 1 else 0.0,
            "task_sem": float(task.std(ddof=1) / np.sqrt(task.size)) if task.size > 1 else 0.0,
            "training_return_mean": float(np.asarray(self.training_returns).mean()),
            "native_shaping_return_mean": float(np.asarray(self.native_shaping_returns).mean()),
            "llm_return_mean": float(np.asarray(self.llm_returns).mean()),
            "raw_llm_return_mean": float(np.asarray(self.raw_llm_returns).mean()),
            "policy_entropy": float(self.policy_entropy),
            "episode_length_mean": float(np.asarray(self.episode_lengths).mean()),
        }


def _log_probability(logits, actions):
    return jnp.take_along_axis(jax.nn.log_softmax(logits), actions[..., None], axis=-1)[..., 0]


def _entropy(logits):
    log_prob = jax.nn.log_softmax(logits)
    return -jnp.sum(jax.nn.softmax(logits) * log_prob, axis=-1)


def _mask_logits(logits, available):
    return jnp.where(available, logits, jnp.asarray(-1e10, dtype=logits.dtype))


def _select_done(done, reset_value, value):
    shape = done.shape + (1,) * (value.ndim - done.ndim)
    return jnp.where(done.reshape(shape), reset_value, value)


def _actor_sequence(actor, parameters, hidden, inputs, done_before, available, *, recurrent=True):
    if not recurrent:
        # A feed-forward CNN has no temporal dependency during PPO replay.
        # Batch time with environment to avoid hundreds of tiny serial GPU calls.
        steps, batch = inputs.shape[:2]
        flat_hidden = jnp.broadcast_to(hidden, (steps,) + hidden.shape).reshape((steps * batch, -1))
        _, logits = actor.apply(parameters, flat_hidden, inputs.reshape((steps * batch, -1)))
        logits = logits.reshape((steps, batch, -1))
        return hidden, _mask_logits(logits, available)

    def step(carry, values):
        observation, done, mask = values
        carry = jnp.where(done[:, None], 0.0, carry)
        carry, logits = actor.apply(parameters, carry, observation)
        return carry, _mask_logits(logits, mask)

    return jax.lax.scan(step, hidden, (inputs, done_before, available))


def _critic_sequence(critic, parameters, hidden, inputs, done_before, *, recurrent=True):
    if not recurrent:
        steps, batch = inputs.shape[:2]
        flat_hidden = jnp.broadcast_to(hidden, (steps,) + hidden.shape).reshape((steps * batch, -1))
        _, values = critic.apply(parameters, flat_hidden, inputs.reshape((steps * batch, -1)))
        return hidden, values.reshape((steps, batch, -1))

    def step(carry, values):
        observation, done = values
        carry = jnp.where(done[:, None], 0.0, carry)
        return critic.apply(parameters, carry, observation)

    return jax.lax.scan(step, hidden, (inputs, done_before))


def generalized_advantages(payoffs, values, episode_done, last_value, gamma, gae_lambda):
    """Team-episode GAE, including post-death team reward but no next-episode value."""
    def step(carry, transition):
        advantage, next_value = carry
        payoff, value, done = transition
        continuation = 1.0 - done.astype(jnp.float32)[..., None]
        delta = payoff + gamma * next_value * continuation - value
        advantage = delta + gamma * gae_lambda * continuation * advantage
        return (advantage, value), advantage

    _, advantages = jax.lax.scan(
        step, (jnp.zeros_like(last_value), last_value),
        (payoffs, values, episode_done), reverse=True,
    )
    return advantages, advantages + values


def initialize_parameters(env, adapter, seed: int) -> PolicyParameters:
    learner = adapter.config.learner
    actor, critic = adapter.build_networks(env)
    key, reset_key, actor_key, critic_key = jax.random.split(jax.random.PRNGKey(seed), 4)
    del key
    observations, state = env.reset(reset_key)
    actor_inputs = adapter.actor_inputs(observations, state, env)
    global_inputs = adapter.global_inputs(observations, state, env)
    actor_parameters = actor.init(actor_key, zero_hidden(env.num_agents, learner.actor_hidden), actor_inputs)
    critic_parameters = critic.init(critic_key, zero_hidden(1, learner.critic_hidden), global_inputs[None])
    return PolicyParameters(actor_parameters, critic_parameters)


def save_checkpoint(folder: Path, parameters: PolicyParameters, metadata: dict[str, Any]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "actor.msgpack").write_bytes(serialization.to_bytes(parameters.actor))
    (folder / "critic.msgpack").write_bytes(serialization.to_bytes(parameters.critic))
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_checkpoint(env, adapter, folder: Path) -> PolicyParameters:
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("family") != adapter.family:
        raise ValueError("Checkpoint family does not match adapter")
    template = initialize_parameters(env, adapter, int(metadata["seed"]))
    return PolicyParameters(
        serialization.from_bytes(template.actor, (folder / "actor.msgpack").read_bytes()),
        serialization.from_bytes(template.critic, (folder / "critic.msgpack").read_bytes()),
    )


def actor_step(env, adapter, actor_parameters, observations, state, hidden, key, *, greedy: bool):
    actor, _ = adapter.build_networks(env)
    inputs = adapter.actor_inputs(observations, state, env)
    available = adapter.available_actions(state, env)
    hidden, logits = actor.apply(actor_parameters, hidden, inputs)
    logits = _mask_logits(logits, available)
    actions = jnp.argmax(logits, axis=-1) if greedy else jax.random.categorical(key, logits, axis=-1)
    return hidden, actions.astype(jnp.int32), logits


def _learning_rate(learner, optimizer_steps):
    if learner.learning_rate_schedule == "constant" or optimizer_steps <= 1:
        return optax.constant_schedule(learner.learning_rate)
    steps_per_update = learner.ppo_epochs * learner.minibatches
    updates = max(1, optimizer_steps // steps_per_update)
    if learner.learning_rate_schedule == "linear":
        # JaxMARL's SMAX reference keeps the rate fixed within each PPO
        # update, then decreases it linearly between updates.
        return lambda count: learner.learning_rate * (
            1.0 - (count // steps_per_update) / updates
        )
    warmup_updates = min(updates - 1, max(0, int(updates * learner.warmup_fraction)))
    warmup_steps = warmup_updates * steps_per_update
    warmup = optax.linear_schedule(0.0, learner.learning_rate, max(1, warmup_steps))
    cosine = optax.cosine_decay_schedule(
        learner.learning_rate,
        max(1, (updates - warmup_updates) * steps_per_update),
    )
    return optax.join_schedules((warmup, cosine), (warmup_steps,)) if warmup_steps else cosine


def train_policy(env, adapter, mode: str, reward: LoadedReward | None, timesteps: int, seed: int, progress_callback=None, snapshot_callback=None) -> TrainOutput:
    """Train with optional read-only observation at the existing chunk boundaries.

    The observer receives parameters, completed/total updates, and completed
    environment steps. It receives neither the optimizer nor the training RNG;
    diagnostic rollouts must use a separate random stream. Adding an observer
    does not split compiled chunks or change any learning operation.
    """
    if mode not in {"native", "candidate"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if (mode == "candidate") != (reward is not None):
        raise ValueError("Candidate mode requires exactly one generated reward")
    if reward is not None and reward.family != adapter.family:
        raise ValueError("Generated reward family does not match environment adapter")

    learner = adapter.config.learner
    num_envs, num_agents, rollout_steps = learner.num_envs, env.num_agents, learner.rollout_steps
    if num_envs % learner.minibatches:
        raise ValueError("num_envs must be divisible by minibatches for recurrent PPO")
    if timesteps <= 0:
        raise ValueError("Training timesteps must be positive")
    updates = max(1, timesteps // (num_envs * rollout_steps))
    actual_timesteps = updates * num_envs * rollout_steps
    optimizer_steps = updates * learner.ppo_epochs * learner.minibatches
    actor, critic = adapter.build_networks(env)
    initial = initialize_parameters(env, adapter, seed)
    optimizer = optax.chain(
        optax.clip_by_global_norm(learner.max_grad_norm),
        optax.adam(_learning_rate(learner, optimizer_steps), eps=1e-5),
    )
    actor_state = TrainState.create(apply_fn=actor.apply, params=initial.actor, tx=optimizer)
    critic_state = TrainState.create(apply_fn=critic.apply, params=initial.critic, tx=optimizer)
    reset_many, step_many = jax.vmap(env.reset), jax.vmap(env.step_env)
    key = jax.random.PRNGKey(seed)
    key, reset_key = jax.random.split(key)
    observations, environment_state = reset_many(jax.random.split(reset_key, num_envs))
    actor_hidden = jnp.zeros((num_envs, num_agents, learner.actor_hidden), dtype=jnp.float32)
    critic_hidden = zero_hidden(num_envs, learner.critic_hidden)
    agent_done = jnp.zeros((num_envs, num_agents), dtype=bool)
    global_done = jnp.zeros((num_envs,), dtype=bool)
    episode_task = jnp.zeros((num_envs,), dtype=jnp.float32)
    if reward is None:
        memory = fresh_memory = ()
    else:
        fresh_memory = initial_batched_memory(reward, num_envs, num_agents)
        memory = fresh_memory

    def collect_rollout(actor_parameters, critic_parameters, observations, environment_state, memory, actor_hidden, critic_hidden, agent_done, global_done, episode_task, key):
        initial_actor_hidden, initial_critic_hidden = actor_hidden, critic_hidden

        def step(carry, _):
            observations, state, memory, actor_hidden, critic_hidden, agent_done, global_done, episode_task, key = carry
            key, action_key, step_key, reset_key = jax.random.split(key, 4)
            actor_hidden = jnp.where(agent_done[..., None], 0.0, actor_hidden)
            critic_hidden = jnp.where(global_done[:, None], 0.0, critic_hidden)
            local = adapter.actor_inputs(observations, state, env)
            central = adapter.global_inputs(observations, state, env)
            available = adapter.available_actions(state, env)
            flat_local = local.reshape((num_envs * num_agents, -1))
            flat_available = available.reshape((num_envs * num_agents, -1))
            flat_hidden = actor_hidden.reshape((num_envs * num_agents, learner.actor_hidden))
            next_flat_hidden, logits = actor.apply(actor_parameters, flat_hidden, flat_local)
            logits = _mask_logits(logits, flat_available)
            actions_flat = jax.random.categorical(action_key, logits, axis=-1).astype(jnp.int32)
            log_prob = _log_probability(logits, actions_flat)
            actions = actions_flat.reshape((num_envs, num_agents))
            next_actor_hidden = next_flat_hidden.reshape((num_envs, num_agents, learner.actor_hidden))
            next_critic_hidden, values = critic.apply(critic_parameters, critic_hidden, central)
            action_dict = {agent: actions[:, index] for index, agent in enumerate(env.agents)}
            next_obs_raw, next_state_raw, environment_rewards, dones, info = step_many(
                jax.random.split(step_key, num_envs), state, action_dict
            )
            task = jax.vmap(
                lambda old, new, rewards, step_info: adapter.task_reward(
                    old, new, rewards, step_info, env.agents
                )
            )(state, next_state_raw, environment_rewards, info)
            next_agent_done, next_global_done = adapter.done_arrays(dones, env)
            if mode == "native":
                native = adapter.native_shaping(environment_rewards, info, task, env.agents)
                payoffs = jax.vmap(native_payoff)(task, native)
                llm = raw_llm = jnp.zeros_like(payoffs)
                next_memory = memory
                reward_terms = {}
            else:
                components, next_memory = jax.vmap(
                    lambda old, act, new, task_value, mem: candidate_components(
                        reward, adapter.config.reward_weight, old, act, new, task_value, mem
                    )
                )(adapter.reward_state(state), actions, adapter.reward_state(next_state_raw), task, memory)
                payoffs, llm, raw_llm = components.full, components.llm, components.raw_llm
                native = jnp.zeros_like(payoffs)
                reward_terms = jax.vmap(
                    lambda old, act, new, task_value, mem: component_values(
                        reward, old, act, new, task_value, mem
                    )
                )(adapter.reward_state(state), actions, adapter.reward_state(next_state_raw), task, memory)
            reset_obs, reset_state = reset_many(jax.random.split(reset_key, num_envs))
            next_obs = jax.tree_util.tree_map(lambda fresh, current: _select_done(next_global_done, fresh, current), reset_obs, next_obs_raw)
            next_state = jax.tree_util.tree_map(lambda fresh, current: _select_done(next_global_done, fresh, current), reset_state, next_state_raw)
            if mode == "candidate":
                next_memory = reset_memory(next_memory, next_global_done, fresh_memory)
            next_episode_task = episode_task + task
            completed = jnp.where(next_global_done, next_episode_task, jnp.nan)
            completed_length = jnp.where(
                next_global_done, adapter.reward_state(next_state_raw).step, jnp.nan
            )
            next_episode_task = jnp.where(next_global_done, 0.0, next_episode_task)
            transition = Transition(
                local, central, available, actions,
                log_prob.reshape((num_envs, num_agents)), values, payoffs, task,
                native, llm, raw_llm, agent_done, global_done, next_global_done, completed,
                completed_length, reward_terms,
            )
            return (
                next_obs, next_state, next_memory, next_actor_hidden, next_critic_hidden,
                next_agent_done, next_global_done, next_episode_task, key,
            ), transition

        final, transitions = jax.lax.scan(
            step,
            (observations, environment_state, memory, actor_hidden, critic_hidden, agent_done, global_done, episode_task, key),
            None,
            length=rollout_steps,
        )
        final_obs, final_state, _, _, final_critic_hidden, _, final_global_done, _, _ = final
        central = adapter.global_inputs(final_obs, final_state, env)
        _, last_value = critic.apply(critic_parameters, jnp.where(final_global_done[:, None], 0.0, final_critic_hidden), central)

        advantages, targets = generalized_advantages(
            transitions.payoffs, transitions.old_values, transitions.global_done,
            last_value, learner.gamma, learner.gae_lambda,
        )
        return final, transitions, advantages, targets, initial_actor_hidden, initial_critic_hidden

    def update_networks(actor_state, critic_state, transitions, advantages, targets, initial_actor_hidden, initial_critic_hidden, key):
        envs_per_minibatch = num_envs // learner.minibatches

        def epoch_step(carry, _):
            actor_state, critic_state, key = carry
            key, permutation_key = jax.random.split(key)
            permutation = jax.random.permutation(permutation_key, num_envs)
            shuffled_transitions = jax.tree_util.tree_map(lambda value: jnp.take(value, permutation, axis=1), transitions)
            shuffled_advantages = jnp.take(advantages, permutation, axis=1)
            shuffled_targets = jnp.take(targets, permutation, axis=1)
            shuffled_actor_hidden = jnp.take(initial_actor_hidden, permutation, axis=0)
            shuffled_critic_hidden = jnp.take(initial_critic_hidden, permutation, axis=0)

            def split(value):
                reshaped = value.reshape((rollout_steps, learner.minibatches, envs_per_minibatch) + value.shape[2:])
                return jnp.swapaxes(reshaped, 0, 1)

            transition_batches = jax.tree_util.tree_map(split, shuffled_transitions)
            advantage_batches, target_batches = split(shuffled_advantages), split(shuffled_targets)
            actor_hidden_batches = shuffled_actor_hidden.reshape((learner.minibatches, envs_per_minibatch, num_agents, learner.actor_hidden))
            critic_hidden_batches = shuffled_critic_hidden.reshape((learner.minibatches, envs_per_minibatch, learner.critic_hidden))

            def minibatch_step(states, batch):
                actor_state, critic_state = states
                transition, advantage, target, actor_initial, critic_initial = batch

                def actor_loss(parameters):
                    flat_inputs = transition.actor_inputs.reshape((rollout_steps, envs_per_minibatch * num_agents, -1))
                    flat_done = transition.agent_done_before.reshape((rollout_steps, envs_per_minibatch * num_agents))
                    flat_available = transition.available_actions.reshape((rollout_steps, envs_per_minibatch * num_agents, -1))
                    flat_initial = actor_initial.reshape((envs_per_minibatch * num_agents, learner.actor_hidden))
                    _, logits = _actor_sequence(actor, parameters, flat_initial, flat_inputs, flat_done, flat_available, recurrent=adapter.recurrent)
                    actions = transition.actions.reshape((rollout_steps, envs_per_minibatch * num_agents))
                    old_log = transition.old_log_probabilities.reshape(actions.shape)
                    log_probability = _log_probability(logits, actions)
                    log_ratio = log_probability - old_log
                    ratio = jnp.exp(log_ratio)
                    normalized = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                    normalized = normalized.reshape(actions.shape)
                    unclipped = ratio * normalized
                    clipped = jnp.clip(ratio, 1.0 - learner.clip_eps, 1.0 + learner.clip_eps) * normalized
                    policy_loss = -jnp.minimum(unclipped, clipped).mean()
                    entropy = _entropy(logits).mean()
                    return policy_loss - learner.entropy_coef * entropy, (
                        policy_loss,
                        entropy,
                        ((ratio - 1.0) - log_ratio).mean(),
                        (jnp.abs(ratio - 1.0) > learner.clip_eps).mean(),
                    )

                def critic_loss(parameters):
                    _, values = _critic_sequence(critic, parameters, critic_initial, transition.global_inputs, transition.global_done_before, recurrent=adapter.recurrent)
                    clipped = transition.old_values + jnp.clip(values - transition.old_values, -learner.clip_eps, learner.clip_eps)
                    loss = 0.5 * jnp.maximum(jnp.square(values - target), jnp.square(clipped - target)).mean()
                    return learner.value_coef * loss, loss

                (actor_value, actor_aux), actor_grads = jax.value_and_grad(actor_loss, has_aux=True)(actor_state.params)
                (critic_value, value_loss), critic_grads = jax.value_and_grad(critic_loss, has_aux=True)(critic_state.params)
                actor_state = actor_state.apply_gradients(grads=actor_grads)
                critic_state = critic_state.apply_gradients(grads=critic_grads)
                policy_loss, entropy, approximate_kl, clip_fraction = actor_aux
                metrics = {
                    "loss": actor_value + critic_value,
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "approximate_kl": approximate_kl,
                    "clip_fraction": clip_fraction,
                    "actor_grad_norm": optax.global_norm(actor_grads),
                    "critic_grad_norm": optax.global_norm(critic_grads),
                }
                return (actor_state, critic_state), metrics

            (actor_state, critic_state), metrics = jax.lax.scan(
                minibatch_step,
                (actor_state, critic_state),
                (transition_batches, advantage_batches, target_batches, actor_hidden_batches, critic_hidden_batches),
            )
            return (actor_state, critic_state, key), metrics

        (actor_state, critic_state, key), metrics = jax.lax.scan(epoch_step, (actor_state, critic_state, key), None, length=learner.ppo_epochs)
        return actor_state, critic_state, key, jax.tree_util.tree_map(jnp.mean, metrics)

    action_dim = env.action_space(env.agents[0]).n

    def training_update(runner, _):
        (
            actor_state,
            critic_state,
            observations,
            environment_state,
            memory,
            actor_hidden,
            critic_hidden,
            agent_done,
            global_done,
            episode_task,
            key,
        ) = runner
        final, transitions, advantages, targets, initial_actor_hidden, initial_critic_hidden = collect_rollout(
            actor_state.params,
            critic_state.params,
            observations,
            environment_state,
            memory,
            actor_hidden,
            critic_hidden,
            agent_done,
            global_done,
            episode_task,
            key,
        )
        (
            observations,
            environment_state,
            memory,
            actor_hidden,
            critic_hidden,
            agent_done,
            global_done,
            episode_task,
            key,
        ) = final
        actor_state, critic_state, key, loss_metrics = update_networks(
            actor_state,
            critic_state,
            transitions,
            advantages,
            targets,
            initial_actor_hidden,
            initial_critic_hidden,
            key,
        )
        completed = transitions.completed_task_return
        completed_mask = jnp.isfinite(completed)
        completed_count = jnp.sum(completed_mask)
        completed_mean = jnp.sum(jnp.where(completed_mask, completed, 0.0)) / jnp.maximum(
            completed_count, 1
        )
        actions = transitions.actions.reshape((-1,))
        metrics = {
            **loss_metrics,
            "completed_episodes": completed_count.astype(jnp.float32),
            "completed_task_return_mean": completed_mean,
            "completed_episode_length_mean": jnp.sum(jnp.where(
                completed_mask, transitions.completed_episode_length, 0.0
            )) / jnp.maximum(completed_count, 1),
            "rollout_task_reward_mean": jnp.mean(transitions.task_reward),
            "rollout_training_reward_mean": jnp.mean(transitions.payoffs),
            "rollout_native_shaping_mean": jnp.mean(transitions.native_shaping),
            "rollout_llm_reward_mean": jnp.mean(transitions.llm_reward),
            "action_counts": jnp.bincount(actions, length=action_dim),
        }
        # True on-training-rollout statistics, not a handful of final-policy traces.
        for prefix, values in (
            ("rollout_llm_reward", transitions.llm_reward),
            ("rollout_raw_llm_reward", transitions.raw_llm_reward),
        ):
            metrics.update({f"{prefix}_{name}": function(values) for name, function in (
                ("mean", jnp.mean), ("std", jnp.std), ("min", jnp.min), ("max", jnp.max)
            )})
        for name, values in transitions.reward_terms.items():
            for statistic, function in (("mean", jnp.mean), ("std", jnp.std), ("min", jnp.min), ("max", jnp.max)):
                metrics[f"raw_component_{name}_{statistic}"] = function(values)
            metrics[f"component_{name}_mean"] = adapter.config.reward_weight * jnp.mean(values)
        next_runner = (
            actor_state,
            critic_state,
            observations,
            environment_state,
            memory,
            actor_hidden,
            critic_hidden,
            agent_done,
            global_done,
            episode_task,
            key,
        )
        return next_runner, metrics

    runner = (
        actor_state,
        critic_state,
        observations,
        environment_state,
        memory,
        actor_hidden,
        critic_hidden,
        agent_done,
        global_done,
        episode_task,
        key,
    )

    interval = max(1, updates // 20)
    compiled_chunks = {}
    curve = []
    update_index = 0
    while update_index < updates:
        length = min(interval, updates - update_index)
        if length not in compiled_chunks:
            compiled_chunks[length] = jax.jit(
                lambda current, chunk_length=length: jax.lax.scan(
                    training_update, current, None, length=chunk_length
                )
            )
        runner, chunk_metrics = compiled_chunks[length](runner)
        host_metrics = jax.tree_util.tree_map(np.asarray, chunk_metrics)
        for offset in range(length):
            absolute = update_index + offset + 1
            completed_count = int(host_metrics["completed_episodes"][offset])
            action_counts = host_metrics["action_counts"][offset]
            row = {
                "update": float(absolute),
                "timesteps": float(absolute * num_envs * rollout_steps),
                "completed_episodes": float(completed_count),
                "completed_task_return_mean": (
                    float(host_metrics["completed_task_return_mean"][offset])
                    if completed_count
                    else None
                ),
                "completed_episode_length_mean": (
                    float(host_metrics["completed_episode_length_mean"][offset])
                    if completed_count else None
                ),
            }
            for name in host_metrics:
                if name not in row and name != "action_counts":
                    row[name] = float(host_metrics[name][offset])
            finite_fields = tuple(name for name in row if row[name] is not None)
            invalid = [name for name in finite_fields if not np.isfinite(row[name])]
            if invalid:
                raise FloatingPointError(
                    f"Non-finite MAPPO values at update {absolute}: {', '.join(invalid)}"
                )
            total_actions = max(1, int(action_counts.sum()))
            row.update(
                {
                    f"action_{index}_fraction": float(count / total_actions)
                    for index, count in enumerate(action_counts)
                }
            )
            curve.append(row)
        update_index += length
        if progress_callback is not None:
            progress_callback(curve[-1], update_index, updates)
        if snapshot_callback is not None:
            snapshot_callback(
                PolicyParameters(runner[0].params, runner[1].params),
                update_index, updates, update_index * num_envs * rollout_steps,
            )
    actor_state, critic_state = runner[0], runner[1]
    return TrainOutput(PolicyParameters(actor_state.params, critic_state.params), curve, timesteps, actual_timesteps)


def evaluate_policy(env, adapter, parameters, mode: str, reward: LoadedReward | None, episodes: int, seed: int, *, greedy: bool) -> Evaluation:
    if mode not in {"native", "candidate"} or episodes <= 0:
        raise ValueError("Evaluation requires native/candidate mode and positive episodes")
    if (mode == "candidate") != (reward is not None):
        raise ValueError("Evaluation reward mode is inconsistent")
    learner, num_agents = adapter.config.learner, env.num_agents
    action_dim = env.action_space(env.agents[0]).n
    actor, _ = adapter.build_networks(env)
    reset_many, step_many = jax.vmap(env.reset), jax.vmap(env.step_env)

    def evaluate(key):
        key, reset_key = jax.random.split(key)
        observations, state = reset_many(jax.random.split(reset_key, episodes))
        hidden = jnp.zeros((episodes, num_agents, learner.actor_hidden), dtype=jnp.float32)
        done = jnp.zeros((episodes,), dtype=bool)
        task_return = jnp.zeros((episodes,), dtype=jnp.float32)
        training_return = jnp.zeros((episodes, num_agents), dtype=jnp.float32)
        native_return = jnp.zeros_like(training_return)
        llm_return = jnp.zeros_like(training_return)
        raw_llm_return = jnp.zeros_like(training_return)
        action_counts = jnp.zeros((action_dim,), dtype=jnp.int32)
        entropy_total = active_action_total = jnp.asarray(0.0, dtype=jnp.float32)
        episode_lengths = jnp.zeros((episodes,), dtype=jnp.int32)
        if reward is None:
            memory = fresh_memory = ()
        else:
            fresh_memory = initial_batched_memory(reward, episodes, num_agents)
            memory = fresh_memory

        def step(carry, _):
            observations, state, memory, hidden, done, task_return, training_return, native_return, llm_return, raw_llm_return, action_counts, entropy_total, active_action_total, episode_lengths, key = carry
            key, action_key, step_key, reset_key = jax.random.split(key, 4)
            active = ~done
            inputs = adapter.actor_inputs(observations, state, env)
            available = adapter.available_actions(state, env)
            flat_inputs = inputs.reshape((episodes * num_agents, -1))
            flat_hidden = hidden.reshape((episodes * num_agents, learner.actor_hidden))
            flat_available = available.reshape((episodes * num_agents, action_dim))
            next_hidden, logits = actor.apply(parameters.actor, flat_hidden, flat_inputs)
            logits = _mask_logits(logits, flat_available)
            actions_flat = (jnp.argmax(logits, axis=-1) if greedy else jax.random.categorical(action_key, logits, axis=-1)).astype(jnp.int32)
            actions = actions_flat.reshape((episodes, num_agents))
            action_dict = {agent: actions[:, index] for index, agent in enumerate(env.agents)}
            next_obs, next_state, environment_rewards, dones, info = step_many(jax.random.split(step_key, episodes), state, action_dict)
            task = jax.vmap(
                lambda old, new, rewards, step_info: adapter.task_reward(
                    old, new, rewards, step_info, env.agents
                )
            )(state, next_state, environment_rewards, info)
            _, step_done = adapter.done_arrays(dones, env)
            if mode == "native":
                native = adapter.native_shaping(environment_rewards, info, task, env.agents)
                payoffs = jax.vmap(native_payoff)(task, native)
                llm = raw_llm = jnp.zeros_like(payoffs)
                next_memory = memory
            else:
                components, next_memory = jax.vmap(
                    lambda old, act, new, task_value, mem: candidate_components(
                        reward, adapter.config.reward_weight, old, act, new, task_value, mem
                    )
                )(adapter.reward_state(state), actions, adapter.reward_state(next_state), task, memory)
                payoffs, llm, raw_llm = components.full, components.llm, components.raw_llm
                native = jnp.zeros_like(payoffs)
            active_float, active_agents = active.astype(jnp.float32), active.astype(jnp.float32)[:, None]
            done = done | step_done
            reset_obs, reset_state = reset_many(jax.random.split(reset_key, episodes))
            next_obs = jax.tree_util.tree_map(lambda fresh, current: _select_done(step_done, fresh, current), reset_obs, next_obs)
            next_state = jax.tree_util.tree_map(lambda fresh, current: _select_done(step_done, fresh, current), reset_state, next_state)
            if mode == "candidate":
                next_memory = reset_memory(next_memory, step_done, fresh_memory)
            one_hot = jax.nn.one_hot(actions_flat, action_dim, dtype=jnp.int32).reshape((episodes, num_agents, action_dim))
            active_counts = active[:, None, None].astype(jnp.int32)
            return (
                next_obs, next_state, next_memory,
                next_hidden.reshape((episodes, num_agents, learner.actor_hidden)), done,
                task_return + active_float * task,
                training_return + active_agents * payoffs,
                native_return + active_agents * native,
                llm_return + active_agents * llm,
                raw_llm_return + active_agents * raw_llm,
                action_counts + jnp.sum(one_hot * active_counts, axis=(0, 1)),
                entropy_total + jnp.sum(_entropy(logits).reshape((episodes, num_agents)) * active_agents),
                active_action_total + jnp.sum(active_agents) * num_agents,
                episode_lengths + active.astype(jnp.int32),
                key,
            ), None

        final, _ = jax.lax.scan(
            step,
            (observations, state, memory, hidden, done, task_return, training_return, native_return, llm_return, raw_llm_return, action_counts, entropy_total, active_action_total, episode_lengths, key),
            None,
            length=learner.max_steps,
        )
        return final[5], final[6], final[7], final[8], final[9], final[10], final[11] / jnp.maximum(final[12], 1.0), final[13], final[4]

    values = jax.jit(evaluate)(jax.random.PRNGKey(seed))
    result = Evaluation(
        np.asarray(values[0], dtype=np.float64), np.asarray(values[1], dtype=np.float64),
        np.asarray(values[2], dtype=np.float64), np.asarray(values[3], dtype=np.float64),
        np.asarray(values[4], dtype=np.float64), np.asarray(values[5], dtype=np.int64),
        float(np.asarray(values[6])), np.asarray(values[7], dtype=np.int64),
    )
    if not np.asarray(values[8]).all():
        raise RuntimeError("Evaluation horizon ended before environment termination; check episode limit")
    leaves = (result.task_returns, result.training_returns, result.native_shaping_returns, result.llm_returns, result.raw_llm_returns, np.asarray(result.policy_entropy))
    if not all(np.isfinite(value).all() for value in leaves):
        raise FloatingPointError("Evaluation produced a non-finite reward or policy value")
    return result


def write_curve(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["update"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation(path: Path, evaluation: Evaluation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("episode", "task_return", "training_return_mean", "native_shaping_return_mean", "llm_return_mean", "raw_llm_return_mean", "episode_length")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(evaluation.task_returns.shape[0]):
            writer.writerow({
                "episode": index,
                "task_return": float(evaluation.task_returns[index]),
                "training_return_mean": float(evaluation.training_returns[index].mean()),
                "native_shaping_return_mean": float(evaluation.native_shaping_returns[index].mean()),
                "llm_return_mean": float(evaluation.llm_returns[index].mean()),
                "raw_llm_return_mean": float(evaluation.raw_llm_returns[index].mean()),
                "episode_length": int(evaluation.episode_lengths[index]),
            })
