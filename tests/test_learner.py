"""Small numerical and environment-contract regressions; no LLM or full training."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest

import bootstrap  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np

import config
from common.mappo import _actor_sequence, _critic_sequence, _mask_logits, generalized_advantages
from common.reward import LoadedReward, candidate_components, component_values, native_payoff, reset_memory, validate_source
from SMAX.adapter import Adapter as SmaxAdapter
from Overcooked.adapter import Adapter as OvercookedAdapter


class LearnerContractTests(unittest.TestCase):
    def test_global_terminal_stops_gae_even_when_last_value_is_large(self):
        # Episode 1 gives 1 then 2; episode 2 gives 100 with a bootstrap of 7.
        payoff = jnp.broadcast_to(jnp.array([1., 2., 100.])[:, None, None], (3, 1, 2))
        values = jnp.zeros_like(payoff)
        advantages, targets = generalized_advantages(
            payoff, values, jnp.array([[False], [True], [False]]),
            jnp.full((1, 2), 7.), 1., 1.,
        )
        np.testing.assert_allclose(advantages[:, 0, 0], [3., 2., 107.])
        np.testing.assert_array_equal(advantages, targets)

    def test_gae_retains_team_rewards_for_dead_agents(self):
        # There is intentionally no individual-death cutoff: SMAX awards the
        # eventual team win to sacrificed agents too, just like upstream MAPPO.
        payoff = jnp.array([[[0., 0.]], [[1., 1.]]])
        advantages, _ = generalized_advantages(
            payoff, jnp.zeros_like(payoff), jnp.array([[False], [True]]),
            jnp.zeros((1, 2)), 0.9, 1.,
        )
        np.testing.assert_allclose(advantages[0, 0], [0.9, 0.9])

    def test_smax_survivors_reset_at_global_end(self):
        env = SimpleNamespace(agents=("a", "b"))
        agent_done, global_done = SmaxAdapter.done_arrays({
            "a": jnp.array([False, True]), "b": jnp.array([False, False]),
            "__all__": jnp.array([True, False]),
        }, env)
        np.testing.assert_array_equal(agent_done, [[True, True], [True, False]])
        np.testing.assert_array_equal(global_done, [True, False])

    def test_recurrent_sequence_resets_before_new_observation(self):
        class Accumulator:
            def apply(self, parameters, hidden, inputs):
                del parameters
                next_hidden = hidden + inputs
                return next_hidden, jnp.concatenate((next_hidden, -next_hidden), axis=-1)

        hidden, logits = _actor_sequence(
            Accumulator(), None, jnp.array([[4.]]), jnp.ones((3, 1, 1)),
            jnp.array([[False], [True], [False]]), jnp.ones((3, 1, 2), bool),
        )
        np.testing.assert_array_equal(logits[:, 0, 0], [5., 1., 2.])
        np.testing.assert_array_equal(hidden, [[2.]])

    def test_single_legal_action_has_no_policy_gradient(self):
        mask = jnp.array([False, False, True])
        def log_probability(logits):
            return jax.nn.log_softmax(_mask_logits(logits, mask))[2]
        np.testing.assert_array_equal(jax.grad(log_probability)(jnp.ones(3)), jnp.zeros(3))

    def test_feedforward_replay_vectorization_preserves_outputs_and_gradients(self):
        class FeedForward:
            def apply(self, parameters, hidden, inputs):
                return hidden, jnp.tanh(inputs @ parameters)

        network = FeedForward()
        parameters = jnp.array([[1., 2.], [3., 4.], [5., 6.]])
        inputs = jnp.arange(18, dtype=jnp.float32).reshape((3, 2, 3)) / 20
        hidden = jnp.zeros((2, 4))
        done = jnp.array([[False, False], [True, False], [False, True]])
        mask = jnp.ones((3, 2, 2), dtype=bool)
        def output(params, recurrent):
            return _actor_sequence(network, params, hidden, inputs, done, mask, recurrent=recurrent)[1]
        np.testing.assert_allclose(output(parameters, True), output(parameters, False), atol=1e-6)
        old_grad = jax.grad(lambda p: output(p, True).sum())(parameters)
        new_grad = jax.grad(lambda p: output(p, False).sum())(parameters)
        np.testing.assert_allclose(old_grad, new_grad, atol=1e-6)
        _, old_values = _critic_sequence(network, parameters, hidden, inputs, done)
        _, new_values = _critic_sequence(network, parameters, hidden, inputs, done, recurrent=False)
        np.testing.assert_allclose(old_values, new_values, atol=1e-6)

    def test_smax_task_is_binary_victory_not_damage(self):
        for alive, expected in [([True, True], 0.), ([True, False], 1.),
                                ([False, False], 0.), ([False, True], 0.)]:
            inner = SimpleNamespace(unit_alive=jnp.array(alive), unit_teams=jnp.array([0, 1]))
            task = SmaxAdapter.task_reward(None, SimpleNamespace(state=inner), None, None, None)
            self.assertEqual(float(task), expected)

    def test_native_and_candidate_channels_are_separate(self):
        reward = LoadedReward(
            "SMAX", Path("unused.py"), "test", "", lambda n: (),
            lambda old, actions, new, task, memory: (jnp.array([2., -2.]), memory),
        )
        components, _ = candidate_components(reward, 0.25, None, jnp.zeros(2, int), None, 1., ())
        np.testing.assert_allclose(components.full, [1.5, 0.5])
        np.testing.assert_allclose(native_payoff(1., jnp.array([3., 3.])), [4., 4.])

    def test_component_logging_does_not_change_reward(self):
        reward = LoadedReward(
            "SMAX", Path("unused.py"), "test", "", lambda n: (),
            lambda old, actions, new, task, memory: (jnp.array([2., 2.]), memory),
            lambda old, actions, new, task, memory: {"progress": jnp.array([2., 2.])},
        )
        actions = jnp.zeros(2, int)
        values = component_values(reward, None, actions, None, 0., ())
        components, _ = candidate_components(reward, 0.25, None, actions, None, 0., ())
        np.testing.assert_allclose(values["progress"], [2., 2.])
        np.testing.assert_allclose(components.full, [0.5, 0.5])
        self.assertEqual(component_values(replace(reward, reward_components=None), None, actions, None, 0., ()), {})

    def test_component_shape_is_not_silently_broadcast(self):
        reward = LoadedReward(
            "SMAX", Path("unused.py"), "test", "", lambda n: (), lambda *args: None,
            lambda *args: {"progress": jnp.array(2.)},
        )
        with self.assertRaises(ValueError):
            component_values(reward, None, jnp.zeros(2, int), None, 0., ())

    def test_reward_memory_reset_is_per_episode(self):
        memory = {"x": jnp.array([[1., 2.], [3., 4.]])}
        initial = {"x": jnp.zeros((2, 2))}
        reset = reset_memory(memory, jnp.array([True, False]), initial)
        np.testing.assert_array_equal(reset["x"], [[0., 0.], [3., 4.]])

    def test_reward_cannot_use_jax_file_io(self):
        for imported, call in (
            ("import jax.numpy as jnp", "jnp.load('outside.npy')"),
            ("from jax.numpy import load as read_array", "read_array('outside.npy')"),
        ):
            source = imported + "\ndef initial_memory(n):\n    return ()\n"
            source += "def reward_and_memory(old, actions, new, task, memory):\n    return " + call + ", memory\n"
            with self.assertRaisesRegex(ValueError, "read or write files"):
                validate_source(source, "SMAX")

    def test_reward_callback_ban_resolves_import_aliases(self):
        for imported, call in (
            ("import jax as j", "j.pure_callback(helper, actions, actions)"),
            ("from jax import pure_callback as pc", "pc(helper, actions, actions)"),
        ):
            source = imported + "\ndef initial_memory(n):\n    return ()\ndef helper(x):\n    return x\n"
            source += "def reward_and_memory(old, actions, new, task, memory):\n    return " + call + ", memory\n"
            with self.assertRaisesRegex(ValueError, "Callbacks"):
                validate_source(source, "SMAX")

    def test_smax_has_exact_declared_transition_limit(self):
        learner = replace(config.SMAX.learner, max_steps=4)
        adapter = SmaxAdapter(replace(config.SMAX, learner=learner))
        env = adapter.make_environment(config.SMAX.development[0])
        _, initial = env.reset(jax.random.PRNGKey(0))
        # Keep combat away from termination; exercise the final two time steps.
        state = initial.replace(state=initial.state.replace(step=jnp.asarray(2)))
        actions = {agent: jnp.asarray(4) for agent in env.agents}
        _, state, _, dones, _ = env.step_env(jax.random.PRNGKey(1), state, actions)
        self.assertFalse(bool(dones["__all__"]))
        self.assertEqual(int(state.state.step), 3)
        _, state, _, dones, _ = env.step_env(jax.random.PRNGKey(2), state, actions)
        self.assertTrue(bool(dones["__all__"]))
        self.assertEqual(int(state.state.step), 4)

    def test_overcooked_has_exact_declared_transition_limit(self):
        learner = replace(config.OVERCOOKED.learner, max_steps=4)
        adapter = OvercookedAdapter(replace(config.OVERCOOKED, learner=learner))
        env = adapter.make_environment("cramped_room")
        _, state = env.reset(jax.random.PRNGKey(0))
        state = state.replace(step=jnp.asarray(3))
        actions = {agent: jnp.asarray(4) for agent in env.agents}
        _, state, _, dones, _ = env.step_env(jax.random.PRNGKey(1), state, actions)
        self.assertTrue(bool(dones["__all__"]))
        self.assertEqual(int(state.step), 4)


if __name__ == "__main__":
    unittest.main()
