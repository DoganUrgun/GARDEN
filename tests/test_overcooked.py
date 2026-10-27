"""Spatial fidelity, layout solvability and development-source boundaries."""

from collections import deque
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import bootstrap  # noqa: F401
import jax
import jax.numpy as jnp
import numpy as np

import config
from common.workspace import _copy_reference
from Overcooked.adapter import Adapter
from Overcooked import networks
from jaxmarl.environments.overcooked import overcooked_layouts


class OvercookedTests(unittest.TestCase):
    def test_scenario_split_preserves_budget_and_held_out(self):
        self.assertEqual(config.OVERCOOKED.development, ("cramped_room", "bottleneck", "coord_ring"))
        self.assertEqual(config.OVERCOOKED.held_out, ("forced_coord", "counter_circuit"))
        self.assertEqual(len(config.CANDIDATES), 6)
        self.assertEqual(config.OVERCOOKED.learner.max_steps, 400)

    def test_rectangular_actor_and_critic_preserve_grid_coordinates(self):
        adapter = Adapter()
        for scenario in ("cramped_room", "asymm_advantages", "bottleneck"):
            with self.subTest(scenario=scenario):
                env = adapter.make_environment(scenario)
                observations, state = env.reset(jax.random.PRNGKey(0))
                shape = (env.height, env.width, 26)
                self.assertNotEqual(env.height, env.width)
                self.assertEqual(observations[env.agents[0]].shape, shape)
                self.assertEqual(adapter.network_config(env)["observation_shape"], list(shape))
                actor, critic = adapter.build_networks(env)
                # Unique values expose a reshape error even in mostly empty cells.
                grid = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
                observations = {a: grid + i * 10000 for i, a in enumerate(env.agents)}
                for batched in (False, True):
                    obs = {a: jnp.stack((v, v + 20000)) for a, v in observations.items()} if batched else observations
                    actor_input = adapter.actor_inputs(obs, state, env)
                    critic_input = adapter.global_inputs(obs, state, env)
                    actor_expected = jnp.stack(list(obs.values()), axis=1 if batched else 0).reshape((-1,) + shape)
                    critic_expected = jnp.concatenate(list(obs.values()), axis=-1).reshape((-1, env.height, env.width, 52))
                    for network, values, expected in (
                        (actor, actor_input.reshape((-1, actor_input.shape[-1])), actor_expected),
                        (critic, critic_input.reshape((-1, critic_input.shape[-1])), critic_expected),
                    ):
                        # Inspect the real tensor presented to the first Conv;
                        # replacing layers with identity avoids parameter costs.
                        seen = []
                        def convolution(**kwargs):
                            def apply(x):
                                seen.append(np.asarray(x))
                                return x
                            return apply
                        with patch.object(networks.nn, "Conv", side_effect=convolution), patch.object(networks.nn, "Dense", side_effect=lambda *args, **kwargs: lambda x: x):
                            network.apply({}, jnp.zeros((values.shape[0], 1)), values)
                        np.testing.assert_array_equal(seen[0], expected)

    def test_bottleneck_geometry_and_reachable_fixtures(self):
        layout = overcooked_layouts["bottleneck"]
        self.assertEqual((layout["height"], layout["width"]), (5, 7))
        for key, values in {
            "agent_idx": [9, 11], "onion_pile_idx": [2],
            "plate_pile_idx": [4], "pot_idx": [32, 33], "goal_idx": [29],
        }.items():
            np.testing.assert_array_equal(layout[key], values)
        floor = set(range(35)) - set(map(int, layout["wall_idx"]))
        visited, pending = set(), [9]
        while pending:
            cell = pending.pop()
            if cell in visited:
                continue
            visited.add(cell)
            pending.extend(n for n in (cell - 7, cell + 7, cell - 1, cell + 1) if n in floor - visited)
        self.assertEqual(visited, floor)
        self.assertEqual(len(floor), 13)
        for key in ("onion_pile_idx", "plate_pile_idx", "pot_idx", "goal_idx"):
            for cell in map(int, layout[key]):
                self.assertTrue({cell - 7, cell + 7, cell - 1, cell + 1} & visited)

    def test_bottleneck_can_complete_soup_with_unchanged_dynamics(self):
        adapter = Adapter()
        env = adapter.make_environment("bottleneck")
        _, state = env.reset(jax.random.PRNGKey(0))
        total = 0.
        def step(action, partner=4):
            nonlocal state, total
            _, state, rewards, dones, _ = env.step_env(
                jax.random.PRNGKey(1), state,
                {env.agents[0]: jnp.asarray(action), env.agents[1]: jnp.asarray(partner)},
            )
            total += float(rewards[env.agents[0]])
            self.assertFalse(bool(dones["__all__"]))
        step(4, 2)  # Partner waits at (5,1), leaving plate access open.
        def interact_from(target, facing):
            start = tuple(map(int, state.agent_pos[0]))
            blocked = tuple(map(int, state.agent_pos[1]))
            pending, seen = deque([(start, [])]), {start}
            while pending:
                cell, path = pending.popleft()
                if cell == target:
                    for action in path:
                        step(action)
                    step(facing)  # Face the fixture, which blocks movement.
                    step(5)
                    return
                for action, (dx, dy) in enumerate(((0, -1), (0, 1), (1, 0), (-1, 0))):
                    nxt = (cell[0] + dx, cell[1] + dy)
                    if nxt in seen or nxt == blocked or not (0 <= nxt[0] < env.width and 0 <= nxt[1] < env.height):
                        continue
                    if not bool(state.wall_map[nxt[1], nxt[0]]):
                        seen.add(nxt)
                        pending.append((nxt, path + [action]))
            self.fail(f"No route to {target}")
        for _ in range(3):
            interact_from((2, 1), 0)  # Onion.
            interact_from((4, 3), 1)  # Pot: automatic cooking after third onion.
        for _ in range(21):
            step(4)
        interact_from((4, 1), 0)  # Plate.
        interact_from((4, 3), 1)  # Cooked soup.
        interact_from((1, 3), 1)  # Delivery.
        self.assertEqual(total, 20.)

    def test_development_workspace_keeps_bottleneck_not_held_out_or_witness(self):
        family = config.OVERCOOKED
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference"
            _copy_reference(reference, family)
            path = reference / "runtime/jaxmarl/environments/overcooked/layouts.py"
            source = path.read_text(encoding="utf-8")
            namespace = {}
            exec(compile(source, str(path), "exec"), namespace)
            self.assertEqual(set(namespace["overcooked_layouts"]), set(family.development))
            self.assertNotIn("asymm_advantages", namespace)
            # Private researcher scripts cannot enter the designer source packet.
            self.assertFalse((reference / "diagnostics").exists())
            self.assertFalse(any(reference.rglob("witness.py")))
            for name in family.held_out:
                self.assertNotIn(name, source)


if __name__ == "__main__":
    unittest.main()
