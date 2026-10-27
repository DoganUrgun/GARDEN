"""Observational snapshots preserve the learner and disclose their actual timing."""

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import bootstrap  # noqa: F401
import jax
import numpy as np

import config
from common.evidence import record_trajectories
from common.mappo import train_policy
from Overcooked.adapter import Adapter
import training


class SnapshotTests(unittest.TestCase):
    def test_raw_snapshot_rollouts_do_not_change_parameters_or_learning_curve(self):
        # Forty tiny updates exercise the ordinary TWO-update compiled chunks.
        # Independent random reset/action draws and file I/O happen between
        # those chunks, without obtaining the optimizer's random key.
        learner = replace(config.OVERCOOKED.learner, max_steps=4, num_envs=2,
                          rollout_steps=2, ppo_epochs=1, minibatches=1,
                          actor_hidden=4, critic_hidden=4)
        adapter = Adapter(replace(config.OVERCOOKED, learner=learner))
        env = adapter.make_environment("cramped_room")
        baseline = train_policy(env, adapter, "native", None, 160, 7)
        boundaries, recorded = [], []
        with TemporaryDirectory() as temporary:
            def observer(parameters, update, updates, timesteps):
                boundaries.append((update, updates, timesteps))
                if update in (8, 24):
                    folder = Path(temporary) / str(update)
                    record_trajectories(folder, env, adapter, parameters, "native", None,
                                        300_007, 2, greedy=False)
                    recorded.append(json.loads((folder / "trajectory_arrays.json").read_text()))

            observed = train_policy(env, adapter, "native", None, 160, 7,
                                    snapshot_callback=observer)
        self.assertEqual([item[0] for item in boundaries], list(range(2, 41, 2)))
        self.assertTrue(all(total == 40 and steps == update * 4
                            for update, total, steps in boundaries))
        self.assertEqual(len(recorded), 2)
        self.assertTrue(all(item["seed"] == 300_007 and len(item["episodes"]) == 2
                            for item in recorded))
        self.assertEqual(baseline.curve, observed.curve)
        self.assertEqual(baseline.actual_timesteps, observed.actual_timesteps)
        for before, after in zip(jax.tree_util.tree_leaves(baseline.parameters.actor),
                                 jax.tree_util.tree_leaves(observed.parameters.actor)):
            np.testing.assert_array_equal(before, after)
        for before, after in zip(jax.tree_util.tree_leaves(baseline.parameters.critic),
                                 jax.tree_util.tree_leaves(observed.parameters.critic)):
            np.testing.assert_array_equal(before, after)

    def test_worker_records_actual_chunk_position_and_independent_trace_seeds(self):
        self._check_worker_snapshots(2)

    def test_worker_can_disable_snapshots(self):
        self._check_worker_snapshots(0)

    def _check_worker_snapshots(self, episodes):
        adapter = SimpleNamespace(make_environment=lambda name: object(),
                                  network_config=lambda env: {"test": True})
        evaluation = SimpleNamespace(summary=lambda: {"task_mean": 0.0})

        def fake_train(env, adapter, mode, reward, timesteps, seed,
                       progress_callback, snapshot_callback):
            if episodes == 0:
                self.assertIsNone(snapshot_callback)
            curve = []
            for update in (*range(15, 301, 15), 305):
                row = {"update": update, "timesteps": update * 16_384}
                curve.append(row)
                progress_callback(row, update, 305)
                if snapshot_callback:
                    snapshot_callback("parameters", update, 305, row["timesteps"])
            return SimpleNamespace(parameters="parameters", curve=curve, actual_timesteps=4_997_120)

        with TemporaryDirectory() as temporary, \
                patch.object(config, "SNAPSHOT_FRACTIONS", (0.2, 0.6)), \
                patch.object(training, "adapter_for", return_value=adapter), \
                patch.object(training, "runtime_accelerator", return_value={"backend": "cpu"}), \
                patch.object(training, "train_policy", side_effect=fake_train), \
                patch.object(training, "save_checkpoint") as checkpoints, \
                patch.object(training, "record_trajectories") as traces, \
                patch.object(training, "evaluate_policy", return_value=evaluation), \
                patch.object(training, "write_evaluation"), \
                patch.object(training, "file_hash", return_value="test"), \
                patch.object(training, "write_json", wraps=training.write_json) as writes:
            output = Path(temporary)
            args = SimpleNamespace(family="Overcooked", scenario="cramped_room", seed=7,
                                   mode="native", reward=None, output=output, timesteps=None,
                                   eval_episodes=64, trace_episodes=8, snapshot_episodes=episodes)
            training.run(args)
            result = json.loads((output / "result.json").read_text())
            states = [call.args[1]["status"] for call in writes.call_args_list
                      if call.args[0].name == "progress.json"]
            self.assertIn("evaluation", states)
            self.assertIn("traces", states)
            self.assertEqual(states[-1], "complete")
            self.assertEqual(result["actual_timesteps"], 4_997_120)
            self.assertEqual(traces.call_args_list[-1].args[6:8], (200_007, 8))
            if episodes:
                index = json.loads((output / "training_snapshots" / "index.json").read_text())
                saved = index["snapshots"]
                self.assertEqual([item["update"] for item in saved], [75, 195])
                self.assertEqual([item["requested_fractions"] for item in saved], [[0.2], [0.6]])
                self.assertEqual([item["actual_fraction"] for item in saved], [75 / 305, 195 / 305])
                self.assertTrue(all(item["trace_seed"] == 300_007 for item in saved))
                self.assertEqual(checkpoints.call_count, 3)
                self.assertEqual(traces.call_count, 3)
                self.assertEqual(states.count("snapshots"), 2)
                for call in traces.call_args_list[:2]:
                    self.assertEqual(call.args[6:8], (300_007, 2))
                self.assertEqual(result["training_snapshots"], saved)
            else:
                self.assertFalse((output / "training_snapshots").exists())
                self.assertEqual(checkpoints.call_count, 1)
                self.assertEqual(traces.call_count, 1)
                self.assertEqual(result["training_snapshots"], [])


if __name__ == "__main__":
    unittest.main()
