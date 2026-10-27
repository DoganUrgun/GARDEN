"""Offline probe budget, isolation and worker lifecycle checks; no training/API."""

import asyncio
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import AsyncMock, patch

import config
from common import learning_probes as probes
from common.records import Run, file_hash, read_json, write_json
from common.workspace import protected_manifest


class FakeProcess:
    pid = 123456789

    def __init__(self, returncode=0, wait_error=None):
        self.returncode = returncode
        self.wait_error = wait_error
        self.wait_count = 0

    async def wait(self):
        self.wait_count += 1
        if self.wait_error is not None:
            error, self.wait_error = self.wait_error, None
            raise error
        return self.returncode


class LearningProbeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = Run("test", self.root / "run", "SMAX")
        self.workspace = self.run.folder / "workspaces" / "Gen1a"
        for name in ("reference", "evidence", "scratch", "output"):
            (self.workspace / name).mkdir(parents=True)
        (self.workspace / "reference" / "source.py").write_text("# permitted mechanics\n")
        self.reward = self.workspace / "output" / "reward.py"
        self.reward.write_text("# standalone mock\n")
        write_json(self.run.state_file, {"phase": "reward_design", "reward_locked": False})
        self.accelerator = {"backend": "gpu", "devices": ["cuda:0"], "process_count": 1}
        write_json(self.run.folder / "manifest.json", {"accelerator": self.accelerator})
        write_json(self.workspace / "workspace_manifest.json", {"protected_files": protected_manifest(self.workspace)})
        self.validation = patch.object(probes, "_validate", return_value={"valid": True, "has_reward_components": True})
        self.validate = self.validation.start()
        self.addCleanup(self.validation.stop)
        self.network = {"recurrent": True, "num_agents": 8}
        network_patch = patch.object(probes, "_network_config", return_value=self.network)
        network_patch.start()
        self.addCleanup(network_patch.stop)
        signature_patch = patch.object(probes, "_source_signature", return_value={"source_hash": "fixed", "dependencies": {}})
        signature_patch.start()
        self.addCleanup(signature_patch.stop)
        self.tool = probes.LearningProbes(self.run, config.SMAX, "Gen1a", self.workspace)
        self.spawn_patch = patch.object(probes.asyncio, "create_subprocess_exec", side_effect=self.spawn)
        self.spawn_mock = self.spawn_patch.start()
        self.addCleanup(self.spawn_patch.stop)
        self.worker = FakeProcess()
        self.result_overrides = {}
        self.after_spawn = None

    def result(self, command):
        args = dict(zip(command[2::2], command[3::2]))
        family = config.SMAX
        batch = family.learner.num_envs * family.learner.rollout_steps
        return {
            "family": family.name, "scenario": args["--scenario"], "seed": 0,
            "mode": "candidate", "reward_hash": file_hash(Path(args["--reward"])),
            "requested_timesteps": family.timesteps,
            "actual_timesteps": max(1, family.timesteps // batch) * batch,
            "learner": family.learner.to_dict(), "evaluation_episodes": family.evaluation_episodes,
            "network": self.network,
            "trace_episodes": family.trace_episodes, "snapshot_episodes": config.SNAPSHOT_EPISODES,
            "snapshot_fractions": list(config.SNAPSHOT_FRACTIONS) if config.SNAPSHOT_EPISODES else [],
            "snapshot_seed": 300000, "evaluation_greedy": config.EVALUATION_GREEDY,
            "reward_weight": family.reward_weight, "accelerator": self.accelerator,
            "evaluation": {"task_mean": .5, "native_shaping_return_mean": 0.0},
            "training_snapshots": [{"folder": "training_snapshots/update_1",
                                    "requested_fractions": list(config.SNAPSHOT_FRACTIONS)}]
                                    if config.SNAPSHOT_EPISODES else [],
            **self.result_overrides,
        }

    async def spawn(self, *command, **kwargs):
        args = dict(zip(command[2::2], command[3::2]))
        self.assertEqual(args["--timesteps"], str(config.SMAX.timesteps))
        self.assertEqual(args["--seed"], "0")
        self.assertEqual(args["--mode"], "candidate")
        self.assertTrue(Path(args["--reward"]).is_relative_to(self.tool.root))
        self.assertTrue(Path(args["--output"]).is_relative_to(self.tool.root))
        self.assertEqual(kwargs["env"]["JAX_PLATFORMS"], config.TRAINING_JAX_PLATFORM)
        folder = Path(args["--output"])
        result = self.result(command)
        (folder / "training_curve.csv").write_text(
            f"timesteps,policy_loss,value_loss,entropy\n{result['actual_timesteps']},0.1,0.2,0.3\n")
        (folder / "evaluation_episodes.csv").write_text("task_return,native_shaping_return_mean\n" +
            "".join(f"{index % 2},0\n" for index in range(config.SMAX.evaluation_episodes)))
        (folder / "trajectories.jsonl").write_text("{}\n")
        (folder / "checkpoint").mkdir()
        result["checkpoint_hashes"] = {}
        for network in ("actor", "critic"):
            checkpoint = folder / "checkpoint" / f"{network}.msgpack"
            checkpoint.write_bytes(network.encode())
            result["checkpoint_hashes"][network] = file_hash(checkpoint)
        for snapshot in result["training_snapshots"]:
            saved = folder / snapshot["folder"]
            (saved / "checkpoint").mkdir(parents=True)
            for name in ("metadata.json", "trajectories.jsonl", "checkpoint/actor.msgpack", "checkpoint/critic.msgpack"):
                (saved / name).write_bytes(b"{}")
        write_json(folder / "result.json", result)
        (folder / "raw.csv").write_text(f"path\n{self.run.folder}\n")
        if self.after_spawn:
            self.after_spawn(command)
        return self.worker

    async def call(self, scenario=None, path="output/reward.py"):
        return json.loads(await self.tool.run(path, scenario or config.SMAX.development[0], "Test my independently chosen hypothesis"))

    def ledger(self):
        return read_json(self.tool.ledger_path)

    async def test_success_preserves_canonical_artifacts_and_exposes_redacted_copy(self):
        before = protected_manifest(self.workspace)
        result = await self.call()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["sparse_return"], .5)
        self.assertEqual(self.tool.usage()["charged_jobs"], 1)
        copy = self.workspace / result["artifacts"]
        self.assertEqual(file_hash(copy / "reward.py"), file_hash(self.reward))
        self.assertNotIn(str(self.run.folder), (copy / "training/raw.csv").read_text())
        self.assertIn(str(self.run.folder), (self.tool.root / "probe_001/training/raw.csv").read_text())
        self.assertEqual(before, protected_manifest(self.workspace))
        self.assertEqual(read_json(self.run.state_file)["phase"], "reward_design")
        self.assertIsNone(read_json(self.run.state_file)["current_job"])
        events = [json.loads(line) for line in self.run.events_file.read_text().splitlines()]
        self.assertEqual([row["kind"] for row in events], ["llm_learning_probe_call", "llm_learning_probe_started", "llm_learning_probe_completed"])
        self.assertEqual(self.ledger()["attempts"][0]["artifact_hashes"], probes._hashes(self.tool.root / "probe_001"))

    async def test_rejects_held_out_other_family_bad_paths_and_empty_hypothesis_without_charge(self):
        for scenario in (*config.SMAX.held_out, "cramped_room", "not_registered"):
            with self.subTest(scenario=scenario):
                self.assertEqual((await self.call(scenario))["status"], "rejected")
        for path in (str(self.reward), "../reward.py", "C:\\outside\\reward.py", "output/../reward.py", "output/report.md", "missing.py"):
            with self.subTest(path=path):
                self.assertEqual((await self.call(path=path))["status"], "rejected")
        result = json.loads(await self.tool.run("output/reward.py", config.SMAX.development[0], ""))
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(self.tool.usage()["charged_jobs"], 0)
        self.spawn_mock.assert_not_called()

    async def test_symlink_cannot_escape_workspace(self):
        outside = self.root / "outside.py"
        outside.write_text("# outside")
        link = self.workspace / "output/link.py"
        try:
            link.symlink_to(outside)
        except OSError:
            self.skipTest("Host does not permit creating symlinks")
        self.assertEqual((await self.call(path="output/link.py"))["status"], "rejected")
        self.assertEqual(self.tool.usage()["charged_jobs"], 0)

    async def test_validation_failure_is_not_a_spent_training_job(self):
        self.validate.side_effect = TypeError("wrong reward interface")
        result = await self.call()
        self.assertEqual(result["status"], "rejected")
        self.assertIn("validation failed", result["error"])
        self.assertEqual(self.tool.usage()["charged_jobs"], 0)

    async def test_duplicates_reuse_complete_result_and_budget_persists_on_reconstruction(self):
        first = await self.call()
        second = await self.call()
        self.assertTrue(second["reused"])
        self.assertEqual(first["reward_hash"], second["reward_hash"])
        self.assertEqual(self.spawn_mock.call_count, 1)
        self.assertEqual(self.tool.usage()["tool_calls"], 2)
        reloaded = probes.LearningProbes(self.run, config.SMAX, "Gen1a", self.workspace)
        self.assertEqual(reloaded.usage()["charged_jobs"], 1)
        with self.assertRaisesRegex(RuntimeError, "settings changed"):
            probes.LearningProbes(self.run, config.SMAX, "Gen1a", self.workspace, max_jobs=3)
        await self.call(config.SMAX.development[1])
        rejected = await self.call(config.SMAX.development[2])
        self.assertEqual(rejected["status"], "rejected")
        self.assertIn("exhausted", rejected["error"])
        self.assertEqual(self.spawn_mock.call_count, 2)

    async def test_worker_failure_spends_slot_and_duplicate_never_retrains(self):
        self.worker = FakeProcess(returncode=1)
        result = await self.call()
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.tool.usage()["charged_jobs"], 1)
        repeated = await self.call()
        self.assertEqual(repeated["status"], "failed")
        self.assertTrue(repeated["reused"])
        self.assertEqual(self.spawn_mock.call_count, 1)

    async def test_timeout_reaps_worker_and_spends_slot(self):
        self.worker = FakeProcess(wait_error=asyncio.TimeoutError())
        with patch.object(self.tool, "_stop", new_callable=AsyncMock) as stop:
            result = await self.call()
        stop.assert_awaited_once_with(self.worker)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "Training timed out")
        self.assertEqual(self.tool.usage()["charged_jobs"], 1)

    async def test_cancel_reaps_worker_records_attempt_and_propagates(self):
        self.worker = FakeProcess(wait_error=asyncio.CancelledError())
        with patch.object(self.tool, "_stop", new_callable=AsyncMock) as stop:
            with self.assertRaises(asyncio.CancelledError):
                await self.call()
        stop.assert_awaited_once_with(self.worker)
        self.assertEqual(self.ledger()["attempts"][0]["status"], "cancelled")
        self.assertEqual(self.tool.usage()["charged_jobs"], 1)
        self.assertFalse((self.tool.root / ".active.lock").exists())

    async def test_cancel_during_creation_still_obtains_and_stops_the_worker(self):
        creating, release = asyncio.Event(), asyncio.Event()
        async def slow_spawn(*command, **kwargs):
            creating.set()
            await release.wait()
            return await self.spawn(*command, **kwargs)
        self.spawn_mock.side_effect = slow_spawn
        with patch.object(self.tool, "_stop", new_callable=AsyncMock) as stop:
            task = asyncio.create_task(self.call())
            await creating.wait()
            task.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        stop.assert_awaited_once_with(self.worker)
        self.assertEqual(self.ledger()["attempts"][0]["status"], "cancelled")

    @unittest.skipIf(probes.os.name == "nt", "POSIX process-group behavior")
    async def test_stop_signals_process_group_and_reaps_worker(self):
        self.worker = FakeProcess(returncode=None)
        with patch.object(probes.os, "killpg") as kill:
            await self.tool._stop(self.worker)
        kill.assert_called_once_with(self.worker.pid, probes.signal.SIGTERM)
        self.assertEqual(self.worker.wait_count, 1)

    @unittest.skipIf(probes.os.name == "nt", "POSIX process-group behavior")
    async def test_stop_escalates_and_reaps_when_termination_times_out(self):
        self.worker = FakeProcess(returncode=None, wait_error=asyncio.TimeoutError())
        with patch.object(probes.os, "killpg") as kill:
            await self.tool._stop(self.worker)
        self.assertEqual([call.args for call in kill.call_args_list], [
            (self.worker.pid, probes.signal.SIGTERM), (self.worker.pid, probes.signal.SIGKILL)])
        self.assertEqual(self.worker.wait_count, 2)

    @unittest.skipIf(probes.os.name == "nt", "POSIX process-group behavior")
    async def test_real_sleeping_subprocess_is_stopped_and_reaped(self):
        self.spawn_patch.stop()
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(60)", start_new_session=True,
        )
        try:
            await self.tool._stop(process)
            self.assertIsNotNone(process.returncode)
            self.assertLess(process.returncode, 0)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    async def test_rejects_result_mismatch_native_leak_and_wrong_step_count(self):
        for override in ({"seed": 99}, {"actual_timesteps": 10}, {"evaluation": {"task_mean": .5, "native_shaping_return_mean": 1}}, {"evaluation": {"task_mean": .5}}):
            with self.subTest(override=override):
                self.result_overrides = override
                self.reward.write_text(self.reward.read_text() + "# next probe\n")
                # Independent candidates have independent budgets, not reset ledgers.
                self.tool = probes.LearningProbes(self.run, config.SMAX, f"Other{len(self.reward.read_text())}", self.workspace)
                result = await self.call()
                self.assertEqual(result["status"], "failed")
                self.assertEqual(self.tool.usage()["charged_jobs"], 1)

    async def test_protected_evidence_changes_before_or_during_job_are_rejected(self):
        protected = self.workspace / "evidence" / "added.json"
        protected.write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "Protected"):
            await self.call()
        protected.unlink()
        self.after_spawn = lambda command: protected.write_text("{}")
        result = await self.call()
        self.assertEqual(result["status"], "failed")
        self.assertIn("Protected", result["error"])

    async def test_canonical_artifact_tampering_cannot_be_reused(self):
        await self.call()
        (self.tool.root / "probe_001/reward.py").write_text("# changed")
        with self.assertRaisesRegex(RuntimeError, "Canonical"):
            await self.call()
        self.assertEqual(self.spawn_mock.call_count, 1)

    async def test_frozen_run_and_existing_lock_disallow_jobs(self):
        write_json(self.run.state_file, {"reward_locked": True})
        with self.assertRaisesRegex(RuntimeError, "closed"):
            await self.call()
        write_json(self.run.state_file, {"reward_locked": False})
        (self.tool.root / ".active.lock").write_text("existing")
        self.assertEqual((await self.call())["status"], "rejected")
        self.assertEqual(self.tool.usage()["charged_jobs"], 0)

    async def test_control_responses_only_include_prescribed_diagnostic(self):
        for mode in ("fixed", "eureka"):
            with self.subTest(mode=mode):
                self.tool = probes.LearningProbes(self.run, config.SMAX, mode, self.workspace,
                                                  evidence_mode=mode)
                with patch("common.feedback.diagnostic", return_value={"fixed": True}) as report:
                    result = await self.call()
                self.assertEqual(result["status"], "complete")
                self.assertEqual(result["diagnostic"], {"fixed": True})
                self.assertNotIn("artifacts", result)
                self.assertFalse((self.workspace / "scratch/learning_probes").exists())
                report.assert_called_once_with(self.tool.root / "probe_001/training", components=mode == "eureka")

    async def test_control_failure_does_not_expose_log_paths(self):
        self.tool = probes.LearningProbes(self.run, config.SMAX, "Control", self.workspace, evidence_mode="fixed")
        self.worker = FakeProcess(returncode=1)
        result = await self.call()
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("artifacts", result)
        self.assertNotIn("worker.log", result["error"])
        self.assertIn("consumed", result["error"])

    async def test_eureka_probe_requires_component_feedback_without_spending_a_slot(self):
        self.validate.return_value = {"valid": True, "has_reward_components": False}
        self.tool = probes.LearningProbes(self.run, config.SMAX, "Control", self.workspace, evidence_mode="eureka")
        result = await self.call()
        self.assertEqual(result["status"], "rejected")
        self.assertIn("reward_components", result["error"])
        self.assertEqual(self.tool.usage()["charged_jobs"], 0)
        self.spawn_mock.assert_not_called()

    async def test_rejects_changed_code_or_missing_raw_artifacts(self):
        with patch.object(probes, "_source_signature", return_value={"source_hash": "changed"}):
            with self.assertRaisesRegex(RuntimeError, "Scientific"):
                await self.call()
        self.after_spawn = lambda command: (Path(dict(zip(command[2::2], command[3::2]))["--output"]) / "evaluation_episodes.csv").unlink()
        self.assertEqual((await self.call())["status"], "failed")

    @unittest.skipIf(probes.os.name == "nt", "POSIX process ownership recovery")
    async def test_dead_lock_keeps_interrupted_attempt_charged(self):
        await self.call()
        ledger = self.ledger()
        ledger["attempts"][0]["status"] = "running"
        write_json(self.tool.ledger_path, ledger)
        write_json(self.tool.root / ".active.lock", {"controller_pid": 12345})
        with patch.object(probes.os, "kill", side_effect=ProcessLookupError):
            result = await self.call()
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(self.tool.usage()["charged_jobs"], 1)
        self.assertEqual(self.spawn_mock.call_count, 1)

    async def test_exact_probe_supplies_development_assessment_without_new_training(self):
        await self.call()
        destination = self.run.folder / "family/development/Gen1a" / config.SMAX.development[0] / "seed_0"
        self.assertTrue(probes.reuse_learning_probe(self.run, config.SMAX, "Gen1a", config.SMAX.development[0], 0, self.reward, destination))
        marker = read_json(destination / "probe_reuse.json")
        self.assertFalse(marker["independent_confirmation"])
        self.assertEqual(marker["additional_training_jobs"], 0)
        self.assertEqual(marker["reward_hash"], file_hash(self.reward))
        self.assertEqual(file_hash(destination / "result.json"), file_hash(self.tool.root / "probe_001/training/result.json"))
        self.assertTrue(probes.reuse_learning_probe(self.run, config.SMAX, "Gen1a", config.SMAX.development[0], 0, self.reward, destination))
        self.assertEqual(self.spawn_mock.call_count, 1)
        (destination / "training_curve.csv").write_text("tampered")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            probes.reuse_learning_probe(self.run, config.SMAX, "Gen1a", config.SMAX.development[0], 0, self.reward, destination)

    async def test_reuse_never_crosses_reward_seed_candidate_or_held_out(self):
        await self.call()
        for candidate, scenario, seed in (("Other", config.SMAX.development[0], 0),
                                         ("Gen1a", config.SMAX.development[1], 0),
                                         ("Gen1a", config.SMAX.development[0], 1)):
            target = self.run.folder / "family/development" / candidate / scenario / f"seed_{seed}"
            self.assertFalse(probes.reuse_learning_probe(self.run, config.SMAX, candidate, scenario, seed, self.reward, target))
            self.assertFalse(target.exists())
        self.reward.write_text("# changed final reward\n")
        target = self.run.folder / "family/development/Gen1a" / config.SMAX.development[0] / "seed_0"
        self.assertFalse(probes.reuse_learning_probe(self.run, config.SMAX, "Gen1a", config.SMAX.development[0], 0, self.reward, target))
        with self.assertRaisesRegex(ValueError, "development"):
            probes.reuse_learning_probe(self.run, config.SMAX, "Gen1a", config.SMAX.held_out[0], 0, self.reward,
                                        self.run.folder / "family/held_out/Gen1a" / config.SMAX.held_out[0] / "seed_0")

    async def test_reuse_rejects_tampered_canonical_artifacts_and_changed_source(self):
        await self.call()
        target = self.run.folder / "family/development/Gen1a" / config.SMAX.development[0] / "seed_0"
        with patch.object(probes, "_source_signature", return_value={"source_hash": "changed"}):
            with self.assertRaisesRegex(RuntimeError, "different registered"):
                probes.reuse_learning_probe(self.run, config.SMAX, "Gen1a", config.SMAX.development[0], 0, self.reward, target)
        (self.tool.root / "probe_001/training/checkpoint/actor.msgpack").write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "Canonical"):
            probes.reuse_learning_probe(self.run, config.SMAX, "Gen1a", config.SMAX.development[0], 0, self.reward, target)


if __name__ == "__main__":
    unittest.main()
