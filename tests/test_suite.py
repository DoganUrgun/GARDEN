"""Offline queue and cross-condition information-boundary checks."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import config
from common import records, suite
from common.records import Run, file_hash, manifest_hash, read_json, write_json


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.folder = self.root / "results" / "suites" / "test_suite"
        self.folder.mkdir(parents=True)
        members = []
        for key, experiment, family in suite.MEMBERS:
            folder = self.root / key
            manifest = {"family": family, "experiment": experiment,
                        "suite": {"suite_id": "test_suite", "folder": str(self.folder), "member": key}}
            write_json(folder / "manifest.json", manifest)
            write_json(folder / "state.json", {"status": "registered"})
            members.append({"key": key, "experiment": experiment, "family": family,
                            "run_id": key, "run_folder": str(folder),
                            "manifest_hash": file_hash(folder / "manifest.json")})
        registration = {"members": members, "stages": suite.plan()["stages"]}
        self.suite = {"suite_id": "test_suite", "registration": registration,
                      "registration_hash": manifest_hash(registration), "status": "registered",
                      "frozen_rewards": {}, "stages_status": {stage["id"]: {"status": "pending"} for stage in registration["stages"]}}
        write_json(self.folder / "suite.json", self.suite)
        self.members = {member["key"]: member for member in members}

    def tearDown(self):
        self.temp.cleanup()

    def freeze(self, key):
        member = self.members[key]
        root = Path(member["run_folder"]) / "family"
        root.mkdir(exist_ok=True)
        (root / "frozen_reward.py").write_text(f"# {key}\n", encoding="utf-8")
        write_json(root / "selection.json", {"selected_candidate": "Gen2a", "selected_reward_hash": file_hash(root / "frozen_reward.py"), "selection_uses_held_out": False})
        self.suite["frozen_rewards"][key] = suite._freeze_record(member)
        write_json(self.folder / "suite.json", self.suite)

    def test_fixed_budget_and_order(self):
        with patch.object(config, "TRAIN_SEEDS", (0,)), patch.object(config, "HELD_OUT_SEEDS", tuple(range(10))), patch.object(config, "LEARNING_PROBE_JOBS", 2):
            plan = suite.plan()
        self.assertEqual([item["maximum_jobs"] for item in plan["members"]], [73, 50, 50, 73])
        self.assertEqual(plan["maximum_training_jobs"], 246)
        self.assertEqual([stage["phase"] for stage in plan["stages"][:3]], ["search"] * 3)
        self.assertEqual([stage["phase"] for stage in plan["stages"][3:6]], ["evaluate"] * 3)

    def test_barrier_rejects_two_winners_and_accepts_three(self):
        run = Run("smax_exp1", Path(self.members["smax_exp1"]["run_folder"]), "SMAX")
        self.freeze("smax_exp1")
        self.freeze("smax_exp2")
        with self.assertRaisesRegex(RuntimeError, "All three"):
            records.suite_held_out_barrier(run)
        self.freeze("smax_exp3")
        records.suite_held_out_barrier(run)
        (run.folder / "family" / "frozen_reward.py").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "All three"):
            records.suite_held_out_barrier(run)

    def test_barrier_rejects_registration_tamper(self):
        self.suite["registration"]["members"].pop()
        write_json(self.folder / "suite.json", self.suite)
        run = Run("smax_exp1", Path(self.members["smax_exp1"]["run_folder"]), "SMAX")
        with self.assertRaisesRegex(RuntimeError, "registration"):
            records.suite_held_out_barrier(run)

    def test_standalone_held_out_cannot_bypass_suite_barrier(self):
        from common import study
        self.freeze("smax_exp1")
        run = Run("smax_exp1", Path(self.members["smax_exp1"]["run_folder"]), "SMAX")
        with patch.object(study, "_run_training_job") as train:
            with self.assertRaisesRegex(RuntimeError, "All three"):
                study._held_out(run, config.SMAX, "Gen2a", run.folder / "family" / "frozen_reward.py")
        train.assert_not_called()

    def test_queue_stops_on_first_failed_stage_without_retry(self):
        process = Mock()
        process.pid = 1234
        process.wait.return_value = 1
        with patch.object(suite, "load_suite", return_value=self.suite), patch.object(suite.subprocess, "Popen", return_value=process) as popen:
            with self.assertRaisesRegex(RuntimeError, "No retry"):
                suite.run_suite(self.folder)
        self.assertEqual(popen.call_count, 1)
        self.assertEqual(read_json(self.folder / "suite.json")["status"], "failed")
        self.assertFalse((self.folder / "controller.lock").exists())

    def test_child_session_gets_one_cleanup_interrupt(self):
        import signal
        process = Mock(pid=1234)
        process.wait.side_effect = [KeyboardInterrupt(), 130]
        posix = SimpleNamespace(**vars(suite.os))
        posix.name = "posix"
        with patch.object(suite, "load_suite", return_value=self.suite), patch.object(suite.subprocess, "Popen", return_value=process) as popen, patch.object(suite, "os", posix):
            with self.assertRaises(KeyboardInterrupt):
                suite.run_suite(self.folder)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        process.send_signal.assert_called_once_with(signal.SIGINT)
        process.terminate.assert_not_called()
        process.kill.assert_not_called()
        self.assertEqual(read_json(self.folder / "suite.json")["status"], "interrupted")

    def test_failed_queue_requires_explicit_resume(self):
        self.suite["status"] = "failed"
        with patch.object(suite, "load_suite", return_value=self.suite), patch.object(suite.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "resume explicitly"):
                suite.run_suite(self.folder)
        popen.assert_not_called()

    def test_resume_refuses_live_orphan_stage_worker(self):
        import socket
        self.suite.update(status="running", current_stage="smax_exp1_search", controller_host=socket.gethostname())
        self.suite["stages_status"]["smax_exp1_search"] = {"status": "running", "worker_pid": 1234}
        with patch.object(suite, "load_suite", return_value=self.suite), patch.object(suite, "_pid_alive", return_value=True), patch.object(suite.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "still running"):
                suite.run_suite(self.folder, resume=True)
        popen.assert_not_called()

    def test_queue_freezes_all_searches_before_any_evaluation(self):
        timeline = []
        def start(command, **kwargs):
            self.assertEqual(command[1:4], ["-B", "-m", "common.worker"])
            arguments = dict(zip(command[4::2], command[5::2]))
            folder = Path(arguments["--run"])
            key = folder.name
            phase = arguments["--phase"]
            self.assertEqual(arguments["--experiment"], self.members[key]["experiment"])
            self.assertEqual(arguments["--family"], self.members[key]["family"])
            timeline.append((key, phase))
            root = folder / "family"
            if phase in {"search", "study"}:
                root.mkdir(exist_ok=True)
                (root / "frozen_reward.py").write_text(f"# {key}\n", encoding="utf-8")
                write_json(root / "selection.json", {"selected_candidate": "Gen1b", "selected_reward_hash": file_hash(root / "frozen_reward.py"), "selection_uses_held_out": False})
            if phase == "evaluate":
                self.assertEqual(len(read_json(self.folder / "suite.json")["frozen_rewards"]), 3)
            write_json(folder / "state.json", {"status": "search_complete" if phase == "search" else "complete"})
            process = Mock(pid=1234)
            process.wait.return_value = 0
            return process
        with patch.object(suite, "load_suite", return_value=self.suite), patch.object(suite.subprocess, "Popen", side_effect=start):
            suite.run_suite(self.folder)
        self.assertEqual(timeline, [(member, phase) for _, member, phase in suite.STAGES])
        self.assertEqual(read_json(self.folder / "suite.json")["status"], "complete")

    def test_registration_always_creates_four_fresh_bound_runs(self):
        accelerator = {"backend": "cpu", "devices": ["cpu:0"], "process_count": 1}
        with patch.object(config, "ROOT", self.root), patch.object(records, "probe_accelerator", return_value=accelerator):
            folder = suite.register_suite()
        registration = read_json(folder / "suite.json")
        members = registration["registration"]["members"]
        self.assertEqual(len(members), 4)
        self.assertEqual(len({member["run_id"] for member in members}), 4)
        for member in members:
            manifest_path = Path(member["run_folder"]) / "manifest.json"
            self.assertEqual(file_hash(manifest_path), member["manifest_hash"])
            self.assertEqual(read_json(manifest_path)["suite"]["member"], member["key"])


if __name__ == "__main__":
    unittest.main()
