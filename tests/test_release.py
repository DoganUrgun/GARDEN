"""Public launcher safety and internal worker boundaries, without paid work."""

import builtins
from contextlib import redirect_stdout
from io import StringIO
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import config
import main as launcher
from common import hardware, records, suite, worker
from common.records import Run, write_json


ROOT = Path(__file__).resolve().parents[1]
OFFLINE_ENTRY = """
import runpy, socket, sys
def blocked(*args, **kwargs):
    raise AssertionError('Offline command attempted network access')
socket.create_connection = blocked
socket.socket.connect = blocked
socket.socket.connect_ex = blocked
for name in ('jax', 'agents', 'openai', 'monitor', 'common.study'):
    sys.modules[name] = None
sys.path.insert(0, sys.argv[1])
sys.argv = sys.argv[2:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""


class ReleaseTests(unittest.TestCase):
    def entry(self, *arguments, expected_code=0):
        """Run the real launcher with network and learning libraries blocked."""
        environment = os.environ.copy()
        environment.update(
            OPENAI_API_KEY="", REQUIRE_GPU="1", REWARD_DESIGN_EXPERIMENT="exp1",
            PYTHONDONTWRITEBYTECODE="1", NUM_TRAIN_SEEDS="1", NUM_HELD_OUT_SEEDS="10",
        )
        before = {path.name for path in ROOT.iterdir()}
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-B", "-c", OFFLINE_ENTRY, str(ROOT),
                 str(ROOT / "main.py"), *arguments],
                cwd=directory, env=environment, capture_output=True, text=True, timeout=20,
            )
            self.assertEqual(result.returncode, expected_code, result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual({path.name for path in ROOT.iterdir()}, before)
        return result.stdout + result.stderr

    def test_no_arguments_and_help_do_not_start_work(self):
        for arguments in ((), ("--help",)):
            with self.subTest(arguments=arguments):
                output = self.entry(*arguments)
                self.assertIn("usage:", output)
                self.assertIn("--new", output)
                self.assertIn("--resume", output)
                self.assertIn("plan", output)
                self.assertNotIn("monitor", output.lower())

    def test_plan_explains_both_comparisons_without_importing_learning_stack(self):
        output = self.entry("plan")
        for label in ("GARDEN", "SMAX", "Overcooked", "Eureka", "results"):
            self.assertIn(label, output)
        self.assertIn("fixed", output.lower())
        self.assertIn("seed", output.lower())
        self.assertIn("246", output)

    def test_seed_defaults_and_example_agree(self):
        """Check real defaults without local shell or private .env overrides."""
        from dotenv import dotenv_values

        environment = os.environ.copy()
        environment.pop("NUM_TRAIN_SEEDS", None)
        environment.pop("NUM_HELD_OUT_SEEDS", None)
        with patch.dict(os.environ, environment, clear=True), patch("dotenv.load_dotenv"):
            defaults = runpy.run_path(str(ROOT / "config.py"))
        self.assertEqual(defaults["TRAIN_SEEDS"], (0,))
        self.assertEqual(defaults["HELD_OUT_SEEDS"], tuple(range(10)))
        example = dotenv_values(ROOT / ".env.example")
        self.assertEqual(example["NUM_TRAIN_SEEDS"], "1")
        self.assertEqual(example["NUM_HELD_OUT_SEEDS"], "10")

    def test_held_out_seed_count_remains_configurable(self):
        with patch.dict(os.environ, {"NUM_HELD_OUT_SEEDS": "2"}), patch("dotenv.load_dotenv"):
            settings = runpy.run_path(str(ROOT / "config.py"))
        self.assertEqual(settings["HELD_OUT_SEEDS"], (0, 1))

    def test_protocol_and_plan_end_at_held_out_evaluation(self):
        """The registered workflow contains no additional post-hoc stage."""
        from dotenv import dotenv_values

        self.assertFalse(hasattr(config, "RUN_AUDIT"))
        self.assertNotIn("RUN_POSTHOC_AUDIT", dotenv_values(ROOT / ".env.example"))
        self.assertFalse((ROOT / "common" / "audit.py").exists())
        for experiment in ("exp1", "exp2", "exp3"):
            for family in config.FAMILIES:
                with self.subTest(experiment=experiment, family=family.name), \
                     patch.object(config, "EXPERIMENT", experiment):
                    protocol = config.protocol(family)
                    self.assertNotIn("audit", protocol)
                    self.assertNotIn("post_hoc_audit", suite.family_plan(family))
                    self.assertEqual(protocol["execution"]["controller_validation_backend"], "cpu")
                    self.assertNotIn("controller_validation_and_audit_backend", protocol["execution"])

    def test_status_needs_no_model_key_gpu_or_network(self):
        output = self.entry("status")
        self.assertTrue(output.strip())

    def test_read_only_commands_cannot_be_combined_with_start_or_resume(self):
        for arguments in (
            ("plan", "--new"),
            ("status", "--resume", "latest"),
            ("doctor", "--new"),
            ("--new", "--resume", "latest"),
        ):
            with self.subTest(arguments=arguments):
                output = self.entry(*arguments, expected_code=2)
                self.assertIn("error:", output.lower())

    def test_only_one_public_experiment_launcher_is_shipped(self):
        self.assertTrue((ROOT / "main.py").is_file())
        for name in ("exp_1.py", "exp_2.py", "exp_3.py", "run_suite.py", "common/cli.py"):
            self.assertFalse((ROOT / name).exists(), name)

    def test_stage_rejects_a_different_registered_family_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_json(folder / "manifest.json", {"family": "Overcooked", "experiment": "exp1"})
            with patch.object(config, "EXPERIMENT", "exp1"), \
                 patch.object(hardware, "require_registered_accelerator") as accelerator:
                with self.assertRaisesRegex(RuntimeError, "match"):
                    worker.run_stage(config.SMAX, "search", folder)
            accelerator.assert_not_called()

    def test_stage_rejects_an_unregistered_folder_without_starting_search(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(hardware, "require_registered_accelerator") as accelerator:
                with self.assertRaises(RuntimeError):
                    worker.run_stage(config.SMAX, "search", Path(directory))
            accelerator.assert_not_called()

    def test_completed_stage_does_not_start_new_search(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_json(folder / "manifest.json", {
                "family": "SMAX", "experiment": "exp1",
                "suite": {"suite_id": "offline", "folder": str(folder / "suite"), "member": "smax_exp1"},
            })
            write_json(folder / "state.json", {"status": "complete"})
            with patch.object(config, "EXPERIMENT", "exp1"), \
                 patch.object(records, "compatible", return_value=True), \
                 patch.dict(sys.modules, {"common.study": None}), \
                 patch.dict(os.environ, {"OPENAI_API_KEY": ""}), \
                 patch.object(hardware, "require_registered_accelerator") as accelerator, \
                 redirect_stdout(StringIO()) as output:
                worker.run_stage(config.SMAX, "search", folder)
            accelerator.assert_not_called()
            self.assertIn("Already complete", output.getvalue())

    def test_incompatible_registered_stage_does_not_silently_spend(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_json(folder / "manifest.json", {
                "family": "SMAX", "experiment": "exp1",
                "suite": {"suite_id": "offline", "folder": str(folder / "suite"), "member": "smax_exp1"},
            })
            with patch.object(config, "EXPERIMENT", "exp1"), \
                 patch.object(records, "compatible", return_value=False), \
                 patch.object(hardware, "require_registered_accelerator") as accelerator:
                with self.assertRaisesRegex(RuntimeError, "Cannot resume"):
                    worker.run_stage(config.SMAX, "search", folder)
            accelerator.assert_not_called()

    def test_stage_completion_needs_no_server_or_report_export(self):
        study = ModuleType("common.study")
        study.run_study = Mock()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            run = Run(folder.name, folder, "SMAX")
            write_json(folder / "manifest.json", {
                "family": "SMAX", "experiment": "exp1",
                "suite": {"suite_id": "offline", "folder": str(folder / "suite"), "member": "smax_exp1"},
            })
            with patch.dict(sys.modules, {"common.study": study, "monitor": None}), \
                 patch.dict(os.environ, {"OPENAI_API_KEY": "offline-placeholder"}), \
                 patch.object(config, "EXPERIMENT", "exp1"), \
                 patch.object(records, "compatible", return_value=True), \
                 patch.object(hardware, "require_registered_accelerator"), \
                 patch("subprocess.Popen") as spawn, \
                 patch("socket.create_connection") as connection, \
                 patch("webbrowser.open") as browser, redirect_stdout(StringIO()) as output:
                worker.run_stage(config.SMAX, "search", folder)
            study.run_study.assert_called_once_with(run, config.SMAX, "search")
            spawn.assert_not_called()
            connection.assert_not_called()
            browser.assert_not_called()
            self.assertNotIn("export", output.getvalue().lower())
            self.assertNotIn("monitor", output.getvalue().lower())

    def test_internal_worker_selects_condition_before_loading_configuration(self):
        imported_conditions = []
        real_import = builtins.__import__

        def inspect_import(name, *args, **kwargs):
            if name == "config":
                imported_conditions.append(os.environ.get("REWARD_DESIGN_EXPERIMENT"))
            return real_import(name, *args, **kwargs)

        arguments = ["common.worker", "--experiment", "exp3", "--family", "SMAX",
                     "--phase", "search", "--run", "registered-run"]
        with patch.object(sys, "argv", arguments), \
             patch.dict(os.environ, {"REWARD_DESIGN_EXPERIMENT": "exp1"}), \
             patch.object(worker, "run_stage") as run, \
             patch("builtins.__import__", side_effect=inspect_import):
            worker.main()
        self.assertEqual(imported_conditions, ["exp3"])
        run.assert_called_once_with(config.SMAX, "search", Path("registered-run"))

    def test_new_dispatches_one_complete_suite_not_independent_conditions(self):
        folder = Path("new-registered-suite")
        with patch.object(sys, "argv", ["main.py", "--new"]), \
             patch.dict(os.environ, {"OPENAI_API_KEY": "offline-placeholder"}), \
             patch.object(hardware, "require_registered_accelerator") as accelerator, \
             patch.object(suite, "register_suite", return_value=folder) as register, \
             patch.object(suite, "run_suite") as run, redirect_stdout(StringIO()):
            launcher.main()
        accelerator.assert_called_once_with()
        register.assert_called_once_with()
        run.assert_called_once_with(folder, resume=False)

    def test_resume_dispatches_exact_suite_without_creating_another(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            write_json(folder / "suite.json", {"status": "interrupted"})
            for argument in ("latest", str(folder)):
                with self.subTest(argument=argument), \
                     patch.object(sys, "argv", ["main.py", "--resume", argument]), \
                     patch.dict(os.environ, {"OPENAI_API_KEY": "offline-placeholder"}), \
                     patch.object(launcher, "latest_suite", return_value=folder), \
                     patch.object(hardware, "require_registered_accelerator"), \
                     patch.object(suite, "register_suite") as register, \
                     patch.object(suite, "run_suite") as run, redirect_stdout(StringIO()):
                    launcher.main()
                register.assert_not_called()
                run.assert_called_once_with(folder, resume=True)

    def test_missing_key_stops_before_accelerator_or_registration(self):
        with patch.object(sys, "argv", ["main.py", "--new"]), \
             patch.dict(os.environ, {"OPENAI_API_KEY": ""}), \
             patch.object(hardware, "require_registered_accelerator") as accelerator, \
             patch.object(suite, "register_suite") as register, \
             patch.object(suite, "run_suite") as run:
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                launcher.main()
        accelerator.assert_not_called()
        register.assert_not_called()
        run.assert_not_called()

    def test_latest_suite_uses_record_time_without_successful_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "results" / "suites" / "z_old"
            latest = root / "results" / "suites" / "a_new"
            write_json(old / "suite.json", {"created_at": "2026-01-01", "status": "complete"})
            write_json(latest / "suite.json", {"created_at": "2026-01-02", "status": "failed"})
            with patch.object(config, "ROOT", root):
                self.assertEqual(launcher.latest_suite(), latest)


if __name__ == "__main__":
    unittest.main()
