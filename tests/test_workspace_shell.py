"""Offline regressions for the designer's local shell; never calls an LLM."""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import bootstrap  # noqa: F401
import config
from common import designer
from common.records import Run


def request(command: str):
    return SimpleNamespace(data=SimpleNamespace(action=SimpleNamespace(commands=[command])))


class WorkspaceShellTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.run = Run("shell_test", Path(self.temporary.name), "SMAX")
        self.work = self.run.folder / "workspaces" / "Gen1a"
        (self.work / "scratch").mkdir(parents=True)
        self.shell = designer.WorkspaceShell(self.work, self.run, "SMAX", "Gen1a")

    def tearDown(self):
        self.temporary.cleanup()

    def test_python_suffixes_and_division_are_not_absolute_file_access(self):
        examples = (
            "print(policy + '/3s5z_vs_3s6z/seed_0')",
            "print(k.endswith('/seed_0'))",
            "print(k.startswith('/seed_0'))",
            "print(6672 /11233)",
            "print(numerator /denominator)",
            "print(6672 / 11233)",
        )
        for source in examples:
            for command in (f'python -c "{source}"', f"python - <<'PY'\n{source}\nPY"):
                with self.subTest(command=command):
                    self.assertIsNone(self.shell._reject(command))

    def test_actual_absolute_paths_are_still_rejected(self):
        commands = (
            "cat /etc/passwd", "cat '/etc/passwd'", "cat /123/file",
            'python -c "open(\'/etc/passwd\')"',
            'python -c "open(\'\' + \'/etc/passwd\')"',
            "python - <<'PY'\nopen('/etc/passwd')\nPY",
            "python - <<'PY'\nprint(3 / 2)\nPY\ncat /etc/passwd",
            'Get-Content C:\\Users\\private',
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertEqual(self.shell._reject(command), "absolute paths are unavailable")

    def test_large_stdout_and_stderr_have_readable_workspace_artifact(self):
        full_stdout = "a" * 25_000 + "MIDDLE_EVIDENCE" + "z" * 40_000
        full_stderr = "FIRST_ERROR" + "e" * 25_000
        process = SimpleNamespace(
            returncode=7,
            communicate=AsyncMock(return_value=(full_stdout.encode(), full_stderr.encode())),
        )
        with patch.object(designer.asyncio, "create_subprocess_exec", new=AsyncMock(return_value=process)):
            result = asyncio.run(self.shell(request('python -c "print(1)"')))
        output = result.output[0]
        path = self.work / "scratch/tool_outputs/0001.txt"
        self.assertEqual(output.outcome.exit_code, 7)
        self.assertIn("scratch/tool_outputs/0001.txt", output.stdout)
        self.assertIn("scratch/tool_outputs/0001.txt", output.stderr)
        self.assertNotIn("MIDDLE_EVIDENCE", output.stdout)
        self.assertIn("MIDDLE_EVIDENCE", path.read_text(encoding="utf-8"))
        self.assertIn(full_stderr, path.read_text(encoding="utf-8"))
        self.assertEqual(path.read_bytes(), (self.run.folder / "llm_tool_outputs/Gen1a/0001.txt").read_bytes())
        self.assertIsNone(self.shell._reject("cat scratch/tool_outputs/0001.txt"))

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Production shell requires Linux/WSL bash")
    def test_unhandled_failure_cannot_be_hidden_by_later_success(self):
        result = asyncio.run(self.shell(request("python -c 'import sys; sys.exit(7)'\nprintf 'SHOULD_NOT_RUN'")))
        self.assertEqual(result.output[0].outcome.exit_code, 7)
        self.assertNotIn("SHOULD_NOT_RUN", result.output[0].stdout)

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Production shell requires Linux/WSL bash")
    def test_pipeline_failure_is_not_hidden_by_last_process(self):
        result = asyncio.run(self.shell(request("python -c 'import sys; sys.exit(7)' | cat")))
        self.assertEqual(result.output[0].outcome.exit_code, 7)

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Production shell requires Linux/WSL bash")
    def test_intentional_conditionals_and_recovery_are_allowed(self):
        commands = (
            "if false; then printf unexpected; else printf recovered; fi",
            "false || printf recovered",
            "false && printf unexpected\nprintf recovered",
        )
        for command in commands:
            with self.subTest(command=command):
                result = asyncio.run(self.shell(request(command)))
                self.assertEqual(result.output[0].outcome.exit_code, 0)
                self.assertEqual(result.output[0].stdout, "recovered")

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Production shell requires Linux/WSL bash")
    def test_timeout_preserves_partial_output(self):
        with patch.object(config, "SHELL_TIMEOUT_SECONDS", 0.2):
            result = asyncio.run(self.shell(request("printf 'before timeout'; sleep 5")))
        self.assertEqual(result.output[0].outcome.type, "timeout")
        self.assertIsNone(result.output[0].outcome.exit_code)
        self.assertIn("before timeout", result.output[0].stdout)


if __name__ == "__main__":
    unittest.main()
