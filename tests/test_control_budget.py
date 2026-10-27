"""No-tool controls share the autonomous candidate budget, including resume."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import bootstrap  # noqa: F401
import config
from common import structured_designer as designer
from common.designer import REPORT_HEADINGS
from common.records import Run, file_hash, read_json, write_json
from common.reward import normalized_program_hash


class ControlBudgetTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(config, "LEARNING_PROBE_JOBS", 0))
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Run("control_budget", Path(self.temporary.name), "SMAX")
        write_json(self.run.state_file, {"reward_locked": False})
        self.progress = self.run.folder / "llm_progress" / "Gen1a.json"
        self.draft = self.run.folder / "control_drafts" / "Gen1a"
        self.proposal = designer.Proposal(
            reward_source="def initial_memory(n):\n    return 0\n",
            report_markdown="\n\n".join(f"{heading}\nTest evidence." for heading in REPORT_HEADINGS),
            reasoning_summary="Offline test response.",
        )
        self.context = self.enterContext(ExitStack())
        self.context.enter_context(patch.object(config, "EXPERIMENT", "exp2"))
        self.context.enter_context(patch.object(config, "AGENT_TIMEOUT_SECONDS", 1000.0))
        self.context.enter_context(patch.object(config, "MAX_DESIGN_REPAIRS", 2))
        self.context.enter_context(patch.object(designer, "control_context", return_value="PUBLIC EVIDENCE"))
        self.context.enter_context(patch.object(designer, "validate", side_effect=self.validation))

    @staticmethod
    def validation(path, family):
        return {"reward_hash": file_hash(path),
                "normalized_program_hash": normalized_program_hash(path.read_text(encoding="utf-8")),
                "has_reward_components": True}

    def design(self):
        return designer.design_control(self.run, config.SMAX, "Gen1a", 1, "a")

    def test_repairs_share_budget_and_are_limited_to_reserve(self):
        responses = iter((ValueError("first correction"), ValueError("second correction"), None))

        def validate(path, family):
            failure = next(responses)
            if failure:
                raise failure
            return self.validation(path, family)

        clock = SimpleNamespace(monotonic=Mock(side_effect=(0, 890, 1000, 1090, 2000, 2010)))
        with patch.object(designer, "time", clock), patch.object(designer, "validate", side_effect=validate), patch.object(designer, "_propose", new=AsyncMock(return_value=self.proposal)) as propose:
            target = self.design()
        self.assertEqual([call.kwargs["timeout_seconds"] for call in propose.call_args_list], [900, 100, 20])
        record = read_json(target / "design.json")
        self.assertEqual(record["design_budget"]["elapsed_seconds"], 990)
        self.assertEqual(record["design_budget"], read_json(self.progress))
        self.assertEqual(record["submission"]["reward_hash"], file_hash(target / "reward.py"))
        self.assertEqual(record["submission"]["report_hash"], file_hash(target / "report.md"))

    def test_actual_timeout_preserves_request_and_spent_time_without_submission(self):
        import agents

        async def pending(*args, **kwargs):
            await asyncio.sleep(1)

        client = AsyncMock()
        client.__aenter__.return_value = client
        with patch("openai.AsyncOpenAI", return_value=client) as client_type, patch.object(config, "AGENT_TIMEOUT_SECONDS", 0.02), patch.object(agents, "Agent", side_effect=lambda **kwargs: SimpleNamespace(**kwargs)), patch.object(agents.Runner, "run", new=AsyncMock(side_effect=pending)):
            with self.assertRaises(asyncio.TimeoutError):
                self.design()
        self.assertAlmostEqual(client_type.call_args.kwargs["timeout"], 0.018)
        self.assertEqual(client_type.call_args.kwargs["max_retries"], 0)
        client.__aexit__.assert_awaited_once()
        self.assertGreater(read_json(self.progress)["elapsed_seconds"], 0)
        self.assertTrue((self.draft / "request_1.json").exists())
        self.assertFalse((self.draft / "response_1.json").exists())
        self.assertFalse((self.run.folder / "family/candidates/Gen1a/design.json").exists())

    def test_sdk_uses_remaining_phase_timeout_without_hidden_retries(self):
        import agents

        client = AsyncMock()
        client.__aenter__.return_value = client
        with patch("openai.AsyncOpenAI", return_value=client) as client_type, patch.object(agents, "OpenAIResponsesModel") as model_type, patch.object(agents, "Agent", side_effect=lambda **kwargs: SimpleNamespace(**kwargs)), patch.object(agents.Runner, "run", new=AsyncMock(return_value=SimpleNamespace(final_output=self.proposal))) as runner:
            self.design()
        # This exceeds the SDK's default 600-second read timeout, but remains
        # inside the same 1000-second candidate budget (100 seconds reserved).
        client_type.assert_called_once_with(timeout=900.0, max_retries=0)
        model_type.assert_called_once_with(config.MODEL, openai_client=client)
        engineer = runner.call_args.args[0]
        self.assertIs(engineer.model, model_type.return_value)
        self.assertEqual(engineer.tools, [])
        self.assertEqual(engineer.model_settings.reasoning.effort, config.REASONING_EFFORT)
        client.__aexit__.assert_awaited_once_with(None, None, None)
        log = self.run.events_file.read_text(encoding="utf-8")
        self.assertIn('"kind":"llm_transport_configured"', log)
        self.assertIn('"sdk_max_retries":0', log)

    def test_sdk_timeout_closes_client_and_records_failure_without_submission(self):
        import agents
        import httpx
        from openai import APITimeoutError

        client = AsyncMock()
        client.__aenter__.return_value = client
        failure = APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
        with patch("openai.AsyncOpenAI", return_value=client), patch.object(agents, "Agent", side_effect=lambda **kwargs: SimpleNamespace(**kwargs)), patch.object(agents.Runner, "run", new=AsyncMock(side_effect=failure)) as runner:
            with self.assertRaises(APITimeoutError):
                self.design()
        runner.assert_awaited_once()
        client.__aexit__.assert_awaited_once()
        self.assertIs(client.__aexit__.call_args.args[1], failure)
        self.assertTrue((self.draft / "request_1.json").exists())
        self.assertFalse((self.draft / "response_1.json").exists())
        self.assertFalse((self.run.folder / "family/candidates/Gen1a/design.json").exists())
        log = self.run.events_file.read_text(encoding="utf-8")
        self.assertIn('"kind":"llm_request_failed"', log)
        self.assertIn("APITimeoutError", log)
        self.assertGreater(read_json(self.progress)["elapsed_seconds"], 0)

    def test_exhausted_resume_cannot_start_a_model_call(self):
        write_json(self.progress, {"elapsed_seconds": 1000.0})
        with patch.object(designer, "_propose", new=AsyncMock()) as propose:
            with self.assertRaisesRegex(RuntimeError, "exhausted its shared design budget"):
                self.design()
        propose.assert_not_called()

    def test_saved_response_revalidates_even_after_budget_exhaustion(self):
        with patch.object(designer, "_propose", new=AsyncMock(return_value=self.proposal)), patch.object(designer, "validate", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.design()
        request_hash = file_hash(self.draft / "request_1.json")
        write_json(self.progress, {"elapsed_seconds": 1000.0})
        clock = SimpleNamespace(monotonic=Mock())
        with patch.object(designer, "time", clock), patch.object(designer, "_propose", new=AsyncMock()) as propose:
            target = self.design()
        propose.assert_not_called()
        clock.monotonic.assert_not_called()
        self.assertEqual(read_json(target / "design.json")["design_budget"]["elapsed_seconds"], 1000)
        self.assertEqual(file_hash(self.draft / "request_1.json"), request_hash)

    def test_transport_retry_preserves_request_and_spent_budget(self):
        clock = SimpleNamespace(monotonic=Mock(side_effect=(0, 50, 100, 120)))
        with patch.object(designer, "time", clock):
            with patch.object(designer, "_propose", new=AsyncMock(side_effect=RuntimeError("transport failure"))):
                with self.assertRaisesRegex(RuntimeError, "transport failure"):
                    self.design()
            request_hash = file_hash(self.draft / "request_1.json")
            with patch.object(designer, "_propose", new=AsyncMock(return_value=self.proposal)) as propose:
                self.design()
        self.assertEqual(propose.call_args.kwargs["timeout_seconds"], 850)
        self.assertEqual(read_json(self.progress)["elapsed_seconds"], 70)
        self.assertEqual(file_hash(self.draft / "request_1.json"), request_hash)
        self.assertIn('"kind":"llm_request_failed"', self.run.events_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
