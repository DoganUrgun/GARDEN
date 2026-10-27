"""Evidence visibility and structural recovery without any model/API calls."""

from __future__ import annotations

import asyncio
import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import bootstrap  # noqa: F401
import config
from common import designer, feedback, structured_designer, study, workspace
from common.designer import REPORT_HEADINGS
from common.records import Run, file_hash, read_json, write_json
from common.reward import normalized_program_hash


class DesignerTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(config, "LEARNING_PROBE_JOBS", 0))
        # SDK clients validate credentials before the mocked request runner.
        self.enterContext(patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test-placeholder"}))
        self.temp = tempfile.TemporaryDirectory()
        self.run = Run("designer_test", Path(self.temp.name), "SMAX")
        write_json(self.run.state_file, {"reward_locked": False})

    def tearDown(self):
        self.temp.cleanup()

    def _context_mocks(self):
        return (
            patch.object(feedback, "reference_packet", return_value="PERMITTED SOURCE"),
            patch.object(feedback, "design_text", side_effect=lambda run, candidate: f"SOURCE[{candidate}]"),
            patch.object(feedback, "measured_report", side_effect=lambda run, family, candidate, **kwargs: {"MEASURED": candidate}),
        )

    def test_fixed_b_cannot_see_any_sibling_design_or_results(self):
        packet, design, measured = self._context_mocks()
        with patch.object(config, "EXPERIMENT", "exp2"), packet, design, measured as report:
            text = feedback.control_context(self.run, config.SMAX, 2, "b")
        self.assertNotIn("SOURCE[Gen2a]", text)
        self.assertIn("SOURCE[Gen1a]", text)
        self.assertIn("SOURCE[Gen1b]", text)
        self.assertEqual([call.args[2] for call in report.call_args_list], ["native", "Gen1a", "Gen1b"])
        self.assertNotIn('"MEASURED": "Gen2a"', text)
        self.assertNotIn("Gen2b", text)
        for scenario in config.SMAX.held_out:
            self.assertNotIn(scenario, text)

    def test_fixed_a_has_no_same_generation_sibling(self):
        packet, design, measured = self._context_mocks()
        with patch.object(config, "EXPERIMENT", "exp2"), packet, design, measured:
            text = feedback.control_context(self.run, config.SMAX, 2, "a")
        self.assertNotIn("SOURCE[Gen2a]", text)
        self.assertNotIn("SOURCE[Gen2b]", text)

    def test_eureka_same_generation_contexts_identical_best_previous_only(self):
        family_root = self.run.folder / "family"
        baseline = {name: 0.2 for name in config.SMAX.development}
        write_json(family_root / "baseline_summary.json", {"scenario_sparse_returns": baseline})
        for candidate, value in (("Gen2a", 0.3), ("Gen2b", 0.7)):
            write_json(family_root / "candidates" / candidate / "summary.json", {"scenario_sparse_returns": {name: value for name in baseline}})
        packet, design, measured = self._context_mocks()
        with patch.object(config, "EXPERIMENT", "exp3"), packet, design, measured as report:
            a = feedback.control_context(self.run, config.SMAX, 3, "a")
            b = feedback.control_context(self.run, config.SMAX, 3, "b")
        self.assertEqual(a, b)
        self.assertIn("SOURCE[Gen2b]", a)
        for excluded in ("Gen1a", "Gen1b", "Gen2a", "Gen3a", "Gen3b"):
            self.assertNotIn(f"SOURCE[{excluded}]", a)
        self.assertEqual([call.args[2] for call in report.call_args_list], ["Gen2b", "Gen2b"])
        self.assertTrue(all(call.kwargs["components"] for call in report.call_args_list))

    def test_eureka_initial_contexts_identical_without_measured_feedback(self):
        packet, design, measured = self._context_mocks()
        with patch.object(config, "EXPERIMENT", "exp3"), packet, design, measured as report:
            self.assertEqual(feedback.control_context(self.run, config.SMAX, 1, "a"), feedback.control_context(self.run, config.SMAX, 1, "b"))
        report.assert_not_called()

    def test_real_fixed_packet_excludes_held_out_identifiers_and_secrets(self):
        packet = feedback.reference_packet(self.run, config.SMAX)
        for identifier in config.SMAX.held_out:
            self.assertNotIn(identifier, packet)
        self.assertIn("reward_and_memory", packet)
        self.assertTrue((self.run.folder / "fixed_reference" / "manifest.json").is_file())
        self.assertFalse((self.run.folder / "fixed_reference" / "source" / ".env").exists())
        self.assertEqual(packet, feedback.reference_packet(self.run, config.SMAX))
        # The copied runtime exposes only the declared development set.
        source = self.run.folder / "fixed_reference/source/runtime/jaxmarl/environments/smax/smax_env.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        registry = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "MAP_NAME_TO_SCENARIO"
                                for target in node.targets))
        self.assertEqual({key.value for key in registry.keys}, set(config.SMAX.development))

    def test_incomplete_packet_setup_rebuilds_before_any_model_request(self):
        folder = self.run.folder / "fixed_reference"
        (folder / "source").mkdir(parents=True)
        (folder / "packet.md").write_text("Interrupted old content", encoding="utf-8")

        def copy(reference, family):
            reference.mkdir(parents=True)
            (reference / "example.py").write_text("VALID_PUBLIC_SOURCE", encoding="utf-8")

        with patch.object(feedback, "SOURCE_FILES", ("example.py",)), patch.object(feedback, "_copy_reference", side_effect=copy) as copier:
            packet = feedback.reference_packet(self.run, config.SMAX)
        copier.assert_called_once()
        self.assertIn("VALID_PUBLIC_SOURCE", packet)
        self.assertNotIn("Interrupted old content", packet)
        self.assertTrue((folder / "manifest.json").is_file())

    def test_registered_packet_tampering_is_not_silently_rebuilt(self):
        folder = self.run.folder / "fixed_reference"
        write_json(folder / "manifest.json", {"packet_hash": "registered_hash"})
        (folder / "packet.md").write_text("tampered", encoding="utf-8")
        with patch.object(feedback, "_copy_reference") as copier:
            with self.assertRaisesRegex(RuntimeError, "source packet changed"):
                feedback.reference_packet(self.run, config.SMAX)
        copier.assert_not_called()

    def test_missing_packet_registration_after_request_is_not_reconstructed(self):
        write_json(self.run.folder / "control_drafts" / "Gen1a" / "request_1.json", {})
        with patch.object(feedback, "_copy_reference") as copier:
            with self.assertRaisesRegex(RuntimeError, "after model requests"):
                feedback.reference_packet(self.run, config.SMAX)
        copier.assert_not_called()

    def test_episode_length_aggregation_weights_completed_episodes(self):
        rows = [
            {"completed_episodes": "1", "completed_episode_length_mean": "10"},
            {"completed_episodes": "9", "completed_episode_length_mean": "100"},
            {"completed_episodes": "0", "completed_episode_length_mean": "999"},
            {"completed_episodes": "0", "completed_episode_length_mean": "NaN"},
        ]
        self.assertEqual(feedback.episode_average(rows, "completed_episode_length_mean"), 91.0)
        self.assertIsNone(feedback.episode_average(rows[2:], "completed_episode_length_mean"))

    def test_reward_statistics_pool_second_moments_and_extrema(self):
        rows = [
            {"raw_component_damage_mean": "1", "raw_component_damage_std": "0",
             "raw_component_damage_min": "1", "raw_component_damage_max": "1"},
            {"raw_component_damage_mean": "3", "raw_component_damage_std": "0",
             "raw_component_damage_min": "3", "raw_component_damage_max": "3"},
        ]
        self.assertEqual(feedback.shaping_summary(rows, "raw_component_damage"), {
            "mean": 2.0, "std": 1.0, "min": 1.0, "max": 3.0,
        })

    def test_fixed_reports_include_same_mechanical_fitness_fields(self):
        normalized = {scenario: 0.2 for scenario in config.SMAX.development}
        write_json(self.run.folder / "family" / "candidates" / "Gen1a" / "summary.json", {
            "aggregate_score": 0.2, "scenario_normalized_improvements": normalized,
        })
        with patch.object(feedback, "diagnostic", return_value={"sparse_task_return": 0.7}):
            candidate = feedback.measured_report(self.run, config.SMAX, "Gen1a", components=True)
            native = feedback.measured_report(self.run, config.SMAX, "native")
        self.assertEqual(set(candidate), set(native))
        self.assertEqual(candidate["aggregate_score"], 0.2)
        self.assertEqual(candidate["scenario_normalized_improvements"], normalized)
        self.assertEqual(native["aggregate_score"], 0.0)
        self.assertTrue(all(value == 0.0 for value in native["scenario_normalized_improvements"].values()))

    def test_autonomous_workspace_visibility(self):
        measured = workspace._visibility(config.SMAX, "Gen3b", 3, "b")
        self.assertEqual(measured["visible_measured_candidates"], ["Gen1a", "Gen1b", "Gen2a", "Gen2b"])
        self.assertEqual(measured["visible_same_generation_without_results"], [])
        self.assertTrue(measured["same_generation_private"])
        self.assertFalse(measured["held_out_visible"])

    def _proposal(self):
        return structured_designer.Proposal(
            reward_source="def initial_memory(n):\n    return 0\n",
            report_markdown="\n\n".join(f"{heading}\nTest evidence." for heading in REPORT_HEADINGS),
            reasoning_summary="Mocked response; no model was called.",
        )

    def _validation(self, path, family):
        return {"reward_hash": file_hash(path), "normalized_program_hash": normalized_program_hash(path.read_text(encoding="utf-8")), "has_reward_components": True}

    def test_structural_repair_records_requests_and_commits_only_valid_program(self):
        count = 0

        def validate(path, family):
            nonlocal count
            count += 1
            if count == 1:
                raise ValueError("mocked structural error")
            return self._validation(path, family)

        with patch.object(config, "EXPERIMENT", "exp2"), patch.object(config, "MAX_DESIGN_REPAIRS", 1), patch.object(structured_designer, "control_context", return_value="PUBLIC EVIDENCE"), patch.object(structured_designer, "_propose", new=AsyncMock(return_value=self._proposal())) as propose, patch.object(structured_designer, "validate", side_effect=validate):
            target = structured_designer.design_control(self.run, config.SMAX, "Gen1a", 1, "a")
        self.assertEqual(propose.await_count, 2)
        requests = sorted((self.run.folder / "control_drafts" / "Gen1a").glob("request_*.json"))
        self.assertEqual(len(requests), 2)
        self.assertEqual(read_json(requests[0])["task"].split("\n")[0], "PUBLIC EVIDENCE")
        self.assertIn("mocked structural error", read_json(requests[1])["task"])
        self.assertTrue(study._design_complete(target))
        (target / "report.md").write_text("changed report", encoding="utf-8")
        self.assertFalse(study._design_complete(target))

    def test_validation_failure_does_not_commit_candidate(self):
        with patch.object(config, "EXPERIMENT", "exp2"), patch.object(config, "MAX_DESIGN_REPAIRS", 0), patch.object(structured_designer, "control_context", return_value="PUBLIC EVIDENCE"), patch.object(structured_designer, "_propose", new=AsyncMock(return_value=self._proposal())), patch.object(structured_designer, "validate", side_effect=ValueError("invalid")):
            with self.assertRaisesRegex(RuntimeError, "did not pass validation"):
                structured_designer.design_control(self.run, config.SMAX, "Gen1a", 1, "a")
        self.assertFalse((self.run.folder / "family" / "candidates" / "Gen1a" / "design.json").exists())

    def test_saved_valid_response_recovers_without_another_model_call(self):
        with patch.object(config, "EXPERIMENT", "exp2"), patch.object(structured_designer, "control_context", return_value="PUBLIC EVIDENCE"), patch.object(structured_designer, "_propose", new=AsyncMock(return_value=self._proposal())), patch.object(structured_designer, "validate", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                structured_designer.design_control(self.run, config.SMAX, "Gen1a", 1, "a")
        draft = self.run.folder / "control_drafts" / "Gen1a"
        self.assertTrue((draft / "response_1.json").is_file())
        original_hash = file_hash(draft / "request_1.json")
        with patch.object(config, "EXPERIMENT", "exp2"), patch.object(structured_designer, "control_context", return_value="PUBLIC EVIDENCE"), patch.object(structured_designer, "_propose", new=AsyncMock()) as propose, patch.object(structured_designer, "validate", side_effect=self._validation):
            target = structured_designer.design_control(self.run, config.SMAX, "Gen1a", 1, "a")
        propose.assert_not_called()
        self.assertTrue(study._design_complete(target))
        self.assertEqual(original_hash, file_hash(draft / "request_1.json"))

    def test_changed_evidence_refuses_resume_without_overwriting_request(self):
        with patch.object(config, "EXPERIMENT", "exp2"), patch.object(structured_designer, "control_context", return_value="ORIGINAL EVIDENCE"), patch.object(structured_designer, "_propose", new=AsyncMock(side_effect=RuntimeError("transport failure"))):
            with self.assertRaisesRegex(RuntimeError, "transport failure"):
                structured_designer.design_control(self.run, config.SMAX, "Gen1a", 1, "a")
        request = self.run.folder / "control_drafts" / "Gen1a" / "request_1.json"
        original_hash = file_hash(request)
        with patch.object(config, "EXPERIMENT", "exp2"), patch.object(structured_designer, "control_context", return_value="CHANGED EVIDENCE"), patch.object(structured_designer, "_propose", new=AsyncMock()) as propose:
            with self.assertRaisesRegex(RuntimeError, "differs from the current evidence"):
                structured_designer.design_control(self.run, config.SMAX, "Gen1a", 1, "a")
        propose.assert_not_called()
        self.assertEqual(original_hash, file_hash(request))

    def test_transport_failure_resumes_same_request_and_preserves_failure_log(self):
        with patch.object(config, "EXPERIMENT", "exp2"), patch.object(structured_designer, "control_context", return_value="PUBLIC EVIDENCE"), patch.object(structured_designer, "_propose", new=AsyncMock(side_effect=RuntimeError("transport failure"))):
            with self.assertRaisesRegex(RuntimeError, "transport failure"):
                structured_designer.design_control(self.run, config.SMAX, "Gen1a", 1, "a")
        draft = self.run.folder / "control_drafts" / "Gen1a"
        request_hash = file_hash(draft / "request_1.json")
        self.assertFalse((draft / "response_1.json").exists())
        with patch.object(config, "EXPERIMENT", "exp2"), patch.object(structured_designer, "control_context", return_value="PUBLIC EVIDENCE"), patch.object(structured_designer, "_propose", new=AsyncMock(return_value=self._proposal())) as propose, patch.object(structured_designer, "validate", side_effect=self._validation):
            target = structured_designer.design_control(self.run, config.SMAX, "Gen1a", 1, "a")
        self.assertEqual(propose.await_count, 1)
        self.assertTrue(study._design_complete(target))
        self.assertEqual(request_hash, file_hash(draft / "request_1.json"))
        self.assertEqual(len(list(draft.glob("request_*.json"))), 1)
        self.assertIn('"kind":"llm_request_failed"', self.run.events_file.read_text(encoding="utf-8"))

    def test_report_only_finalizer_usage_is_counted(self):
        import agents
        work = self.run.folder / "workspaces" / "Gen1a"
        (work / "output").mkdir(parents=True)
        (work / "output" / "reward.py").write_text(self._proposal().reward_source, encoding="utf-8")
        usage = SimpleNamespace(requests=1, input_tokens=100, output_tokens=15, input_tokens_details=SimpleNamespace(cached_tokens=7))
        result = SimpleNamespace(final_output="A report", context_wrapper=SimpleNamespace(usage=usage))
        with patch.object(agents, "Agent", side_effect=lambda **kwargs: SimpleNamespace(**kwargs)), patch.object(agents, "SQLiteSession"), patch.object(agents.Runner, "run", new=AsyncMock(return_value=result)):
            asyncio.run(designer._finalize_report(self.run, work, config.SMAX, "Gen1a"))
        totals = study._llm_usage(self.run)
        self.assertEqual(totals["requests"], 1)
        self.assertEqual(totals["input_tokens"], 100)
        self.assertEqual(totals["cached_input_tokens"], 7)
        self.assertEqual(totals["output_tokens"], 15)
        requests = list((self.run.folder / "llm_requests" / "Gen1a").glob("*.json"))
        self.assertEqual(len(requests), 1)
        self.assertEqual(read_json(requests[0])["purpose"], "documentation_only")

    def _autonomous_workspace(self, report=True):
        work = self.run.folder / "workspaces" / "Gen1a"
        (work / "output").mkdir(parents=True)
        (work / "output/reward.py").write_text(self._proposal().reward_source, encoding="utf-8")
        if report:
            (work / "output/report.md").write_text(self._proposal().report_markdown, encoding="utf-8")
        write_json(work / "workspace_manifest.json", {"protected_files": {}})
        return work

    def test_shell_python_syntax_is_not_mistaken_for_paths_or_deletion(self):
        commands = (
            'python -c "print(array[..., 1]); result = ~walls; del result"',
            "python - <<'PY'\nvalue = data[..., 1]\nmask = ~walls\ndel value\nPY",
            'printf "status:\\n"',
            'python -c "from jaxmarl.environments import SMAX"',
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertIsNone(designer.WorkspaceShell._reject(command))

    def test_shell_still_rejects_out_of_scope_operations_and_paths(self):
        commands = (
            "rm output/reward.py", "echo ok; del output/reward.py", "sudo echo hello",
            "curl https://example.com", "python -m pip install package",
            "cat ../config.py", 'python -c "open(\'../private\')"',
            "cat ~/.env", 'python -c "open(\'/etc/passwd\')"',
            'Get-Content C:\\Users\\secret', 'python -c "print(os.environ)"',
            'python -c "open(\'.env\')"', "cat /home/user/private",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertIsNotNone(designer.WorkspaceShell._reject(command))

    def test_explicit_submission_commits_exact_hashes_and_resumes_without_calls(self):
        work = self._autonomous_workspace()
        with patch.object(designer, "validate", side_effect=self._validation):
            receipt = designer._submit_candidate(self.run, work, config.SMAX, "Gen1a", "Completed analysis.")
        with patch.object(config, "EXPERIMENT", "exp1"), patch.object(designer, "prepare_workspace", return_value=work), patch.object(designer, "validate", side_effect=self._validation), patch.object(designer, "_run_agent", new=AsyncMock()) as agent:
            target = designer.design_candidate(self.run, config.SMAX, "Gen1a", 1, "a")
        agent.assert_not_called()
        self.assertTrue(study._design_complete(target))
        self.assertEqual(read_json(target / "design.json")["submission"], receipt)
        self.assertEqual(file_hash(target / "report.md"), receipt["report_hash"])

    def test_agent_submission_tool_finishes_without_another_model_turn(self):
        import agents
        from agents.tool_context import ToolContext
        work = self._autonomous_workspace()

        async def complete(agent, task, **kwargs):
            self.assertEqual(agent.tool_use_behavior, {"stop_at_tool_names": ["submit_candidate"]})
            self.assertIn("100 seconds", task)
            tool = agent.tools[1]
            arguments = json.dumps({"summary": "Final artifacts checked."})
            context = ToolContext(context=None, tool_name="submit_candidate", tool_call_id="mock_submission", tool_arguments=arguments)
            result = await tool.on_invoke_tool(context, arguments)
            return SimpleNamespace(final_output=result)

        with patch.object(designer, "validate", side_effect=self._validation), patch.object(agents, "SQLiteSession"), patch.object(agents.Runner, "run", new=AsyncMock(side_effect=complete)) as runner:
            summary = asyncio.run(designer._run_agent(self.run, work, config.SMAX, "Gen1a", "Investigate freely.", timeout_seconds=100))
        self.assertEqual(summary, "Final artifacts checked.")
        self.assertEqual(runner.await_count, 1)
        self.assertEqual(designer._submission(self.run, work, "Gen1a")["mode"], "explicit_tool")

    def test_real_agent_timeout_keeps_drafts_unsubmitted(self):
        import agents
        work = self._autonomous_workspace()

        async def pending(*args, **kwargs):
            await asyncio.sleep(1)

        with patch.object(agents, "SQLiteSession"), patch.object(agents.Runner, "run", new=AsyncMock(side_effect=pending)):
            summary = asyncio.run(designer._run_agent(self.run, work, config.SMAX, "Gen1a", "Investigate.", timeout_seconds=0.01))
        self.assertEqual(summary, "")
        self.assertEqual(designer._submission(self.run, work, "Gen1a"), {})
        self.assertIn('"kind":"llm_budget_reached"', self.run.events_file.read_text(encoding="utf-8"))

    def test_external_probe_or_time_budget_overrides_are_not_supported(self):
        with patch.object(config, "EXPERIMENT", "exp1"), patch.object(designer, "prepare_workspace") as prepare:
            with self.assertRaisesRegex(TypeError, "learning_probes"):
                designer.design_candidate(self.run, config.SMAX, "Gen1a", 1, "a", learning_probes=object())
            with self.assertRaisesRegex(TypeError, "budget_seconds"):
                designer.design_candidate(self.run, config.SMAX, "Gen1a", 1, "a", budget_seconds=5400)
        prepare.assert_not_called()

    def test_learning_probe_sdk_tool_uses_registered_prompt_and_allowance(self):
        import agents
        from agents.tool_context import ToolContext
        work = self._autonomous_workspace()
        probes = SimpleNamespace(instructions=lambda: "Fixed two-job test budget.",
                                 run=AsyncMock(return_value='{"status":"complete"}'))
        instructions = (config.PROMPTS / "reward_engineer.md").read_text(encoding="utf-8")

        async def complete(agent, task, **kwargs):
            self.assertEqual(len(agent.tools), 3)
            tool = agent.tools[2]
            self.assertEqual(tool.name, "run_learning_probe")
            self.assertIn("Fixed two-job test budget", task)
            self.assertEqual(agent.instructions, instructions)
            arguments = json.dumps({"reward_file": "scratch/draft.py", "scenario": config.SMAX.development[0],
                                    "hypothesis": "Agent-chosen question"})
            context = ToolContext(context=None, tool_name=tool.name, tool_call_id="mock_probe", tool_arguments=arguments)
            result = await tool.on_invoke_tool(context, arguments)
            self.assertEqual(result, '{"status":"complete"}')
            return SimpleNamespace(final_output="Mocked probe-enabled run")

        with patch.object(agents, "SQLiteSession"), patch.object(agents.Runner, "run", new=AsyncMock(side_effect=complete)):
            result = asyncio.run(designer._run_agent(self.run, work, config.SMAX, "Gen1a", "Investigate.",
                                                     timeout_seconds=100, learning_probes=probes))
        self.assertEqual(result, "Mocked probe-enabled run")
        probes.run.assert_awaited_once_with("scratch/draft.py", config.SMAX.development[0], "Agent-chosen question")
        request = read_json(self.run.folder / "llm_requests/Gen1a/0001.json")
        self.assertEqual(request["instructions"], instructions)
        self.assertIn("Fixed two-job test budget", request["task"])

    def test_sdk_agent_without_probe_allowance_has_no_probe_tool(self):
        import agents
        work = self._autonomous_workspace()

        async def complete(agent, task, **kwargs):
            self.assertEqual(len(agent.tools), 2)
            self.assertEqual([tool.name for tool in agent.tools[1:]], ["submit_candidate"])
            return SimpleNamespace(final_output="Mocked ordinary run")

        with patch.object(agents, "SQLiteSession"), patch.object(agents.Runner, "run", new=AsyncMock(side_effect=complete)):
            asyncio.run(designer._run_agent(self.run, work, config.SMAX, "Gen1a", "Investigate.", timeout_seconds=100))

    def test_timeout_with_valid_drafts_does_not_commit_or_delete_them(self):
        work = self._autonomous_workspace()
        source_hash = file_hash(work / "output/reward.py")
        with patch.object(config, "EXPERIMENT", "exp1"), patch.object(config, "MAX_DESIGN_REPAIRS", 1), patch.object(designer, "prepare_workspace", return_value=work), patch.object(designer, "validate", side_effect=self._validation), patch.object(designer, "_run_agent", new=AsyncMock(return_value="")) as agent:
            with self.assertRaisesRegex(RuntimeError, "Explicit submission is missing"):
                designer.design_candidate(self.run, config.SMAX, "Gen1a", 1, "a")
        self.assertEqual(agent.await_count, 2)
        self.assertEqual(file_hash(work / "output/reward.py"), source_hash)
        self.assertFalse((self.run.folder / "family/candidates/Gen1a/design.json").exists())
        self.assertLess(agent.call_args_list[0].kwargs["timeout_seconds"], config.AGENT_TIMEOUT_SECONDS)
        self.assertIn("reserved finalization", agent.call_args_list[1].args[4])

    def test_changed_report_invalidates_a_previous_explicit_submission(self):
        work = self._autonomous_workspace()
        with patch.object(designer, "validate", side_effect=self._validation):
            designer._submit_candidate(self.run, work, config.SMAX, "Gen1a", "Ready.")
        (work / "output/report.md").write_text(self._proposal().report_markdown + "\nNew draft", encoding="utf-8")
        self.assertEqual(designer._submission(self.run, work, "Gen1a"), {})

    def test_missing_report_only_recovery_explicitly_submits_unchanged_source(self):
        work = self._autonomous_workspace(report=False)
        source_hash = file_hash(work / "output/reward.py")
        with patch.object(config, "EXPERIMENT", "exp1"), patch.object(designer, "prepare_workspace", return_value=work), patch.object(designer, "validate", side_effect=self._validation), patch.object(designer, "_run_agent", new=AsyncMock(return_value="")), patch.object(designer, "_finalize_report", new=AsyncMock(return_value=self._proposal().report_markdown)) as finalizer:
            target = designer.design_candidate(self.run, config.SMAX, "Gen1a", 1, "a")
        self.assertEqual(finalizer.await_count, 1)
        self.assertEqual(file_hash(target / "reward.py"), source_hash)
        self.assertEqual(read_json(target / "design.json")["submission"]["mode"], "documentation_only")

    def test_exhausted_budget_on_resume_cannot_start_more_model_calls(self):
        work = self._autonomous_workspace()
        write_json(self.run.folder / "llm_progress/Gen1a.json", {"elapsed_seconds": config.AGENT_TIMEOUT_SECONDS})
        with patch.object(config, "EXPERIMENT", "exp1"), patch.object(designer, "prepare_workspace", return_value=work), patch.object(designer, "validate", side_effect=self._validation), patch.object(designer, "_run_agent", new=AsyncMock()) as agent:
            with self.assertRaisesRegex(RuntimeError, "no candidate was committed"):
                designer.design_candidate(self.run, config.SMAX, "Gen1a", 1, "a")
        agent.assert_not_called()
        self.assertTrue((work / "output/reward.py").is_file())

    def test_frozen_run_refuses_any_design_call(self):
        write_json(self.run.state_file, {"reward_locked": True})
        with patch.object(structured_designer, "_propose", new=AsyncMock()) as propose:
            with self.assertRaisesRegex(RuntimeError, "disabled after freezing"):
                structured_designer.design_control(self.run, config.SMAX, "Gen3b", 3, "b")
            propose.assert_not_called()

    def test_no_tool_model_call_disables_silent_context_truncation(self):
        import agents
        proposal = self._proposal()
        with patch.object(agents, "Agent") as agent, patch.object(agents.Runner, "run", new=AsyncMock(return_value=SimpleNamespace(final_output=proposal))):
            asyncio.run(structured_designer._propose(self.run, config.SMAX, "Gen1a", "PUBLIC", "RULES"))
        self.assertEqual(agent.call_args.kwargs["tools"], [])
        self.assertEqual(agent.call_args.kwargs["model_settings"].truncation, "disabled")


if __name__ == "__main__":
    unittest.main()
