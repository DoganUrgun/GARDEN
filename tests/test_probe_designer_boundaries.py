"""Probe-enabled designer boundaries and dialogue recovery, entirely offline."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import bootstrap  # noqa: F401
import config
from common import designer, feedback, structured_designer as control, workspace
from common.records import Run, file_hash, read_json, write_json


class ProbeDesignerBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Run("boundaries", Path(self.temporary.name), "SMAX")
        write_json(self.run.state_file, {"reward_locked": False})

    def test_workspace_never_publishes_current_sibling_private_artifacts(self):
        for candidate, secret in (("Gen1a", "EARLIER_EVIDENCE"), ("Gen2a", "SIBLING_SECRET")):
            for location in (f"family/candidates/{candidate}/report.md",
                             f"family/development/{candidate}/training.txt",
                             f"workspaces/{candidate}/scratch/analysis.txt",
                             f"learning_probes/{candidate}/probe_001/training/result.json",
                             f"llm_tool_outputs/{candidate}/0001.txt"):
                path = self.run.folder / location
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(secret, encoding="utf-8")

        def reference(path, family):
            path.mkdir(parents=True)
            (path / "allowed.py").write_text("PUBLIC_SOURCE", encoding="utf-8")

        with patch.object(workspace, "_copy_reference", side_effect=reference):
            work = workspace.prepare_workspace(self.run, config.SMAX, "Gen2b", 2, "b")
        files = [path for path in work.rglob("*") if path.is_file()]
        text = "\n".join(path.read_text(encoding="utf-8") for path in files)
        self.assertIn("EARLIER_EVIDENCE", text)
        self.assertNotIn("SIBLING_SECRET", text)
        self.assertFalse((work / "evidence/same_generation").exists())
        self.assertTrue((work / "evidence/candidates/Gen1a/learning_probes").exists())

    def test_sessions_cannot_share_private_sibling_conversation(self):
        a = designer._session_id(self.run, config.SMAX, "Gen1a")
        b = designer._session_id(self.run, config.SMAX, "Gen1b")
        self.assertNotEqual(a, b)
        self.assertEqual(a, designer._session_id(self.run, config.SMAX, "Gen1a"))

    def test_duplicate_rejection_does_not_disclose_hidden_sibling(self):
        with self.assertRaises(ValueError) as caught:
            designer._reject_duplicate({"reward_hash": "same", "normalized_program_hash": "program"},
                                       [{"candidate": "SECRET_SIBLING", "reward_hash": "same"}])
        self.assertNotIn("SECRET_SIBLING", str(caught.exception))
        self.assertIn("duplicates", str(caught.exception))

    def test_fixed_history_is_projected_not_raw(self):
        root = self.run.folder / "learning_probes/Gen1a"
        archive = root / "probe_001"
        archive.mkdir(parents=True)
        (archive / "reward.py").write_text("LEGITIMATE_DRAFT", encoding="utf-8")
        (archive / "private.txt").write_text("RAW_PRIVATE_TRAJECTORY", encoding="utf-8")
        write_json(root / "ledger.json", {"attempts": [{
            "id": "probe_001", "status": "complete", "scenario": config.SMAX.development[0],
            "seed": 0, "hypothesis": "Agent question", "reward_hash": file_hash(archive / "reward.py"),
            "artifact_hashes": {"reward.py": file_hash(archive / "reward.py")},
        }]})
        with patch.object(feedback, "diagnostic", return_value={"sparse_task_return": 0.5}) as diagnostic:
            result = feedback.learning_probe_history(self.run, "Gen1a")
        diagnostic.assert_called_once_with(archive / "training", components=False)
        self.assertIn("LEGITIMATE_DRAFT", json.dumps(result))
        self.assertNotIn("RAW_PRIVATE_TRAJECTORY", json.dumps(result))
        self.assertNotIn(str(self.run.folder), json.dumps(result))
        (archive / "reward.py").write_text("TAMPERED", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "history changed"):
            feedback.learning_probe_history(self.run, "Gen1a")

    def test_control_probe_then_finalize_and_resume_do_not_repeat_probe(self):
        report = "\n\n".join(f"{heading}\nObserved only fixed feedback." for heading in designer.REPORT_HEADINGS)
        probe = control.Proposal(action="probe", reward_source="DRAFT_SOURCE", report_markdown=report,
                                 reasoning_summary="Test my own question.", scenario=config.SMAX.development[0],
                                 hypothesis="Agent-selected question")
        final = control.Proposal(reward_source="FINAL_SOURCE", report_markdown=report,
                                 reasoning_summary="Updated using fixed report only.")
        probes = SimpleNamespace(workspace=self.run.folder / "control_drafts/Gen1a/probe_workspace",
                                 instructions=lambda: "Two registered probes.",
                                 run=AsyncMock(return_value='{"status":"complete","diagnostic":{"sparse_task_return":0.5}}'),
                                 usage=lambda: {"charged_jobs": 1})
        validation = lambda path, family: {"reward_hash": file_hash(path), "normalized_program_hash": "example", "has_reward_components": True}
        with patch.object(config, "EXPERIMENT", "exp2"), patch.object(config, "LEARNING_PROBE_JOBS", 2), \
             patch.object(control, "control_context", return_value="PERMITTED SOURCE"), \
             patch("common.learning_probes.LearningProbes", return_value=probes), \
             patch.object(control, "_propose", new=AsyncMock(side_effect=[probe, final])) as model, \
             patch.object(control, "validate", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                control.design_control(self.run, config.SMAX, "Gen1a", 1, "a")
        probes.run.assert_awaited_once_with("output/request_1.py", config.SMAX.development[0], "Agent-selected question")
        self.assertEqual(model.await_count, 2)
        self.assertIn('"sparse_task_return": 0.5', model.call_args_list[1].args[3])
        with patch.object(config, "EXPERIMENT", "exp2"), patch.object(config, "LEARNING_PROBE_JOBS", 2), \
             patch.object(control, "control_context", return_value="PERMITTED SOURCE"), \
             patch("common.learning_probes.LearningProbes", return_value=probes), \
             patch.object(control, "_propose", new=AsyncMock()) as resumed_model, \
             patch.object(control, "validate", side_effect=validation):
            result = control.design_control(self.run, config.SMAX, "Gen1a", 1, "a")
        resumed_model.assert_not_called()
        self.assertEqual(probes.run.await_count, 1)
        self.assertEqual(read_json(result / "design.json")["learning_probes"]["charged_jobs"], 1)

    def test_main_autonomous_candidate_receives_registered_probe_budget(self):
        work = self.run.folder / "workspaces/Gen1a"
        (work / "output").mkdir(parents=True)
        (work / "output/reward.py").write_text("FINAL_SOURCE", encoding="utf-8")
        (work / "output/report.md").write_text("\n".join(designer.REPORT_HEADINGS), encoding="utf-8")
        probes = SimpleNamespace(usage=lambda: {"max_jobs": 2, "charged_jobs": 0})
        receipt = {"summary": "Completed", "reward_hash": file_hash(work / "output/reward.py"),
                   "report_hash": file_hash(work / "output/report.md")}
        write_json(self.run.folder / "llm_submissions/Gen1a.json", receipt)
        with patch.object(config, "EXPERIMENT", "exp1"), patch.object(config, "LEARNING_PROBE_JOBS", 2), \
             patch.object(designer, "prepare_workspace", return_value=work), \
             patch.object(designer, "validate", return_value={"reward_hash": receipt["reward_hash"], "normalized_program_hash": "program"}), \
             patch("common.learning_probes.LearningProbes", return_value=probes) as constructor:
            result = designer.design_candidate(self.run, config.SMAX, "Gen1a", 1, "a")
        self.assertEqual(constructor.call_args.kwargs, {
            "max_jobs": 2, "seed": config.TRAIN_SEEDS[0],
            "timeout_seconds": config.LEARNING_PROBE_TIMEOUT_SECONDS, "evidence_mode": "autonomous",
        })
        self.assertEqual(read_json(result / "design.json")["learning_probes"]["max_jobs"], 2)

    def test_probe_deadline_preserves_reserve_and_resume_cannot_launch_more_probes(self):
        report = "\n".join(designer.REPORT_HEADINGS)
        probe = control.Proposal(action="probe", reward_source="DRAFT_SOURCE", report_markdown=report,
                                 reasoning_summary="Question", scenario=config.SMAX.development[0], hypothesis="Question")
        final = control.Proposal(reward_source="FINAL_SOURCE", report_markdown=report, reasoning_summary="Finished")
        probes = SimpleNamespace(workspace=self.run.folder / "control_drafts/Gen1a/probe_workspace",
                                 instructions=lambda: "Two registered probes.",
                                 run=AsyncMock(side_effect=asyncio.TimeoutError),
                                 usage=lambda: {"charged_jobs": 1})
        clock = SimpleNamespace(monotonic=Mock(side_effect=(0, 1, 10, 909, 1000, 1001, 2000, 2001, 3000, 3001)))
        validation = lambda path, family: {"reward_hash": file_hash(path), "normalized_program_hash": "example", "has_reward_components": True}
        with patch.object(config, "EXPERIMENT", "exp2"), patch.object(config, "LEARNING_PROBE_JOBS", 2), \
             patch.object(config, "AGENT_TIMEOUT_SECONDS", 1000), patch.object(control, "time", clock), \
             patch.object(control, "control_context", return_value="PERMITTED SOURCE"), \
             patch("common.learning_probes.LearningProbes", return_value=probes), \
             patch.object(control, "validate", side_effect=validation):
            with patch.object(control, "_propose", new=AsyncMock(side_effect=[probe, KeyboardInterrupt])) as first:
                with self.assertRaises(KeyboardInterrupt):
                    control.design_control(self.run, config.SMAX, "Gen1a", 1, "a")
            self.assertEqual([call.kwargs["timeout_seconds"] for call in first.call_args_list], [900, 100])
            draft = self.run.folder / "control_drafts/Gen1a"
            request_hash = file_hash(draft / "request_2.json")
            self.assertEqual(read_json(draft / "probe_feedback_1.json")["status"], "exploration_deadline")
            # A model asking for another probe in the reserve gets a rejection,
            # not another worker. Resume reuses the same saved request.
            with patch.object(control, "_propose", new=AsyncMock(side_effect=[probe, final])) as resumed:
                target = control.design_control(self.run, config.SMAX, "Gen1a", 1, "a")
            self.assertEqual([call.kwargs["timeout_seconds"] for call in resumed.call_args_list], [99, 98])
        self.assertEqual(probes.run.await_count, 1)
        self.assertEqual(file_hash(draft / "request_2.json"), request_hash)
        self.assertEqual(read_json(draft / "probe_feedback_2.json")["status"], "rejected")
        self.assertTrue(read_json(target / "design.json")["design_budget"]["finalization_only"])
        self.assertEqual(read_json(target / "design.json")["learning_probes"]["charged_jobs"], 1)

    def test_model_exploration_deadline_after_probe_uses_finalization_reserve(self):
        report = "\n".join(designer.REPORT_HEADINGS)
        probe = control.Proposal(action="probe", reward_source="DRAFT_SOURCE", report_markdown=report,
                                 reasoning_summary="Question", scenario=config.SMAX.development[0], hypothesis="Question")
        final = control.Proposal(reward_source="FINAL_SOURCE", report_markdown=report, reasoning_summary="Finished")
        probes = SimpleNamespace(workspace=self.run.folder / "control_drafts/Gen1a/probe_workspace",
                                 instructions=lambda: "Two registered probes.",
                                 run=AsyncMock(return_value='{"status":"complete","diagnostic":{"sparse_task_return":0.5}}'),
                                 usage=lambda: {"charged_jobs": 1})
        clock = SimpleNamespace(monotonic=Mock(side_effect=(0, 1, 2, 3, 4, 902, 1000, 1001)))
        validation = lambda path, family: {"reward_hash": file_hash(path), "normalized_program_hash": "example", "has_reward_components": True}
        with patch.object(config, "EXPERIMENT", "exp2"), patch.object(config, "LEARNING_PROBE_JOBS", 2), \
             patch.object(config, "AGENT_TIMEOUT_SECONDS", 1000), patch.object(control, "time", clock), \
             patch.object(control, "control_context", return_value="PERMITTED SOURCE"), \
             patch("common.learning_probes.LearningProbes", return_value=probes), \
             patch.object(control, "validate", side_effect=validation), \
             patch.object(control, "_propose", new=AsyncMock(side_effect=[probe, asyncio.TimeoutError, final])) as model:
            target = control.design_control(self.run, config.SMAX, "Gen1a", 1, "a")
        self.assertEqual([call.kwargs["timeout_seconds"] for call in model.call_args_list], [900, 898, 100])
        self.assertEqual(probes.run.await_count, 1)
        self.assertTrue(read_json(target / "design.json")["design_budget"]["finalization_only"])
        self.assertTrue((self.run.folder / "control_drafts/Gen1a/deadline_2.json").is_file())


if __name__ == "__main__":
    unittest.main()
