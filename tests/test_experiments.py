"""Offline protocol tests: no API calls, policy training, or GPU probing."""

from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import bootstrap  # noqa: F401
import config
from common import comparison, records, study
from common.suite import family_plan
from common.records import Run, file_hash, read_json, write_json


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = Run("test", self.root / "test", "SMAX")
        self.run.folder.mkdir()
        write_json(self.run.state_file, {"status": "running"})

    def tearDown(self):
        self.temp.cleanup()

    def test_default_job_counts(self):
        with patch.object(config, "TRAIN_SEEDS", (0,)), patch.object(config, "HELD_OUT_SEEDS", tuple(range(10))):
            for experiment, expected in (("exp1", 61), ("exp2", 38), ("exp3", 38)):
                with patch.object(config, "EXPERIMENT", experiment):
                    self.assertEqual(family_plan(config.SMAX)["total_new_training_jobs"], expected)

    def test_smax_split_and_five_million_budget_are_shared(self):
        self.assertEqual(config.SMAX.development, ("3s5z_vs_3s6z", "smacv2_5_units", "5m_vs_6m"))
        self.assertEqual(config.SMAX.held_out, ("10m_vs_11m", "smacv2_10_units"))
        self.assertFalse(set(config.SMAX.development) & set(config.SMAX.held_out))
        # Exercise the registered default independently of local overrides.
        family = replace(config.SMAX, timesteps=5_000_000)
        protocols = []
        for experiment in ("exp1", "exp2", "exp3"):
            with self.subTest(experiment=experiment), patch.object(config, "EXPERIMENT", experiment):
                current = family_plan(family)
                self.assertEqual(current["actual_steps_per_job"], 4_997_120)
                self.assertEqual(current["development_scenarios"], list(family.development))
                self.assertEqual(current["held_out_scenarios"], list(family.held_out))
                protocols.append(config.protocol(family)["family"])
        self.assertEqual(protocols, [family.private_dict()] * 3)

    def test_different_smax_budget_or_split_cannot_supply_control_reference(self):
        family = replace(config.SMAX, timesteps=5_000_000)
        old_protocols = (
            replace(family, timesteps=9_994_240),
            replace(family, development=("2s3z", "smacv2_5_units", "5m_vs_6m")),
            replace(family, held_out=("3s5z", "smacv2_10_units")),
        )
        write_json(self.run.state_file, {"status": "complete"})
        with patch.object(records, "dependency_versions", return_value={}), patch.object(records, "scientific_source_manifest", return_value={}):
            current = records.comparison_signature(family)
            for old_family in old_protocols:
                previous = records.comparison_signature(old_family)
                self.assertNotEqual(previous, current)
                write_json(self.run.folder / "manifest.json", {"experiment": "exp1", "comparison_signature": previous})
                with patch.object(comparison, "latest_run", return_value=self.run):
                    for experiment in ("exp2", "exp3"):
                        with self.subTest(experiment=experiment, old_family=old_family), patch.object(config, "EXPERIMENT", experiment):
                            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                                comparison.reference_run(family)

    def test_comparison_matches_modes_but_rejects_changed_evidence_settings(self):
        with patch.object(records, "dependency_versions", return_value={}), patch.object(records, "scientific_source_manifest", return_value={}):
            signatures = []
            for experiment in ("exp1", "exp2", "exp3"):
                with patch.object(config, "EXPERIMENT", experiment):
                    signatures.append(records.comparison_signature(config.SMAX))
            self.assertEqual(signatures[0], signatures[1])
            self.assertEqual(signatures[0], signatures[2])
            with patch.object(config, "SNAPSHOT_EPISODES", config.SNAPSHOT_EPISODES + 1):
                self.assertNotEqual(signatures[0], records.comparison_signature(config.SMAX))

    def test_job_reuse_requires_matching_snapshot_configuration(self):
        family = config.SMAX
        accelerator = {"backend": "gpu", "devices": ["cuda:0"], "process_count": 1}
        result = {
            "family": family.name, "scenario": family.development[0], "seed": 0,
            "mode": "native", "reward_hash": None, "reward_weight": None,
            "requested_timesteps": family.timesteps, "learner": family.learner.to_dict(),
            "actual_timesteps": (family.timesteps // (family.learner.num_envs * family.learner.rollout_steps)) * (family.learner.num_envs * family.learner.rollout_steps),
            "evaluation_episodes": family.evaluation_episodes,
            "trace_episodes": family.trace_episodes, "evaluation_greedy": config.EVALUATION_GREEDY,
            "snapshot_episodes": config.SNAPSHOT_EPISODES,
            "snapshot_fractions": list(config.SNAPSHOT_FRACTIONS) if config.SNAPSHOT_EPISODES else [],
            "snapshot_seed": 300_000, "accelerator": accelerator,
        }
        def matches(value):
            return study._job_matches(value, family, family.development[0], 0, "native", None, accelerator)
        self.assertTrue(matches(result))
        for key, wrong in (("snapshot_episodes", 99), ("snapshot_fractions", [0.5]), ("snapshot_seed", 1)):
            self.assertFalse(matches({**result, key: wrong}))
        self.assertFalse(matches({**result, "requested_timesteps": family.timesteps + 16_384}))
        self.assertFalse(matches({**result, "actual_timesteps": 16_384}))

    def test_latest_is_creation_record_not_pointer_or_completed_fallback(self):
        with patch.object(config, "ROOT", self.root), patch.object(config, "SEARCH_REPLICATE", 0):
            folder = records.run_root(config.SMAX, "exp1")
            for name, date, status in (("z_old", "2026-01-01", "complete"), ("a_new", "2026-01-02", "failed")):
                write_json(folder / name / "manifest.json", {"family": "SMAX", "created_at": date})
                write_json(folder / name / "state.json", {"status": status})
            (folder / "latest.txt").write_text("z_old", encoding="utf-8")
            latest = records.latest_run(config.SMAX, "exp1")
            self.assertEqual(latest.id, "a_new")
            with self.assertRaisesRegex(RuntimeError, "not complete"):
                comparison.reference_run(config.SMAX)

    def test_manifest_ignores_results_and_virtual_environment(self):
        with patch.object(config, "ROOT", self.root):
            (self.root / "training.py").write_text("pass\n", encoding="utf-8")
            before = records.source_manifest()
            write_json(self.root / "results" / "exp1" / "result.json", {"task_mean": 1})
            write_json(self.root / ".venv" / "package.json", {"version": 1})
            self.assertEqual(before, records.source_manifest())

    def test_prompts_are_flat_and_registered_by_content_hash(self):
        self.assertEqual(config.PROMPTS, config.ROOT / "prompts")
        names = {"reward_engineer.md", "fixed.md", "eureka.md"}
        self.assertEqual({path.name for path in config.PROMPTS.iterdir()}, names)
        sources = records.source_manifest()
        for name in names:
            self.assertEqual(sources[f"prompts/{name}"], file_hash(config.PROMPTS / name))

    def test_registration_archives_exact_source_without_secrets_or_results(self):
        with patch.object(config, "ROOT", self.root), patch.object(records, "probe_accelerator", return_value={"backend": "cpu", "devices": ["cpu:0"], "process_count": 1}):
            (self.root / "training.py").write_text("pass\n", encoding="utf-8")
            (self.root / "PROTOCOL.md").write_text("Public protocol\n", encoding="utf-8")
            (self.root / ".env").write_text("OPENAI_API_KEY=must-not-be-copied\n", encoding="utf-8")
            write_json(self.root / "prompts" / "schema.json", {"name": "test-prompt"})
            write_json(self.root / "results" / "old" / "result.json", {"task_mean": 99})
            run = records.create_run(config.SMAX)
        sources = read_json(run.folder / "manifest.json")["source_manifest"]
        archive = run.folder / "registered" / "source"
        copied = {path.relative_to(archive).as_posix(): file_hash(path) for path in archive.rglob("*") if path.is_file()}
        self.assertEqual(copied, sources)
        self.assertIn("training.py", copied)
        self.assertIn("PROTOCOL.md", copied)
        self.assertIn("prompts/schema.json", copied)
        self.assertFalse((archive / ".env").exists())
        self.assertFalse((archive / "results").exists())

    def test_usage_summary_reports_truncated_event_without_failing(self):
        self.run.events_file.write_text(
            '{"kind":"llm_reasoning_completed","usage":{"requests":1,"input_tokens":25,"output_tokens":9}}\n'
            '{"kind":"llm_reason', encoding="utf-8")
        usage = study._llm_usage(self.run)
        self.assertEqual(usage["requests"], 1)
        self.assertEqual(usage["input_tokens"], 25)
        self.assertEqual(usage["unreadable_event_lines"], 1)

    def test_both_designs_precede_generation_training(self):
        timeline = []
        frozen = self.run.folder / "family" / "frozen_reward.py"

        def design(run, family, candidate, generation, position):
            timeline.append(("design", candidate))

        def train(run, family, candidate, baseline):
            timeline.append(("train", candidate))

        def freeze(run, family, baseline):
            frozen.parent.mkdir(exist_ok=True)
            frozen.write_text("pass\n", encoding="utf-8")
            write_json(frozen.parent / "selection.json", {"selected_candidate": "Gen3a", "selected_reward_hash": file_hash(frozen)})
            return "Gen3a", frozen

        with patch.object(config, "EXPERIMENT", "exp2"), patch.object(study, "_run_development_baseline", return_value={}), patch.object(study, "_design_complete", return_value=False), patch.object(study, "design_candidate", side_effect=design), patch.object(study, "_train_candidate", side_effect=train), patch.object(study, "_select_and_lock", side_effect=freeze), patch.object(study, "_held_out", return_value={"status": "complete"}), patch.object(study, "_write_development_table"):
            study.run_family(self.run, config.SMAX)
        expected = []
        for a, b in config.GENERATIONS:
            expected.extend((("design", a), ("design", b), ("train", a), ("train", b)))
        self.assertEqual(timeline, expected)

    def test_held_out_uses_ten_fresh_seeds_and_progressive_table(self):
        frozen = self.run.folder / "family" / "frozen_reward.py"
        frozen.parent.mkdir()
        frozen.write_text("pass\n", encoding="utf-8")
        digest = file_hash(frozen)
        write_json(frozen.parent / "selection.json", {"selected_reward_hash": digest})
        calls = []

        def train(run, family, phase, method, scenario, seed, mode, reward):
            calls.append((method, scenario, seed))
            result = {"evaluation": {"task_mean": float(seed)}}
            write_json(study._job_folder(run, family, phase, method, scenario, seed) / "result.json", result)
            return result

        with patch.object(config, "EXPERIMENT", "exp1"), patch.object(config, "TRAIN_SEEDS", (0,)), patch.object(config, "HELD_OUT_SEEDS", tuple(range(10))), patch.object(study, "_run_training_job", side_effect=train):
            table = study._held_out(self.run, config.SMAX, "Gen3a", frozen)
        self.assertEqual(len(calls), 40)
        self.assertEqual(set(calls), {
            (method, scenario, seed)
            for method in ("native", "Gen3a")
            for scenario in config.SMAX.held_out
            for seed in range(10)
        })
        self.assertEqual(table["evaluation_seeds"], list(range(10)))
        self.assertEqual(table["selected_sparse_returns"][config.SMAX.held_out[0]], 4.5)
        self.assertEqual(len(table["per_seed"]["Gen3a"][config.SMAX.held_out[0]]), 10)

    def test_frozen_resume_never_calls_designer(self):
        family_root = self.run.folder / "family"
        frozen = family_root / "frozen_reward.py"
        family_root.mkdir()
        frozen.write_text("pass\n", encoding="utf-8")
        write_json(family_root / "baseline_summary.json", {"scenario_sparse_returns": {}})
        write_json(family_root / "selection.json", {"selected_candidate": "Gen3b", "selected_reward_hash": file_hash(frozen)})
        write_json(family_root / "held_out_table.json", {"status": "running"})
        with patch.object(config, "EXPERIMENT", "exp3"), patch.object(study, "design_candidate") as design, patch.object(study, "_run_development_baseline") as baseline, patch.object(study, "_held_out", return_value={"status": "complete"}) as held:
            study.run_family(self.run, config.SMAX)
            design.assert_not_called()
            baseline.assert_not_called()
            held.assert_called_once()
        self.assertTrue(read_json(self.run.state_file)["reward_locked"])

    def test_frozen_garden_finishes_after_held_out_without_counterfactual_stage(self):
        """Final evaluation completes the study without redesign or extra analysis."""
        family_root = self.run.folder / "family"
        family_root.mkdir()
        frozen = family_root / "frozen_reward.py"
        frozen.write_text("pass\n", encoding="utf-8")
        write_json(family_root / "baseline_summary.json", {"scenario_sparse_returns": {}})
        selection = {"selected_candidate": "Gen3a", "selected_reward_hash": file_hash(frozen)}
        write_json(family_root / "selection.json", selection)
        held_out = {"status": "complete", "selected_sparse_returns": {"held_out": 0.75}}

        def evaluate(*args):
            write_json(family_root / "held_out_table.json", held_out)
            return held_out

        with patch.object(config, "EXPERIMENT", "exp1"), \
             patch.dict(sys.modules, {"common.audit": None}), \
             patch.object(study, "design_candidate") as design, \
             patch.object(study, "_run_development_baseline") as baseline, \
             patch.object(study, "_held_out", side_effect=evaluate) as held:
            study.run_study(self.run, config.SMAX, "evaluate")
            # A completed study is also safe to resume without repeating evaluation.
            study.run_study(self.run, config.SMAX, "evaluate")
        design.assert_not_called()
        baseline.assert_not_called()
        held.assert_called_once_with(self.run, config.SMAX, "Gen3a", frozen)
        summary = read_json(family_root / "family_summary.json")
        self.assertEqual(summary["held_out"], held_out)
        self.assertEqual(summary["selection"], selection)
        self.assertEqual(summary["status"], "complete")
        self.assertNotIn("audit", summary)
        self.assertFalse((family_root / "audit").exists())
        self.assertEqual(read_json(self.run.state_file)["status"], "complete")
        self.assertTrue(read_json(self.run.state_file)["reward_locked"])

    def test_no_reference_heldout_before_lock(self):
        with self.assertRaisesRegex(RuntimeError, "before reward freeze"):
            comparison.reference_held_out(self.run, config.SMAX)

    def test_search_only_stops_before_held_out(self):
        family_root = self.run.folder / "family"
        family_root.mkdir()
        frozen = family_root / "frozen_reward.py"
        frozen.write_text("pass\n", encoding="utf-8")
        write_json(family_root / "baseline_summary.json", {"scenario_sparse_returns": {}})
        write_json(family_root / "selection.json", {"selected_candidate": "Gen1a", "selected_reward_hash": file_hash(frozen)})
        with patch.object(study, "design_candidate") as design, patch.object(study, "_held_out") as held:
            study.run_study(self.run, config.SMAX, "search")
        self.assertEqual(read_json(self.run.state_file)["status"], "search_complete")
        self.assertFalse(read_json(self.run.state_file).get("held_out_revealed", False))
        design.assert_not_called()
        held.assert_not_called()

    def test_evaluation_only_never_starts_unfinished_search(self):
        with patch.object(study, "design_candidate") as design, patch.object(study, "_run_development_baseline") as baseline:
            with self.assertRaisesRegex(RuntimeError, "already selected"):
                study.run_study(self.run, config.SMAX, "evaluate")
        design.assert_not_called()
        baseline.assert_not_called()

    def test_frozen_reference_available_before_held_out_but_not_for_final_table(self):
        write_json(self.run.state_file, {"status": "search_complete", "reward_locked": True})
        write_json(self.run.folder / "manifest.json", {"experiment": "exp1", "comparison_signature": {"same": True}})
        frozen = self.run.folder / "family" / "frozen_reward.py"
        frozen.parent.mkdir()
        frozen.write_text("pass\n", encoding="utf-8")
        write_json(frozen.parent / "selection.json", {"selected_reward_hash": file_hash(frozen)})
        with patch.object(comparison, "latest_run", return_value=self.run), patch.object(comparison, "comparison_signature", return_value={"same": True}):
            self.assertEqual(comparison.reference_run(config.SMAX), self.run)
            with self.assertRaisesRegex(RuntimeError, "not complete"):
                comparison.reference_run(config.SMAX, require_held_out=True)

    def test_probe_reuse_counts_once_not_as_second_training(self):
        output = self.run.folder / "family" / "development" / "Gen1a" / config.SMAX.development[0] / "seed_0"
        write_json(output / "result.json", {})
        write_json(output / "probe_reuse.json", {"probe_id": "probe_001"})
        write_json(self.run.folder / "learning_probes" / "Gen1a" / "ledger.json", {"attempts": [{"status": "complete"}, {"status": "failed"}]})
        with patch.object(config, "EXPERIMENT", "exp1"):
            counts = study._training_counts(self.run)
        self.assertEqual(counts["development"], 0)
        self.assertEqual(counts["development_assessments_reused_from_probes"], 1)
        self.assertEqual(counts["charged_learning_probes"], 2)
        self.assertEqual(counts["total_new_trainings"], 2)

    def test_reference_requires_matching_scientific_provenance(self):
        write_json(self.run.state_file, {"status": "complete"})
        write_json(self.run.folder / "manifest.json", {"experiment": "exp1", "comparison_signature": {"model": "old"}})
        with patch.object(comparison, "latest_run", return_value=self.run), patch.object(comparison, "comparison_signature", return_value={"model": "new"}):
            with self.assertRaisesRegex(RuntimeError, "incompatible"):
                comparison.reference_run(config.SMAX)

    def test_matching_reference_is_read_only_and_requires_exact_frozen_hash(self):
        write_json(self.run.state_file, {"status": "complete"})
        write_json(self.run.folder / "manifest.json", {"experiment": "exp1", "comparison_signature": {"model": "same"}})
        frozen = self.run.folder / "family" / "frozen_reward.py"
        frozen.parent.mkdir()
        frozen.write_text("pass\n", encoding="utf-8")
        write_json(frozen.parent / "selection.json", {"selected_reward_hash": file_hash(frozen)})
        write_json(frozen.parent / "held_out_table.json", {"status": "complete"})
        before = {p.relative_to(self.run.folder).as_posix(): file_hash(p) for p in self.run.folder.rglob("*") if p.is_file()}
        with patch.object(comparison, "latest_run", return_value=self.run), patch.object(comparison, "comparison_signature", return_value={"model": "same"}):
            self.assertEqual(comparison.reference_run(config.SMAX), self.run)
            after = {p.relative_to(self.run.folder).as_posix(): file_hash(p) for p in self.run.folder.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            frozen.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash changed"):
                comparison.reference_run(config.SMAX)

    def test_comparison_job_counts_exclude_copied_baselines(self):
        for method in ("native", "Gen1a"):
            write_json(self.run.folder / "family" / "development" / method / config.SMAX.development[0] / "seed_0" / "result.json", {})
        with patch.object(config, "EXPERIMENT", "exp2"):
            counts = study._training_counts(self.run)
        self.assertEqual(counts["reused_native_development"], 1)
        self.assertEqual(counts["total_new_trainings"], 1)


if __name__ == "__main__":
    unittest.main()
