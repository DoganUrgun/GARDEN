"""Structured-only control dialogue with budgeted, fixed-feedback learning probes."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

import config
from common.designer import REPORT_HEADINGS, _check_report, _prior_reward_pool, _reject_duplicate, _usage, _reasoning_summary
from common.feedback import control_context
from common.records import event, file_hash, now, read_json, update, write_json
from validate_reward import validate


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reward_source: str
    report_markdown: str
    reasoning_summary: str
    action: Literal["finalize", "probe"] = "finalize"
    scenario: str = ""
    hypothesis: str = ""


async def _propose(run, family, candidate, task, instruction, *, timeout_seconds=None):
    from agents import Agent, ModelSettings, OpenAIResponsesModel, RunHooks, Runner
    from openai import AsyncOpenAI
    from openai.types.shared import Reasoning

    class Hooks(RunHooks):
        async def on_llm_start(self, context, agent, system_prompt, input_items):
            event(run, "llm_reasoning_started", "Reasoning over the registered fixed evidence", family=family.name, candidate=candidate)

        async def on_llm_end(self, context, agent, response):
            event(run, "llm_reasoning_completed", "Completed a no-tool design response", family=family.name,
                  candidate=candidate, usage=_usage(response), reasoning_summary=_reasoning_summary(response) or None)

    # Each call receives only its own saved dialogue plus the permitted history.
    # There is no persistent session through which sibling results can leak.
    timeout = config.AGENT_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    # A complete no-tool proposal can reason for more than the SDK's default
    # 600-second read timeout. Use the registered remaining phase allowance,
    # without hidden SDK retries that could repeat a paid generation. The outer
    # deadline still caps the entire run, including any structured-output turns.
    event(run, "llm_transport_configured", "One request attempt per model turn; the shared design deadline remains enforced",
          family=family.name, candidate=candidate, timeout_seconds=timeout, sdk_max_retries=0)
    async with AsyncOpenAI(timeout=timeout, max_retries=0) as client:
        engineer = Agent(
            name="Prescribed reward engineer" if config.EXPERIMENT == "exp2" else "Eureka-inspired reward engineer",
            model=OpenAIResponsesModel(config.MODEL, openai_client=client),
            instructions=instruction, tools=[], output_type=Proposal,
            model_settings=ModelSettings(
                reasoning=Reasoning(mode=config.REASONING_MODE, effort=config.REASONING_EFFORT,
                                    context="all_turns", summary="detailed"),
                verbosity=config.MODEL_VERBOSITY, parallel_tool_calls=False,
                truncation="disabled", store=True,
            ),
        )
        result = await asyncio.wait_for(
            Runner.run(engineer, task, max_turns=config.MAX_AGENT_TURNS, hooks=Hooks()),
            timeout=timeout,
        )
    value = result.final_output
    return value if isinstance(value, Proposal) else Proposal.model_validate(value)


def design_control(run, family, candidate, generation, position) -> Path:
    if read_json(run.state_file, {}).get("reward_locked"):
        raise RuntimeError("Reward design is disabled after freezing")
    context = control_context(run, family, generation, position)
    instruction_name = "fixed.md" if config.EXPERIMENT == "exp2" else "eureka.md"
    instruction = (config.PROMPTS / instruction_name).read_text(encoding="utf-8")
    # Do not insert candidate identifiers or A/B fingerprints into Eureka's input:
    # both independent samples must receive exactly the same generation context.
    task = context + "\n\nReport headings:\n" + "\n".join(REPORT_HEADINGS)
    target = run.folder / "family" / "candidates" / candidate
    draft = run.folder / "control_drafts" / candidate
    draft.mkdir(parents=True, exist_ok=True)
    probes = None
    if config.LEARNING_PROBE_JOBS:
        from common.learning_probes import LearningProbes
        work = draft / "probe_workspace"
        (work / "output").mkdir(parents=True, exist_ok=True)
        probes = LearningProbes(
            run, family, candidate, work, max_jobs=config.LEARNING_PROBE_JOBS,
            seed=config.TRAIN_SEEDS[0], timeout_seconds=config.LEARNING_PROBE_TIMEOUT_SECONDS,
            evidence_mode="fixed" if config.EXPERIMENT == "exp2" else "eureka",
        )
        task += "\n\n" + probes.instructions()
    update(run, phase="reward_design", candidate=candidate, current_job=None,
           activity=f"Designing {candidate} using {'prescribed diagnostics' if config.EXPERIMENT == 'exp2' else 'best-parent reward reflection'}")
    event(run, "llm_design_started", f"Started {candidate}", family=family.name, candidate=candidate, generation=generation)
    pool = _prior_reward_pool(run, exclude=candidate)
    progress_path = run.folder / "llm_progress" / f"{candidate}.json"
    progress = read_json(progress_path, {"elapsed_seconds": 0.0})
    if progress.get("budget_seconds", config.AGENT_TIMEOUT_SECONDS) != config.AGENT_TIMEOUT_SECONDS:
        raise RuntimeError("Cannot resume with a different design-time budget")
    reserve = min(300.0, config.AGENT_TIMEOUT_SECONDS * 0.10)
    progress.update(budget_seconds=config.AGENT_TIMEOUT_SECONDS,
                    finalization_reserve_seconds=reserve)
    dialogue, corrections, finalization_only = [], 0, False
    deadline_notice = {
        "status": "exploration_deadline", "finalization_only": True,
        "error": "The investigation allowance ended. Any launched probe remains charged. Use only the existing draft and evidence: finalize the reward and report without further probes or diagnostics.",
    }

    def remaining():
        return max(0.0, config.AGENT_TIMEOUT_SECONDS - progress["elapsed_seconds"])

    def invoke(call, allowed):
        started = time.monotonic()
        try:
            return asyncio.run(asyncio.wait_for(call(), timeout=allowed))
        finally:
            progress["elapsed_seconds"] += time.monotonic() - started
            write_json(progress_path, progress)

    # Probe requests do not consume structural-repair attempts. All dialogue and
    # training still share the same persisted wall-time and turn ceilings.
    for attempt in range(config.MAX_AGENT_TURNS):
        request = task + ("\n\nYour candidate-local dialogue:\n" + json.dumps(dialogue, sort_keys=True) if dialogue else "")
        request_path = draft / f"request_{attempt + 1}.json"
        response_path = draft / f"response_{attempt + 1}.json"
        deadline_path = draft / f"deadline_{attempt + 1}.json"
        request_record = {
            "model": config.MODEL, "settings": config.protocol(family)["model"],
            "instructions": instruction, "task": request,
            "evidence_interface": config.EXPERIMENT, "generation": generation,
        }
        # Replay saved proposals before spending another model call. Request
        # records are immutable: resume cannot silently change the evidence or
        # erase an earlier repair. Only transport failures without a saved
        # response are retried on an explicit command-line resume.
        if request_path.exists():
            if read_json(request_path) != request_record:
                raise RuntimeError(f"{candidate}: saved design request differs from the current evidence")
        else:
            write_json(request_path, request_record)
        if deadline_path.is_file():
            # Replay a finished exploration phase without changing its saved
            # request or repeating its timed-out model call.
            dialogue.append({"controller": read_json(deadline_path)})
            finalization_only = True
            continue
        if response_path.exists():
            proposal = Proposal.model_validate(read_json(response_path))
            event(run, "llm_proposal_recovered", f"Revalidating saved proposal {attempt + 1} without a model call",
                  family=family.name, candidate=candidate)
        else:
            # Initial generation, repairs and transport retries share one
            # persisted active-time budget, as in the autonomous condition.
            # Revalidating an already saved response consumes no model time.
            allowed = min(reserve, remaining()) if corrections or finalization_only else max(0.0, remaining() - reserve)
            if allowed <= 0:
                if dialogue and not finalization_only and remaining() > 0:
                    write_json(deadline_path, deadline_notice)
                    dialogue.append({"controller": deadline_notice})
                    finalization_only = True
                    progress["finalization_only"] = True
                    write_json(progress_path, progress)
                    continue
                event(run, "llm_budget_reached", "No model time remains for this candidate phase",
                      family=family.name, candidate=candidate, attempt=attempt + 1)
                raise RuntimeError(f"{candidate} exhausted its shared design budget; saved proposals remain available, but no candidate was committed")
            try:
                proposal = invoke(lambda: _propose(run, family, candidate, request, instruction,
                                                   timeout_seconds=allowed), allowed)
            except Exception as error:
                # Errors are retained chronologically in events.jsonl. Do not
                # replace a failed request or silently switch the model.
                event(run, "llm_request_failed", f"{type(error).__name__}: {error}",
                      family=family.name, candidate=candidate, attempt=attempt + 1,
                      request_artifact=str(request_path.relative_to(run.folder)))
                if isinstance(error, asyncio.TimeoutError) and dialogue and not finalization_only and remaining() > 0:
                    write_json(deadline_path, deadline_notice)
                    dialogue.append({"controller": deadline_notice})
                    finalization_only = True
                    progress["finalization_only"] = True
                    write_json(progress_path, progress)
                    continue
                raise
            write_json(response_path, proposal.model_dump())
        if proposal.action == "probe":
            feedback_path = draft / f"probe_feedback_{attempt + 1}.json"
            if feedback_path.is_file():
                feedback = read_json(feedback_path)
            elif probes is None or corrections or finalization_only:
                feedback = {"status": "rejected", "error": "Learning probes are unavailable in this phase. Finalize the current reward and report."}
                write_json(feedback_path, feedback)
            else:
                allowed = max(0.0, remaining() - reserve)
                if allowed <= 0:
                    feedback = deadline_notice
                else:
                    probe_file = probes.workspace / "output" / f"request_{attempt + 1}.py"
                    if probe_file.exists() and probe_file.read_text(encoding="utf-8") != proposal.reward_source:
                        raise RuntimeError("Saved structured probe source changed")
                    probe_file.write_text(proposal.reward_source, encoding="utf-8")
                    try:
                        feedback = json.loads(invoke(
                            lambda: probes.run(probe_file.relative_to(probes.workspace).as_posix(),
                                               proposal.scenario, proposal.hypothesis), allowed))
                    except asyncio.TimeoutError:
                        # LearningProbes cancels/reaps its worker and preserves
                        # the charged attempt. Spend only the remaining reserve
                        # on finalization rather than stranding a valid draft.
                        feedback = deadline_notice
                write_json(feedback_path, feedback)
            if feedback.get("finalization_only"):
                finalization_only = True
                progress["finalization_only"] = True
                write_json(progress_path, progress)
            event(run, "llm_structured_probe_feedback", "Returned only the registered diagnostic interface",
                  family=family.name, candidate=candidate, evidence_interface=config.EXPERIMENT,
                  feedback_artifact=str(feedback_path.relative_to(run.folder)))
            dialogue.append({"response": proposal.model_dump(), "probe_feedback": feedback})
            continue
        source, report = draft / "reward.py", draft / "report.md"
        source.write_text(proposal.reward_source, encoding="utf-8")
        report.write_text(proposal.report_markdown, encoding="utf-8")
        try:
            validation = validate(source, family.name)
            if config.EXPERIMENT == "exp3" and not validation["has_reward_components"]:
                raise ValueError("Eureka reflection requires additive reward_components")
            _reject_duplicate(validation, pool)
            _check_report(report)
        except (ValueError, TypeError, SyntaxError, RuntimeError) as error:
            event(run, "llm_design_repair", f"Structural correction: {type(error).__name__}: {error}",
                  family=family.name, candidate=candidate, attempt=attempt + 1)
            if corrections >= config.MAX_DESIGN_REPAIRS:
                raise RuntimeError(f"{candidate} did not pass validation: {error}") from error
            corrections += 1
            dialogue.append({"response": proposal.model_dump(), "correction":
                             f"Correct this invalid draft only; no new performance evidence or probes are available. Error: {type(error).__name__}: {error}"})
            continue
        target.mkdir(parents=True, exist_ok=True)
        (target / "reward.py").write_text(proposal.reward_source, encoding="utf-8")
        (target / "report.md").write_text(proposal.report_markdown, encoding="utf-8")
        (target / "agent_final_summary.md").write_text(proposal.reasoning_summary, encoding="utf-8")
        write_json(target / "validation.json", validation)
        write_json(target / "design.json", {
            "family": family.name, "candidate": candidate, "generation": generation,
            "position": position, "model": config.MODEL,
            "prompt_file": f"prompts/{instruction_name}",
            "prompt_hash": file_hash(config.PROMPTS / instruction_name),
            "reward_hash": validation["reward_hash"], "normalized_program_hash": validation["normalized_program_hash"],
            "report_hash": file_hash(target / "report.md"), "evidence_interface": config.EXPERIMENT,
            "design_budget": progress,
            "learning_probes": probes.usage() if probes is not None else {"max_jobs": 0, "charged_jobs": 0},
            "submission": {
                "mode": "structured_response",
                "reward_hash": validation["reward_hash"],
                "report_hash": file_hash(target / "report.md"),
                "request_hash": file_hash(request_path),
                "response_hash": file_hash(response_path),
            },
            "completed_at": now(),
        })
        event(run, "llm_design_completed", f"Validated and committed {candidate}", family=family.name,
              candidate=candidate, reward_hash=validation["reward_hash"], reasoning_summary=proposal.reasoning_summary,
              report=str((target / "report.md").relative_to(run.folder)),
              reward_source=str((target / "reward.py").relative_to(run.folder)))
        return target
    raise RuntimeError(f"{candidate} exhausted its registered dialogue-turn budget; no candidate was committed")
