"""Persistent autonomous reward engineer with an isolated local workspace."""

from __future__ import annotations

import asyncio
import ast
import os
import re
import shlex
import signal
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import config
from common.records import Run, event, file_hash, now, read_json, update, write_json
from common.workspace import prepare_workspace, protected_manifest
from validate_reward import validate


REPORT_HEADINGS = (
    "## Problem identified", "## Evidence inspected", "## Reward design hypothesis",
    "## Changes and rationale", "## Expected cross-scenario benefits", "## Risks and limitations",
)


def _python_path_literals(source: str) -> str:
    """Check Python string paths, not division operators or harmless suffixes.

    This is a mistake detector, not a Python security sandbox. Dynamic file
    access still requires the separate workspace/isolation policy.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source  # Retain the conservative check for unparsed commands.
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    literals = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        parent = parents.get(node)
        # These are string operations, not attempts to open absolute paths.
        if (isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Add)
                and parent.right is node
                and not (isinstance(parent.left, ast.Constant) and parent.left.value == "")):
            continue
        if (isinstance(parent, ast.Call) and isinstance(parent.func, ast.Attribute)
                and parent.func.attr in {"startswith", "endswith"}):
            continue
        literals.append(repr(node.value))
    return "\n".join(literals)


def _absolute_path_text(command: str) -> str:
    """Separate the common Python shell forms from shell path arguments."""
    command = re.sub(
        r"\bpython(?:3)?\s+-\s*<<-?\s*(?P<quote>['\"]?)(?P<marker>\w+)(?P=quote)[^\n]*\n"
        r"(?P<body>.*?)\n(?P=marker)(?=\n|$)",
        lambda match: "python -\n" + _python_path_literals(match["body"]),
        command, flags=re.DOTALL,
    )

    def inline(match):
        try:
            source = shlex.split(match["code"])[0]
        except ValueError:
            return match[0]
        return "python -c " + _python_path_literals(source)

    return re.sub(
        r"\bpython(?:3)?\s+-c\s+(?P<code>'[^']*'|\"(?:\\.|[^\"\\])*\")",
        inline, command,
    )


class WorkspaceShell:
    """Logged local shell in a development-only workspace.

    The command filter and manifest detect mistakes; they are NOT an OS security
    sandbox against adversarial Python. See docs/PROTOCOL_LIMITATIONS.md.
    """

    def __init__(self, workspace: Path, run: Run, family: str, candidate: str, deadline: float | None = None):
        self.workspace = workspace.resolve()
        self.run, self.family, self.candidate = run, family, candidate
        self.deadline = deadline
        artifact_root = run.folder / "llm_tool_outputs" / candidate
        prior_indexes = [
            int(path.stem)
            for path in artifact_root.glob("*.txt")
            if path.stem.isdigit()
        ]
        self.call_index = max(prior_indexes, default=0)

    @staticmethod
    def _reject(command: str) -> str | None:
        # Inspect shell operations separately from quoted code and here-documents.
        # In particular, Python ``del x`` is not the shell's file-deletion command.
        shell = re.sub(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\2(?=\n|$)", "", command, flags=re.DOTALL)
        shell = re.sub(r"(?m)^@(['\"])\r?\n.*?^\1@", "", shell, flags=re.DOTALL)
        shell = re.sub(r"'(?:[^']*)'|\"(?:\\.|[^\"\\])*\"", "''", shell)
        forbidden = (
            r"(?:^|[;&|\n])\s*(?:curl|wget|invoke-webrequest|ssh|scp|git|pip|conda|"
            r"printenv|remove-item|rm|del|erase|rmdir|chmod|chown|sudo|kill|pkill|"
            r"taskkill|start-process)\b|\bpython(?:3)?\s+-m\s+pip\b"
        )
        if re.search(forbidden, shell, re.IGNORECASE) or re.search(
            r"\bos\.environ\b|\bgetenv\b|\bsocket\b|\brequests\s*\.|\bapi_key\b|\$env:|get-childitem\s+env:",
            command, re.IGNORECASE,
        ):
            return "network, credentials, package changes, destructive actions, and process control are unavailable"
        # Match an actual dotenv path/token, not valid modules such as
        # ``jaxmarl.environments``.
        if re.search(r"(^|[\s'\"`/\\])\.env(?![A-Za-z0-9_])", command, re.IGNORECASE):
            return "credential files are unavailable"
        # Match path components, not array ellipses or bitwise-negation masks.
        if re.search(r"(?<![\w.])\.\.(?=[/\\\s'\"),;]|$)|(?<!\w)~(?:[/\\]|[\w.-]+[/\\]|(?=['\"\s]|$))", command):
            return "parent and home paths are unavailable"
        if re.search(r"(?<!\w)[A-Za-z]:[\\/]|(?:^|[\s'\"(=])/(?![/*\s])[\w.-]|\\\\[\w.-]+\\", _absolute_path_text(command)):
            return "absolute paths are unavailable"
        return None

    async def __call__(self, request):
        from agents import ShellCallOutcome, ShellCommandOutput, ShellResult

        event(self.run, "llm_shell_call", "The engineer invoked the shell tool",
              family=self.family, candidate=self.candidate,
              command_count=len(request.data.action.commands))
        outputs = []
        for command in request.data.action.commands:
            remaining = self.deadline - time.monotonic() if self.deadline else config.SHELL_TIMEOUT_SECONDS
            if remaining <= 0:
                outputs.append(ShellCommandOutput(command=command, stdout="", stderr="The current design phase has ended; preserve drafts for finalization.", outcome=ShellCallOutcome(type="timeout", exit_code=None)))
                break
            self.call_index += 1
            event(self.run, "llm_shell_command", f"{self.family} {self.candidate}: executing an autonomous analysis command", family=self.family, candidate=self.candidate, command=command[:10_000], command_truncated=len(command) > 10_000)
            reason = self._reject(command)
            if reason:
                stderr = f"Rejected: {reason}"
                outputs.append(ShellCommandOutput(command=command, stdout="", stderr=stderr, outcome=ShellCallOutcome(type="exit", exit_code=2)))
                event(self.run, "llm_shell_result", stderr, family=self.family, candidate=self.candidate, exit_code=2)
                continue
            environment = {
                "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
                "PYTHONPATH": os.pathsep.join((str(self.workspace / "reference"), str(self.workspace / "reference/runtime"))),
                "TMP": str(self.workspace / "scratch"),
                "TEMP": str(self.workspace / "scratch"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "JAX_ENABLE_COMPILATION_CACHE": "false",
            }
            for name in ("SYSTEMROOT", "WINDIR", "SYSTEMDRIVE", "COMSPEC", "PATHEXT", "PSMODULEPATH", "PROCESSOR_ARCHITECTURE", "NUMBER_OF_PROCESSORS"):
                if os.environ.get(name):
                    environment[name] = os.environ[name]
            powershell = shutil.which("powershell.exe") or shutil.which("powershell")
            if os.name == "nt" and powershell:
                process = await asyncio.create_subprocess_exec(
                    powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
                    "$ErrorActionPreference = 'Stop';\n" + command,
                    cwd=self.workspace, env=environment, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
            else:
                # Stop an unhandled failure before a later successful command
                # hides it. Bash preserves intentional if / && / || handling.
                process = await asyncio.create_subprocess_exec(
                    "bash", "-e", "-o", "pipefail", "-c", command,
                    cwd=self.workspace,
                    env=environment,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
            timed_out = False
            communication = asyncio.create_task(process.communicate())
            try:
                stdout, stderr = await asyncio.wait_for(
                    asyncio.shield(communication),
                    timeout=min(config.SHELL_TIMEOUT_SECONDS, remaining),
                )
            except asyncio.TimeoutError:
                timed_out = True
                await self._stop_process_tree(process)
                stdout, stderr = await communication
            except asyncio.CancelledError:
                await self._stop_process_tree(process)
                await communication
                raise
            full_output, full_error = stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")
            artifact = self.run.folder / "llm_tool_outputs" / self.candidate / f"{self.call_index:04d}.txt"
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(f"COMMAND\n{command}\n\nSTDOUT\n{full_output}\n\nSTDERR\n{full_error}\n", encoding="utf-8")
            output_text, error_text = full_output, full_error
            if len(full_output) > 40_000 or len(full_error) > 20_000:
                # The run artifact is outside the permitted workspace. Keep a
                # readable copy so clipping never makes evidence inaccessible.
                readable = self.workspace / "scratch" / "tool_outputs" / f"{self.call_index:04d}.txt"
                readable.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(artifact, readable)
                full_path = readable.relative_to(self.workspace).as_posix()
                notice = f"\n[Output clipped. Full command, stdout and stderr: {full_path}. Read this workspace-relative file in chunks.]\n"
            if len(full_output) > 40_000:
                output_text = full_output[:8_000] + notice + full_output[-32_000:]
            if len(full_error) > 20_000:
                error_text = notice + full_error[-20_000:]
            if self.deadline:
                output_text += f"\n[Controller: {max(0, self.deadline - time.monotonic()):.0f} seconds remain in this phase. Finalize both files and call submit_candidate before the deadline.]"
            outcome = ShellCallOutcome(type="timeout" if timed_out else "exit", exit_code=None if timed_out else process.returncode)
            outputs.append(ShellCommandOutput(command=command, stdout=output_text, stderr=error_text, outcome=outcome))
            event(self.run, "llm_shell_result", f"Command finished with exit code {process.returncode}", family=self.family, candidate=self.candidate, exit_code=process.returncode, timed_out=timed_out, output_artifact=str(artifact.relative_to(self.run.folder)), stdout_preview=output_text[:2_000], stderr_preview=error_text[:2_000])
            if timed_out:
                break
        return ShellResult(output=outputs, provider_data={"working_directory": "."})

    @staticmethod
    async def _stop_process_tree(process) -> None:
        if process.returncode is not None:
            return
        try:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()


def _usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    details = getattr(usage, "input_tokens_details", None)
    return {
        "requests": int(getattr(usage, "requests", 1) or 1),
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "cached_input_tokens": int(getattr(details, "cached_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
    }


def _reasoning_summary(response: Any) -> str:
    texts = []
    for item in getattr(response, "output", ()):
        if getattr(item, "type", None) == "reasoning":
            texts.extend(str(part.text) for part in getattr(item, "summary", ()) if getattr(part, "text", None))
    return "\n\n".join(texts)[:8000]


def _submission(run: Run, workspace: Path, candidate: str) -> dict[str, Any]:
    """An interrupted draft is not a submission; both exact artifact hashes must match."""
    receipt = read_json(run.folder / "llm_submissions" / f"{candidate}.json", {})
    for name in ("reward", "report"):
        path = workspace / "output" / ("reward.py" if name == "reward" else "report.md")
        if not path.is_file() or receipt.get(f"{name}_hash") != file_hash(path):
            return {}
    return receipt


def _submit_candidate(run: Run, workspace: Path, family: config.FamilyConfig, candidate: str, summary: str, mode: str = "explicit_tool") -> dict[str, Any]:
    """Record deliberate completion, without judging the reward's scientific merit."""
    if read_json(run.state_file, {}).get("reward_locked"):
        raise RuntimeError("Reward design is locked")
    registered = read_json(workspace / "workspace_manifest.json", {}).get("protected_files")
    if registered is not None and registered != protected_manifest(workspace):
        raise RuntimeError("Reward engineer modified protected evidence")
    validation = validate(workspace / "output/reward.py", family.name)
    _reject_duplicate(validation, _prior_reward_pool(run, exclude=candidate))
    _check_report(workspace / "output/report.md")
    receipt = {
        "candidate": candidate, "reward_hash": validation["reward_hash"],
        "report_hash": file_hash(workspace / "output/report.md"),
        "summary": summary.strip(), "mode": mode, "submitted_at": now(),
    }
    write_json(run.folder / "llm_submissions" / f"{candidate}.json", receipt)
    event(run, "llm_candidate_submitted", f"The engineer explicitly submitted {candidate}", family=family.name, **receipt)
    return receipt


async def _run_agent(run: Run, workspace: Path, family: config.FamilyConfig, candidate: str, task: str, timeout_seconds: float | None = None, *, learning_probes=None) -> str:
    from agents import Agent, MaxTurnsExceeded, ModelSettings, OpenAIResponsesModel, RunHooks, Runner, SQLiteSession, ShellTool, function_tool
    from openai import AsyncOpenAI
    from openai.types.shared import Reasoning

    timeout = config.AGENT_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
    shell = WorkspaceShell(workspace, run, family.name, candidate, time.monotonic() + timeout)
    task += f"\n\nThis phase has at most {timeout:.0f} seconds. Tool outputs report remaining time. Finish by calling submit_candidate after the report accurately describes the saved reward; a final prose message alone does not submit files."

    @function_tool
    def submit_candidate(summary: str) -> str:
        """Explicitly submit the saved reward and its accurate final report for training.

        Args:
            summary: Concise account of the completed investigation and artifacts.
        """
        event(run, "llm_submit_candidate_call", "The engineer invoked the submission tool",
              family=family.name, candidate=candidate, summary=summary)
        _submit_candidate(run, workspace, family, candidate, summary)
        return summary

    instructions = (config.PROMPTS / "reward_engineer.md").read_text(encoding="utf-8")
    agent_tools = [ShellTool(executor=shell, needs_approval=False), submit_candidate]
    if learning_probes is not None:
        task += "\n\n" + learning_probes.instructions()

        @function_tool
        async def run_learning_probe(reward_file: str, scenario: str, hypothesis: str) -> str:
            """Train a saved reward draft with the fixed learner on one development scenario.

            Args:
                reward_file: Workspace-relative Python reward source to test.
                scenario: One of the permitted development scenarios.
                hypothesis: The question this learning experiment is intended to investigate.
            """
            return await learning_probes.run(reward_file, scenario, hypothesis)

        agent_tools.append(run_learning_probe)

    class Hooks(RunHooks):
        async def on_llm_start(self, context, agent, system_prompt, input_items):
            del context, agent, system_prompt, input_items
            update(run, activity=f"{family.name} {candidate}: the autonomous engineer is reasoning")
            event(run, "llm_reasoning_started", "The reward engineer started an investigation turn", family=family.name, candidate=candidate)

        async def on_llm_end(self, context, agent, response):
            del context, agent
            event(run, "llm_reasoning_completed", "The reward engineer completed an investigation turn", family=family.name, candidate=candidate, usage=_usage(response), reasoning_summary=_reasoning_summary(response) or None)

    calls = run.folder / "llm_requests" / candidate
    calls.mkdir(parents=True, exist_ok=True)
    request_number = len(list(calls.glob("*.json"))) + 1
    write_json(calls / f"{request_number:04d}.json", {
        "model": config.MODEL, "settings": config.protocol(family)["model"],
        "instructions": instructions,
        "task": task, "workspace_manifest": read_json(workspace / "workspace_manifest.json", {}),
        "phase_timeout_seconds": timeout,
        "session_id": _session_id(run, family, candidate),
    })
    client = AsyncOpenAI(timeout=timeout, max_retries=0)
    engineer = Agent(
        name=f"{family.name} autonomous reward engineer",
        model=OpenAIResponsesModel(config.MODEL, openai_client=client),
        instructions=instructions,
        tools=agent_tools,
        tool_use_behavior={"stop_at_tool_names": ["submit_candidate"]},
        model_settings=ModelSettings(
            reasoning=Reasoning(
                mode=config.REASONING_MODE,
                effort=config.REASONING_EFFORT,
                context="all_turns",
                summary="detailed",
            ),
            verbosity=config.MODEL_VERBOSITY,
            parallel_tool_calls=False,
            truncation="auto",
            store=True,
            context_management=[{"type": "compaction", "compact_threshold": 100_000}],
        ),
    )
    # A shared family session would leak A's private probes and reasoning to B.
    # Earlier-generation history is explicitly supplied through the workspace.
    session = SQLiteSession(_session_id(run, family, candidate), run.folder / "agent_sessions.sqlite")
    event(run, "llm_transport_configured", "No hidden SDK retries; candidate-local session and shared design deadline",
          family=family.name, candidate=candidate, timeout_seconds=timeout, sdk_max_retries=0)
    try:
        async with client:
            result = await asyncio.wait_for(Runner.run(engineer, task, session=session, max_turns=config.MAX_AGENT_TURNS, hooks=Hooks()), timeout=timeout)
        return str(result.final_output or "")
    except (asyncio.TimeoutError, MaxTurnsExceeded) as error:
        event(run, "llm_budget_reached", f"The engineer reached its registered controller budget: {type(error).__name__}", family=family.name, candidate=candidate)
        return ""


def _session_id(run: Run, family: config.FamilyConfig, candidate: str) -> str:
    return f"{run.id}_{family.name}_{candidate}_engineer"


async def _finalize_report(
    run: Run,
    workspace: Path,
    family: config.FamilyConfig,
    candidate: str,
    timeout_seconds: float | None = None,
) -> str:
    """Ask the same engineer for documentation only, without executable tools."""
    from agents import Agent, MaxTurnsExceeded, ModelSettings, OpenAIResponsesModel, Runner, SQLiteSession
    from openai import AsyncOpenAI
    from openai.types.shared import Reasoning

    reward_source = (workspace / "output" / "reward.py").read_text(encoding="utf-8")
    reward_hash = file_hash(workspace / "output/reward.py")
    headings = "\n".join(REPORT_HEADINGS)
    task = f"""Finalize the missing Markdown report for {family.name} {candidate}.

The reward program below already passed the fixed structural validator. Do not
redesign it and do not request further diagnostics. Using your completed
investigation in this session, return only the concise report body with exactly
these required headings:

{headings}

Accurately describe the evidence you actually inspected, the implemented
hypothesis, expected cross-scenario benefits, and important limitations.
Returning this report explicitly finalizes this exact reward program for
training; do not return unfinished intentions or descriptions of different code.

```python
{reward_source}
```
"""
    timeout = min(config.AGENT_TIMEOUT_SECONDS, 600) if timeout_seconds is None else timeout_seconds
    client = AsyncOpenAI(timeout=timeout, max_retries=0)
    finalizer = Agent(
        name=f"{family.name} reward report finalizer",
        model=OpenAIResponsesModel(config.MODEL, openai_client=client),
        instructions=(
            "Complete documentation for an already designed reward. Return only "
            "the requested Markdown report. You have no tools and must not propose "
            "more analysis or change the reward."
        ),
        tools=[],
        model_settings=ModelSettings(
            reasoning=Reasoning(
                mode=config.REASONING_MODE,
                effort=config.REASONING_EFFORT,
                context="all_turns",
                summary="detailed",
            ),
            verbosity=config.MODEL_VERBOSITY,
            parallel_tool_calls=False,
            truncation="auto",
            store=True,
            context_management=[{"type": "compaction", "compact_threshold": 100_000}],
        ),
    )
    session = SQLiteSession(
        _session_id(run, family, candidate),
        run.folder / "agent_sessions.sqlite",
    )
    calls = run.folder / "llm_requests" / candidate
    calls.mkdir(parents=True, exist_ok=True)
    request_number = len(list(calls.glob("*.json"))) + 1
    write_json(calls / f"{request_number:04d}.json", {
        "purpose": "documentation_only", "model": config.MODEL,
        "settings": config.protocol(family)["model"],
        "instructions": finalizer.instructions, "task": task,
    })
    event(
        run,
        "llm_report_finalization_started",
        f"Finalizing the missing report for {candidate} without shell access",
        family=family.name,
        candidate=candidate,
    )
    try:
        async with client:
            result = await asyncio.wait_for(
                Runner.run(finalizer, task, session=session, max_turns=2), timeout=timeout,
            )
    except (asyncio.TimeoutError, MaxTurnsExceeded) as error:
        event(
            run,
            "llm_report_finalization_failed",
            f"Report finalization reached its budget: {type(error).__name__}",
            family=family.name,
            candidate=candidate,
        )
        return ""
    report = str(result.final_output or "").strip()
    if report.startswith("```markdown") and report.endswith("```"):
        report = report[len("```markdown"):-3].strip()
    elif report.startswith("```") and report.endswith("```"):
        report = report[3:-3].strip()
    if file_hash(workspace / "output/reward.py") != reward_hash:
        raise RuntimeError("Reward changed during documentation-only finalization")
    event(
        run,
        "llm_report_finalization_completed",
        f"Completed the report-only finalization for {candidate}",
        family=family.name,
        candidate=candidate,
        report_characters=len(report),
        usage=_usage(result.context_wrapper),
    )
    return report


def _check_report(path: Path) -> None:
    if not path.is_file():
        raise ValueError("output/report.md is missing")
    source = path.read_text(encoding="utf-8")
    missing = [heading for heading in REPORT_HEADINGS if heading not in source]
    if missing:
        raise ValueError("report.md is missing headings: " + ", ".join(missing))


def _prior_reward_pool(run: Run, exclude: str | None = None) -> list[dict[str, str]]:
    """Return fingerprints for every finalized candidate in this family run."""
    pool = []
    root = run.folder / "family" / "candidates"
    for prior in config.CANDIDATES:
        if prior == exclude:
            continue
        design = read_json(root / prior / "design.json", {})
        if design:
            pool.append({
                "candidate": prior,
                "reward_hash": str(design.get("reward_hash") or ""),
                "normalized_program_hash": str(
                    design.get("normalized_program_hash") or ""
                ),
            })
    return pool


def _reject_duplicate(validation: dict[str, Any], pool: list[dict[str, str]]) -> None:
    for prior in pool:
        if validation["reward_hash"] == prior["reward_hash"]:
            raise ValueError(
                "reward.py duplicates a registered candidate; "
                "produce a materially different reward program"
            )
        if (
            prior["normalized_program_hash"]
            and validation["normalized_program_hash"]
            == prior["normalized_program_hash"]
        ):
            raise ValueError(
                "reward.py duplicates a registered executable program "
                "after comments and docstrings are removed"
            )


def design_candidate(run: Run, family: config.FamilyConfig, candidate: str, generation: int, position: str) -> Path:
    """Investigate and commit one reward under the registered search settings."""
    budget = config.AGENT_TIMEOUT_SECONDS
    learning_probes = None
    if budget <= 0:
        raise ValueError("Design budget must be positive")
    if config.EXPERIMENT != "exp1":
        from common.structured_designer import design_control
        return design_control(run, family, candidate, generation, position)
    if read_json(run.state_file, {}).get("reward_locked"):
        raise RuntimeError(f"Reward design is locked for {family.name}")
    workspace = prepare_workspace(run, family, candidate, generation, position)
    if learning_probes is None and config.LEARNING_PROBE_JOBS:
        from common.learning_probes import LearningProbes
        learning_probes = LearningProbes(
            run, family, candidate, workspace, max_jobs=config.LEARNING_PROBE_JOBS,
            seed=config.TRAIN_SEEDS[0], timeout_seconds=config.LEARNING_PROBE_TIMEOUT_SECONDS,
            evidence_mode="autonomous",
        )
    before = protected_manifest(workspace)
    reward_pool = _prior_reward_pool(run, exclude=candidate)
    visible_names = {name for pair in config.GENERATIONS[:generation - 1] for name in pair}
    visible_pool = [item for item in reward_pool if item["candidate"] in visible_names]
    distinction = "Both candidates are independent within this generation. Sibling designs, reasoning, probes and results are private until the generation completes. Use only completed earlier-generation evidence."
    task = f"""Design and finalize {candidate}, generation {generation} for {family.name}.

{distinction}

Use the complete permitted workspace and decide the investigation yourself.
The same source will be trained independently on these development scenarios:
{', '.join(family.development)}.

The finalized reward pool is:
{visible_pool if visible_pool else 'empty (no earlier generation)'}

Your candidate must use a materially distinct reward hypothesis or a meaningful
functional revision. It must not reproduce any prior executable reward. Reusing
well-supported concepts is allowed; returning the same program with cosmetic
edits, renamed variables, comments, or prose changes is not.

Write and validate output/reward.py and output/report.md. In your final response,
submit the exact completed artifacts with submit_candidate and a concise summary.
"""
    output_reward, output_report = workspace / "output/reward.py", workspace / "output/report.md"
    resumed_artifacts = output_reward.is_file() or output_report.is_file()
    progress_path = run.folder / "llm_progress" / f"{candidate}.json"
    progress = read_json(progress_path, {"elapsed_seconds": 0.0})
    if progress.get("budget_seconds", budget) != budget:
        raise RuntimeError("Cannot resume with a different design-time budget")
    reserve = min(300.0, budget * 0.10)

    def remaining() -> float:
        return max(0.0, budget - progress["elapsed_seconds"])

    def invoke(call, limit: float) -> str:
        # Repairs and finalization consume the same active-time budget. Graceful
        # interruption preserves spent time and drafts; hard kills cannot run
        # this finally block (see the documented provenance limitation).
        allowed = min(limit, remaining())
        if allowed <= 0:
            return ""
        started = time.monotonic()
        try:
            return asyncio.run(call(allowed))
        finally:
            progress["elapsed_seconds"] += time.monotonic() - started
            progress["budget_seconds"] = budget
            progress["finalization_reserve_seconds"] = reserve
            write_json(progress_path, progress)

    update(run, family=family.name, candidate=candidate, phase="reward_design", activity=f"{family.name} {candidate}: autonomous source investigation", current_job=None)
    if resumed_artifacts:
        event(
            run,
            "llm_design_resumed",
            f"Resumed verified unfinished artifacts for {candidate}",
            family=family.name,
            candidate=candidate,
            generation=generation,
        )
    else:
        event(run, "llm_design_started", f"Started autonomous design for {candidate}", family=family.name, candidate=candidate, generation=generation)
    receipt = _submission(run, workspace, candidate)
    final_summary = receipt.get("summary", "")
    if not receipt:
        task += f"\nThe total candidate budget is {budget:.0f}s, including repairs; {reserve:.0f}s are reserved for finalization."
        if resumed_artifacts:
            task += "\nResume your existing output and scratch files; they are drafts, not a completed submission."
        final_summary = invoke(
            lambda limit: _run_agent(run, workspace, family, candidate, task, timeout_seconds=limit, learning_probes=learning_probes),
            max(0.0, remaining() - reserve),
        )
    validation, last_error = None, None
    for attempt in range(config.MAX_DESIGN_REPAIRS + 1):
        try:
            if not output_reward.is_file():
                raise ValueError("output/reward.py is missing")
            validation = validate(output_reward, family.name)
            _reject_duplicate(validation, reward_pool)
            _check_report(output_report)
            if not _submission(run, workspace, candidate):
                raise ValueError("Explicit submission is missing for the current reward.py and report.md")
            last_error = None
            break
        except Exception as error:
            last_error = error
            if attempt >= config.MAX_DESIGN_REPAIRS or remaining() <= 0:
                break
            event(run, "llm_design_repair", f"Requested correction after validation failed: {type(error).__name__}: {error}", family=family.name, candidate=candidate, attempt=attempt + 1)
            if validation is not None and "report.md" in str(error) and "Explicit submission" not in str(error):
                report = invoke(
                    lambda limit: _finalize_report(run, workspace, family, candidate, timeout_seconds=limit),
                    min(reserve, remaining()),
                )
                if report:
                    output_report.write_text(report + "\n", encoding="utf-8")
                    final_summary = "Completed the missing report for the validated reward."
                    try:
                        _submit_candidate(run, workspace, family, candidate, final_summary, mode="documentation_only")
                    except (ValueError, RuntimeError) as submission_error:
                        last_error = submission_error
                continue
            repair = f"""The proposed {candidate} artifacts did not pass the fixed structural validation: {type(error).__name__}: {error}

Correct the required artifact before doing any optional analysis. Write both
output/reward.py and output/report.md, rerun the fixed validator, and finish.
Do not launch exploratory or long-running diagnostics until valid draft
artifacts exist. Do not change the experiment. This is the reserved finalization
phase: finish and accurately document the existing hypothesis, not a new research
investigation. Submit using submit_candidate; valid files alone are not a submission.
"""
            final_summary = invoke(
                lambda limit: _run_agent(run, workspace, family, candidate, repair, timeout_seconds=limit),
                min(reserve, remaining()),
            )
    after = protected_manifest(workspace)
    if before != after:
        changed = sorted(name for name in set(before) | set(after) if before.get(name) != after.get(name))
        raise RuntimeError("Reward engineer modified protected evidence: " + ", ".join(changed))
    if last_error is not None or validation is None:
        raise RuntimeError(f"{family.name} {candidate} was not submitted successfully: {last_error}. Drafts and the remaining budget are preserved; no candidate was committed.")
    receipt = _submission(run, workspace, candidate)
    final_summary = receipt["summary"]
    target = run.folder / "family" / "candidates" / candidate
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copy2(output_reward, target / "reward.py")
    shutil.copy2(output_report, target / "report.md")
    (target / "agent_final_summary.md").write_text((final_summary.strip() or "Agent completed artifacts without a final prose summary") + "\n", encoding="utf-8")
    write_json(target / "validation.json", validation)
    write_json(target / "design.json", {
        "family": family.name, "candidate": candidate, "generation": generation,
        "position": position, "reward_hash": validation["reward_hash"],
        "normalized_program_hash": validation["normalized_program_hash"],
        "workspace": str(workspace.relative_to(run.folder)), "model": config.MODEL,
        "report_hash": file_hash(target / "report.md"),
        "prompt_file": "prompts/reward_engineer.md",
        "prompt_hash": file_hash(config.PROMPTS / "reward_engineer.md"),
        "completed_at": now(),
        "submission": receipt,
        "design_budget": progress,
        **({"learning_probes": learning_probes.usage()}
           if learning_probes is not None else {}),
    })
    event(run, "llm_design_completed", f"Finalized and validated {candidate}", family=family.name, candidate=candidate, reward_hash=validation["reward_hash"], reasoning_summary=final_summary.strip()[:8000] or "Agent completed the validated artifacts without a prose summary.", report=str((target / "report.md").relative_to(run.folder)), reward_source=str((target / "reward.py").relative_to(run.folder)))
    return target
