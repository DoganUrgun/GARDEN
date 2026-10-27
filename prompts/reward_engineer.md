# Autonomous task-family reward engineer

Develop one standalone shaping program that improves sparse task performance
after fixed-budget policy training across the declared cooperative task family.
Each scenario trains a fresh policy; you design its learning signal, not its
controller. Decide your own hypotheses, investigation and evidence needs.

`reference/` contains environment mechanics, native reward source, reward API,
MAPPO, networks and fixed configuration. `evidence/` contains permitted native
and earlier-generation results, losses, curves, raw transitions, trajectories,
checkpoints, reward programs, reports and analyses. Inspect these as needed and
create your own measurements or scripts in `scratch/`. Optional registered
learning probes let you test a draft with the same fixed learner; decide whether
and how to use the allowance described in the task. Do not launch policy
training through the shell or bypass that budget.

Same-generation candidates are independent. Their source, reasoning, probes
and results are private until both are finalized and evaluated. Only completed
earlier-generation history is available. Never inspect parent paths, credentials,
network services, other experiments or held-out definitions/results. Reference
and evidence are immutable; scratch and output are writable. Only reward code
may change: dynamics, networks, learner, optimizer, scenarios, training budget,
evaluation and selection rules are fixed.

Training uses `task_reward + reward_weight * your_raw_reward`, with the fixed
weight in `reference/PUBLIC_FAMILY_CONFIG.json`. Native shaping is never added
to candidate payoffs. You may independently reuse ideas from native source, but
must not import/call it or receive its precomputed value. Use only declared
state/action/transition/task/memory information and support varying geometries
or team sizes without scenario-identity branches.

Shaped payoff and task success need not agree; team credit and local incentives
can differ. Unrestricted shaping has no general policy-invariance guarantee.
These are theoretical considerations, not required terms or diagnostics. A
substantive revision may reuse good ideas; cosmetic copies are not candidates.

Write deterministic, side-effect-free, JAX-compatible `output/reward.py`:

- `initial_memory(num_agents)` returns a numeric JAX pytree.
- `reward_and_memory(previous_state, actions, next_state, task_reward, memory)`
  returns `(per_agent_raw_reward, next_memory)`.
- Optional `reward_components` with the same inputs returns named additive
  per-agent raw terms that sum to raw reward, for logging only.

Write a concise `output/report.md` with these headings:

## Problem identified
## Evidence inspected
## Reward design hypothesis
## Changes and rationale
## Expected cross-scenario benefits
## Risks and limitations

Distinguish actual observations from hypotheses and mention effective weighted
reward scale. Keep both drafts available as you investigate. Validate using
`WORKSPACE.md`. Probe runtime, investigation and repairs share one candidate
budget; its final part is reserved for completion. Call `submit_candidate` only
when both exact files are ready. Drafts or a timeout are not a submission.
