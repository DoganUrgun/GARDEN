Design executable reward functions using evolutionary reward reflection. Use
the supplied environment, learner, reward interface and task description.
Candidates are independent samples from the same generation context. After
generation 1, history contains only the preceding generation's highest-scoring
reward, its report, and standardized performance/component feedback. No sibling
source, reasoning, probe results or training results are visible. Reflect on
the supplied evidence and propose a substantively revised reward.

You may request an optional learning probe using a structured response with
action="probe", reward_source=<complete draft>, scenario=<development scenario>,
hypothesis=<your investigation question>. Other fields can contain a draft
report and concise reasoning summary. A fresh policy uses the fixed settings;
only standardized performance and component feedback is returned. The task
states the allowance; probes count toward the same design deadline. No shell,
arbitrary artifacts, custom diagnostics or other evidence requests are available.
Finalize with action="finalize", complete reward_source, report_markdown and
reasoning_summary; scenario and hypothesis may be empty.

Training is task_reward + reward_weight * your raw reward. Native shaping is
absent. You may independently reuse native ideas, but cannot call/import the
native reward. Only reward code changes. Learner, budgets, scenarios, evaluation
and selection are fixed. Support varying team sizes without scenario-identity
branches. Sparse task return determines fitness, not shaped payoff. Unrestricted
shaping has no general policy-invariance guarantee.

Define initial_memory(num_agents); reward_and_memory(previous_state, actions,
next_state, task_reward, memory) returning per-agent raw rewards and next memory;
and reward_components with the same inputs returning named additive per-agent
raw terms. Components must sum to raw reward and are logged for reflection only.
Use deterministic side-effect-free JAX. The concise report uses the six supplied
headings, distinguishes observations from hypotheses, and states limitations.
