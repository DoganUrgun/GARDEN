# GARDEN experimental protocol

## Objective and scope

Learn one standalone shaping program per cooperative environment family using
three development scenarios. Select and freeze its exact source before testing
on two scenarios excluded from reward design. Transfer means training fresh
policies with unchanged reward code, not transferring a trained policy without
further learning.

The exact split is registered in `config.py`. Held-out identifiers are omitted
from this document because it is included in the designer's permitted source
packet. The designer receives development-only configuration and source.

## Common reward and training contract

Native training uses `task + native_shaping`. Candidate training uses
`task + family_weight * raw_generated_shaping`, with weights 0.10 for Overcooked
and 0.01 for SMAX. Native shaping is not passed to, imported by, or automatically
added to a generated program. A program may independently reconstruct similar
concepts from legitimate transition information.

The reward interface exposes previous state, joint actions, next state, sparse
task reward, and candidate-owned episode memory. Optional additive components
are for logging and must sum to the raw generated shaping reward.

Overcooked task reward is 20 per delivered soup. SMAX task reward is terminal
victory: all enemies eliminated with a surviving ally. Losses, draws, and
timeouts receive zero. Sparse episode return is the evaluation objective. Native
and generated shaping are stationary; there is no reward annealing.

Each reward/scenario/training-seed combination receives a policy trained from
scratch. Within each family, conditions share MAPPO, network architecture,
optimizer, evaluation action sampling, and training budget. Both families request
5,000,000 environment transitions and execute 4,997,120 transitions in 305 full
rollout batches. Episode limits are 400 transitions for Overcooked and 100 for
SMAX. The linear learning-rate schedule spans that training budget.

Overcooked uses feed-forward CNN actor and centralized critic networks; SMAX
uses recurrent actor and critic networks. The default final evaluation is 64
episodes per Overcooked policy and 256 per SMAX policy, with sampled rather than
greedy actions. Exact settings and actual step counts are registered.

## Three experimental conditions

Every search generates six valid rewards in three generations of two candidates.
Both siblings' final programs are committed before that generation's assessment
evidence is released. Siblings have independent sessions and workspaces: neither
sees the other's source, report, private probes, or results. Only completed
earlier generations can enter subsequent design context.

1. **GARDEN (Experiment 1):** autonomous investigation of permitted source and
   raw learning evidence, shell analyses, and the complete completed
   earlier-generation history. No researcher-chosen diagnostic vector or
   reward-repair hypothesis is prescribed. This condition runs in both families
   and is compared with native shaping.
2. **GARDEN-Fixed (Experiment 2):** SMAX only, with the same model and sibling
   history rule, but no shell, arbitrary file inspection, or ad-hoc evidence
   requests. The complete initial mechanics/learner source packet is supplied.
   Fixed reports contain per-scenario/seed sparse return, five-bin task learning
   curves, episode length, policy/value loss, entropy, and actual training
   shaping mean, standard deviation, minimum, and maximum. Native development
   reports and all completed earlier-generation candidate reports are visible.
3. **Eureka-inspired (Experiment 3):** SMAX only, a budget-matched adaptation,
   not an original-paper reproduction. Siblings receive identical initial
   generation contexts in independent sessions. Later generations receive only
   the preceding generation's highest-scoring program/report and standardized
   task/component feedback, including its permitted probe feedback. Feedback
   additionally includes per-component raw reward statistics in each learning
   curve bin. The model, reward interface, family objective, and six-candidate
   policy-training budget match the other conditions.

Restricted reports include the registered aggregate selection score and
scenario-level normalized improvements. Shaping statistics summarize training
rollout agent-transitions, not a selected set of final trajectories.

## Bounded learning probes and candidate submission

All conditions allow up to two private full-budget learning probes per candidate,
or twelve per search. Each probe trains a new policy on one chosen development
scenario with the first registered development training seed. The designer may
choose the reward source, scenario, and hypothesis, but not the learner, step
budget, evaluation, or seed.

A launched failed or interrupted probe consumes its slot. There is no automatic
training retry or additional hidden training. The complete probe counts toward
the candidate's shared 5,400-second active design budget, including reasoning,
tools, repairs, and finalization. The probe worker timeout is 3,600 seconds.
GARDEN receives raw probe artifacts. Controls request probes through structured
messages and receive only their registered report schemas.

This is an equal maximum allowance, not a guarantee of identical realized
compute. Record charged jobs, actual transitions, reused jobs, model calls,
tokens, and wall time for every condition.

Probe results do not directly determine selection. Each committed reward must
receive a complete assessment across all three development scenarios and all
registered development training seeds. A successful probe may fill one assessment
cell only for the same candidate when reward hash, scenario, seed, learner,
training/evaluation/snapshot settings, execution provenance, and checkpoint and
artifact hashes match exactly. This reuse is recorded as one training job, not
an independent replication. Changed drafts, other candidates, other conditions,
and held-out jobs cannot be reused through this rule. Unfavorable valid results
are retained.

Exact-source and normalized-AST duplicates are rejected. These tests detect
identical programs, not general semantic equivalence. Candidate source and
behavior still require scientific interpretation.

Up to 10% of the candidate's shared time budget, capped at 300 seconds, is
reserved for finalization. Only an explicit submission with recorded reward and
report hashes commits a candidate; a timeout never accepts an unfinished draft.
Controls' structured reward/report responses are explicit submissions validated
before training. Missing-document recovery is recorded separately and cannot
change reward source. Repairs and resumes do not reset the candidate's budget.

## Observational learning evidence

In addition to final checkpoints, learning curves, and raw trajectories, the
learner captures intermediate checkpoints and two raw rollouts at the first
existing training chunk reaching 20% and 60% of the budget. Actual capture steps
are recorded. These rollouts have a separate random stream: they do not change
training samples, update boundaries, or selection evaluations.

GARDEN chooses whether and how to analyze its permitted evidence. No event vector
or preferred diagnostic is imposed. Controls do not receive raw trajectories,
checkpoints, or arbitrary file access. Reference and assessment-cell reuse require
matching snapshot settings.

## Mechanical selection and frozen transfer

For development scenario `i`, let `J_i` be mean sparse evaluation return over
registered policy-training seeds. Rank candidates by:

```text
score = mean_i((J_i(candidate) - J_i(native)) / max(abs(J_i(native)), 1))
```

Ties are broken by higher worst-scenario normalized improvement, then higher
mean sparse return, then ascending candidate identifier. Select the best of all
six valid candidates; there is no subjective language-model veto. This is
development selection, not an independent validation estimate.

The registered suite completes and freezes all three SMAX searches before any
SMAX held-out training. Freeze records contain exact reward-source hashes, and
the evaluation entry point enforces this barrier on resume. Before freezing,
only compatible native development artifacts may pass from Experiment 1 to
the controls. No condition sees another condition's reward-search evidence.

After freezing, native and selected-reward policies are trained from scratch on
both held-out scenarios at ten training seeds (0 through 9) by default. Controls
train their own selected-reward policies and reuse the compatible Experiment 1 native and
GARDEN held-out results only for comparison. No further reward-design model
calls are permitted in the run after freezing.

The default is one development policy seed and one reward-search realization
per condition/family. Repeating policy seeds does not repeat reward search.
`SEARCH_REPLICATE` registers a separate search with separate artifacts and no
cross-search history. Replicate choices must be fixed before inspecting
comparisons; do not search repeatedly until a favorable result appears.

## Provenance, boundaries, and interpretation

Manifests record model/settings, instruction sources, dependencies, source and
reward/report hashes, family/seed/configuration, device, and job budget. The
registered model is `gpt-6.1-sol`, with reasoning mode `pro`, reasoning effort
`max`, and verbosity `medium`. Changing a model or scientific setting requires
a new registered run, not an incompatible resume.

Development workspaces exclude held-out definitions/results and other searches.
Protected copies are hashed. Local command restrictions are not an operating
system security sandbox against adversarial generated Python. Use an isolated
execution environment without unrelated credentials or private data.

Held-out exclusion is a boundary on designer evidence, not proof that scenario
selection was researcher-blind. Results from one selected reward do not establish
repeatability across independent language-model searches. Selection estimates
can be optimistic because the same development scenarios score all candidates.
Report search replication separately from policy-seed variation and report
realized resource use alongside performance.

Completed jobs resume only under a compatible manifest. Source, dependencies,
protocol, and accelerator checks guard reuse. Preserve raw learning artifacts,
including unsuccessful proposals and probes, as part of the reproducibility
record. Execution smoke tests do not establish learning quality or superiority
over native shaping.
