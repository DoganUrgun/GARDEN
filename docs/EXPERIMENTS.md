# Running experiments

This guide describes the executable workflow. The scientific contract and
evidence restrictions are in [PROTOCOL.md](../PROTOCOL.md). Complete
[installation](INSTALL.md) before starting policy training or model calls.

## Inspect before running

From the repository root with the virtual environment activated:

```bash
python main.py
python main.py plan
```

The first command shows help. The second explains the configured study without
registering a search, calling the model, or training a policy.

The paper asks two questions: **does GARDEN improve transfer relative to native
shaping**, and **does autonomous evidence navigation help relative to restricted
reward designers**? The first is tested in Overcooked V1 and SMAX; the second is
tested in SMAX. Three design conditions implement these comparisons:

| Internal result label | Design condition | Scope |
|---|---|---|
| `exp1` | Autonomous GARDEN, compared with native shaping | One search per family: Overcooked V1 and SMAX |
| `exp2` | GARDEN-Fixed, with prescribed reports and restricted evidence access | One SMAX family-level search |
| `exp3` | Eureka-inspired best-parent/component-feedback control | One SMAX family-level search |

These labels organize results; they are not separate launch commands.
The controls are compared with the compatible GARDEN/native SMAX reference.
They are not per-scenario searches. Every selected program is reused unchanged
across the family's scenarios while policies are trained separately.

## Run the controlled suite

```bash
python main.py --new
```

The queue registers four fresh runs before execution and uses these seven stages:

1. SMAX GARDEN search and reward freeze.
2. SMAX GARDEN-Fixed search and reward freeze.
3. SMAX Eureka-inspired search and reward freeze.
4. SMAX GARDEN and native held-out evaluation.
5. SMAX GARDEN-Fixed held-out evaluation and comparison.
6. SMAX Eureka-inspired held-out evaluation and comparison.
7. Overcooked GARDEN search and held-out evaluation.

All three SMAX rewards must be frozen before any SMAX held-out policy is trained.
The controller preserves this barrier on resume. Only native development
evidence is shared into control searches; another condition's candidate rewards
or held-out evidence are never designer inputs.

One training worker runs at a time. Do not start another suite on the same GPU
while one is active. The suite stops on a failed stage; it does not automatically
replace a run or reset a candidate's allowance.

## Budget at the defaults

Each family uses three development scenarios and two held-out scenarios. Six
candidate rewards are assessed in every search. The default is one development
policy seed, ten held-out policy seeds, and up to two private probes per
candidate.

| Run | Required assessment jobs before probe reuse | Optional probe allowance | Maximum new training jobs |
|---|---:|---:|---:|
| GARDEN/native, Overcooked | 61 | 12 | 73 |
| GARDEN/native, SMAX | 61 | 12 | 73 |
| GARDEN-Fixed, SMAX | 38 | 12 | 50 |
| Eureka-inspired, SMAX | 38 | 12 | 50 |
| Total | 198 | 48 | 246 |

For each GARDEN/native family, the 61 assessment jobs are 3 native development
jobs, 18 candidate development jobs, and 40 held-out jobs (2 methods x 2 scenarios
x 10 seeds). Each control adds 18 candidate development jobs and 20 selected-reward
held-out jobs. Native development and native/GARDEN held-out comparison records
are reused from the compatible GARDEN/native reference.

Every full job requests 5,000,000 environment transitions and executes 4,997,120
after rollout-batch rounding. Optional probes use the same full budget. A valid
exact-match probe can fill its own candidate's corresponding assessment cell,
reducing new work. Unused probes also reduce work. Probe reuse is recorded and
is not an independent replication.

Model calls, source validation, evaluation rollouts, and raw evidence capture
add work beyond these policy-training job counts. Equal
maximum allowances do not imply equal realized compute. Report actual charged
jobs, transitions, model usage, reuse, and wall time.

## Seeds, model, and other settings

Settings are read from the environment and `.env` before run registration.
Change them only for a new registered configuration.

| Variable | Default | Effect |
|---|---|---|
| `NUM_TRAIN_SEEDS` | `1` | Development seeds are `0` through `N-1` for every reward/scenario |
| `NUM_HELD_OUT_SEEDS` | `10` | Held-out seeds are `0` through `N-1`; each policy starts from scratch |
| `SEARCH_REPLICATE` | `0` | Independent reward-search namespace; does not loop automatically |
| `OPENAI_REWARD_MODEL` | `gpt-6.1-sol` | Same model across conditions; account access is required |
| `REWARD_ENGINEER_REASONING_MODE` | `pro` | Registered API reasoning mode |
| `REWARD_ENGINEER_REASONING_EFFORT` | `max` | Registered API reasoning effort |
| `OPENAI_MODEL_VERBOSITY` | `medium` | Registered response verbosity |
| `REWARD_ENGINEER_MAX_TURNS` | `64` | Autonomous designer turn limit |
| `REWARD_ENGINEER_TIMEOUT_SECONDS` | `5400` | Shared active-time budget per candidate, including probes and repairs |
| `LEARNING_PROBE_JOBS` | `2` | Maximum optional full-budget probes per candidate |
| `LEARNING_PROBE_TIMEOUT_SECONDS` | `3600` | Probe worker timeout inside the candidate budget |
| `TRAINING_JOB_RETRIES` | `0` | No automatic failed-training retry |
| `TRAINING_SNAPSHOT_EPISODES` | `2` | Independent raw episodes per intermediate checkpoint |
| `REQUIRE_GPU` | `1` | Require CUDA training for registered studies |

The exact family learner and reward settings are in `config.py`; `.env.example`
lists the public configuration defaults. A changed model, training budget,
evaluation rule, or evidence allowance changes the experimental configuration.
Keep these settings matched across all conditions you compare.

Increasing policy seeds does not multiply reward generation. Choose additional
`SEARCH_REPLICATE` values before inspecting results and start a separate suite
for each. A default run with ten held-out policy seeds (0 through 9) estimates
policy-training variation for one selected reward, not repeatability across ten
independent language-model searches.

## Resume and inspect state

For an interrupted suite:

```bash
python main.py --resume latest
```

To choose an exact suite, replace `SUITE_ID` with its actual directory name:

```bash
python main.py --resume results/suites/SUITE_ID
```

Completed stages and compatible completed training jobs are reused. Saved model
responses and probe budgets remain associated with the same candidates. An
interrupted training job restarts from scratch; resume is not an arbitrary
optimizer-step checkpoint restart. An exhausted candidate budget is not reset.

The controller checks source, dependencies, registered settings, reference
bindings, and frozen hashes. If compatibility fails, inspect the recorded error
instead of editing manifests or forcing reuse. `--new` intentionally starts new
searches and is not a way to repair an interrupted suite.

State can also be read without starting training:

```bash
python main.py status
```

This reads the latest suite without launching work. Exact suite state is in
`results/suites/SUITE_ID/suite.json`; each stage has a log in the same directory.
Inspect each run's `state.json`, `events.jsonl`, and the stage log when execution
stops. `main.py` manages the complete ordering: no separate condition commands
are needed.

## Generated outputs

Execution creates output directories as needed:

```text
results/
  suites/<suite_id>/suite.json
  exp1/<family>/replicate_<n>/<run_id>/
  exp2/smax/replicate_<n>/<run_id>/
  exp3/smax/replicate_<n>/<run_id>/
```

Within each run:

| Path | Contents |
|---|---|
| `manifest.json` | Registered protocol, resolved dependencies, accelerator, and source hashes |
| `registered/` | Exact allowlisted source, prompts, and protocol snapshot |
| `state.json`, `events.jsonl` | Current state and chronological events |
| `family/candidates/Gen*/` | Reward source, design report, validation, and development scores |
| `family/development/` | Native/candidate policy-training jobs and learning evidence |
| `family/selection.json` | Complete mechanical ranking and selected reward hash |
| `family/frozen_reward.py` | Exact selected reward program |
| `family/held_out/` | Fresh held-out policy-training jobs |
| `family/held_out_table.json` | Per-seed evaluations and aggregate comparisons |
| `family/family_summary.json` | Final results, model usage, and training counts |
| `learning_probes/` | Candidate probe ledgers and retained probe artifacts |

Each completed policy-training job retains `job.json`, `result.json`,
`training_curve.csv`, `evaluation_episodes.csv`, `checkpoint/`, raw trajectories
with a readable schema, and final decoded traces. `training_snapshots/` contains
intermediate checkpoints and independent raw episodes at the first existing
learner chunks reaching 20% and 60% of the budget. Actual capture positions are
recorded rather than assumed to be exactly those percentages.

GARDEN additionally records candidate workspaces, model requests, tool outputs,
and session records. Controls retain their fixed source packet and structured
draft records. These records contain model outputs and supplied reasoning
summaries, not access to hidden internal reasoning.

Do not remove raw development artifacts to save space during a search: they are
part of the permitted evidence for later generations and of reuse/provenance
checks. Keep the GARDEN/native SMAX reference available until dependent controls
finish. Archive complete run and suite directories together with their exact
source and environment records; a single frozen reward file is not a complete
experimental record.

## Reading results

Compare sparse task returns, not the magnitude of shaped training rewards.
Overcooked return is 20 per delivered soup; an average need not be a multiple of
20. SMAX sparse return is terminal win rate in `[0, 1]`. Final per-policy
evaluation uses 64 episodes for Overcooked and 256 for SMAX by default, followed
by aggregation over held-out training seeds.

The selected reward is one search outcome. Development selection is not an
independent validation estimate, and multiple policy seeds are not multiple
reward-search replicates. The held-out boundary limits designer access; it does
not establish researcher-blind scenario selection. Record realized compute and
all valid outcomes, including unfavorable ones, when interpreting comparisons.
