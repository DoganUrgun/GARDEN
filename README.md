# GARDEN

Generalist Autonomous Reward Design through Evidence Navigation.

GARDEN uses an autonomous language-model agent to investigate learning evidence
and write one standalone shaping-reward program for a cooperative environment
family. Six candidate programs are assessed on three development scenarios. The
best program is selected mechanically, frozen, and evaluated by training fresh
policies on two held-out scenarios with the same reward source.

This is **reward transfer**, not zero-shot transfer of a trained policy. Policies
are trained separately for every reward, scenario, and training seed; the reward
search is family-level, not a separate search for each scenario.

## What is run

One entry point, `main.py`, runs the two paper comparisons:

| Comparison | Question | Families |
|---|---|---|
| GARDEN versus native shaping | Does a family-level generated reward transfer to held-out scenarios? | Overcooked V1 and SMAX |
| GARDEN versus restricted reward designers | What is gained from autonomous evidence navigation? | SMAX |

The second comparison uses two controls: **GARDEN-Fixed**, with prescribed
diagnostic reports and no shell or arbitrary evidence inspection, and a
**Eureka-inspired** best-parent/component-feedback search. The latter is a
budget-matched adaptation, not a reproduction of the original Eureka paper.

There are therefore **three design conditions**, not three separate paper
comparisons. GARDEN is run for both families, and each control is run for SMAX:
four searches in total, each producing six candidate rewards. The controls train
their own candidates and fresh held-out policies. Compatible native and GARDEN
comparison results are reused, without sharing another condition's search
evidence with their designers.

| Family | Development scenarios | Held-out scenarios | Sparse task objective |
|---|---|---|---|
| Overcooked V1 | `cramped_room`, `bottleneck`, `coord_ring` | `forced_coord`, `counter_circuit` | Episode return: 20 per soup delivered |
| SMAX | `3s5z_vs_3s6z`, `smacv2_5_units`, `5m_vs_6m` | `10m_vs_11m`, `smacv2_10_units` | Terminal win rate |

## Install and run

Use Linux or Ubuntu under WSL2 with an NVIDIA GPU. Follow
[Installation](docs/INSTALL.md) to create a clean Python environment, install
`requirements-gpu.txt`, and configure a private `.env` from `.env.example`.
Training uses CUDA JAX pinned to `0.4.36`. Native Windows is not the CUDA training
target.

From the repository root, with the environment activated:

```bash
python main.py
python main.py doctor
python main.py plan
```

Running `main.py` without arguments shows help and does no work. `doctor` checks
GPU execution; `plan` explains the full study and training budget without starting
a search or calling the model. Neither checks model access. The configured model
is `gpt-6.1-sol`; availability depends on your API account, and there is no
automatic model fallback.

To deliberately start the full study:

```bash
python main.py --new
```

The suite registers four runs and executes seven stages on one GPU. It completes
and freezes all three SMAX searches before any SMAX held-out training, then runs
Overcooked. A full suite can involve up to **246 full-budget policy-training
jobs**, plus model calls and diagnostic work. Inspect the plan and resource
requirements before starting.

After an interruption, resume the same queue without starting new searches:

```bash
python main.py --resume latest
```

Use `python main.py status` to read the latest suite's progress without starting
work. For exact-suite resume and artifact inspection, see
[Running experiments](docs/EXPERIMENTS.md).

## Defaults and reproducibility

Set scientific settings in `.env` before registering a run. Keep them identical
across conditions in a controlled comparison.

| Setting | Default | Meaning |
|---|---|---|
| `NUM_TRAIN_SEEDS` | `1` | Development policy-training seeds, starting at 0 |
| `NUM_HELD_OUT_SEEDS` | `10` | Fresh policy-training seeds per held-out scenario, starting at 0 |
| `SEARCH_REPLICATE` | `0` | Separate reward-search realization and output namespace |
| `LEARNING_PROBE_JOBS` | `2` | Maximum optional full-budget probes per candidate |

Each search produces six valid candidates in three independent A/B generations.
Both families request 5,000,000 transitions per policy job and execute 4,997,120
after rounding down to full rollout batches. Native and generated shaping are
stationary. The family learner, optimizer, training budget, and evaluation rule
are fixed across conditions.

The default suite has one reward-search realization per condition/family and
ten held-out policy seeds (0 through 9). More policy seeds do not repeat the
language-model search. To study independent searches, choose `SEARCH_REPLICATE` values in
advance and run separate suites; this setting does not automatically loop over
replicates.

## Outputs and code map

Run directories are created during execution under
`results/expN/<family>/replicate_<n>/<run_id>/`. Suite state and stage logs are
written under `results/suites/<suite_id>/`. The internal labels `exp1`, `exp2`,
and `exp3` mean GARDEN/native, GARDEN-Fixed, and Eureka-inspired, respectively;
they do not require separate launch commands. Outputs include registered settings
and source hashes, candidate reward/report pairs, selection records, frozen
reward source, per-seed evaluations, training curves, policy checkpoints, raw
trajectories, intermediate snapshots, and model interaction records. Preserve
these artifacts: raw development evidence is part of GARDEN's design interface,
not just a reporting convenience.

Read [PROTOCOL.md](PROTOCOL.md) for the reward contract, evidence restrictions,
selection rule, and frozen transfer evaluation. [Running experiments](docs/EXPERIMENTS.md) explains
job counts, resume behavior, outputs, and interpretation limits.

Start reading at `main.py`: its comments explain the commands and comparisons.
`common/suite.py` schedules the study; the scientific workflow is in
`common/study.py`. Other main modules are `common/mappo.py`
(learner), `common/designer.py` (autonomous designer), and
`common/feedback.py` / `common/structured_designer.py` (restricted controls).
Family adapters and networks are in `Overcooked/` and `SMAX/`; registered
instructions are in `prompts/`. See [runtime provenance](runtime/UPSTREAM.md)
and [third-party notices](THIRD_PARTY_NOTICES.md) for the vendored environments.

## License

The original GARDEN code and documentation are released under the
[MIT License](LICENSE), copyright (c) 2026 GARDEN authors. Bundled third-party
materials retain their own licenses; see
[Third-party notices](THIRD_PARTY_NOTICES.md).
