"""Registered settings shared by the three reward-design conditions.

Environment variables are read once, before a run is registered. A run keeps
its exact settings; changing a scientific setting requires a new run.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

EXPERIMENT = os.getenv("REWARD_DESIGN_EXPERIMENT", "exp1")
if EXPERIMENT not in {"exp1", "exp2", "exp3"}:
    raise ValueError("REWARD_DESIGN_EXPERIMENT must be exp1, exp2, or exp3")
PROMPTS = ROOT / "prompts"
RUNTIME = ROOT / "runtime"
CACHE = ROOT / ".cache"

CANDIDATES = ("Gen1a", "Gen1b", "Gen2a", "Gen2b", "Gen3a", "Gen3b")
GENERATIONS = (("Gen1a", "Gen1b"), ("Gen2a", "Gen2b"), ("Gen3a", "Gen3b"))


def _integer(name: str, default: int, minimum: int = 1) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _number(name: str, default: float, minimum: float = 0.0) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= minimum:
        raise ValueError(f"{name} must be greater than {minimum}")
    return value


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw not in {"0", "false", "no", "off"}


NUM_TRAIN_SEEDS = _integer("NUM_TRAIN_SEEDS", 1)
TRAIN_SEEDS = tuple(range(NUM_TRAIN_SEEDS))
HELD_OUT_SEEDS = tuple(range(_integer("NUM_HELD_OUT_SEEDS", 10)))
SEARCH_REPLICATE = _integer("SEARCH_REPLICATE", 0, minimum=0)


@dataclass(frozen=True)
class LearnerConfig:
    max_steps: int
    num_envs: int
    rollout_steps: int
    ppo_epochs: int
    minibatches: int
    learning_rate: float
    entropy_coef: float
    actor_hidden: int
    critic_hidden: int
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    max_grad_norm: float = 0.25
    learning_rate_schedule: str = "linear"
    warmup_fraction: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FamilyConfig:
    cli_name: str
    name: str
    development: tuple[str, ...]
    held_out: tuple[str, ...]
    timesteps: int
    evaluation_episodes: int
    trace_episodes: int
    reward_weight: float
    learner: LearnerConfig

    def public_dict(self) -> dict[str, Any]:
        return {
            "cli_name": self.cli_name,
            "name": self.name,
            "development_scenarios": list(self.development),
            "timesteps": self.timesteps,
            "evaluation_episodes": self.evaluation_episodes,
            "trace_episodes": self.trace_episodes,
            "reward_weight": self.reward_weight,
            "learner": self.learner.to_dict(),
        }

    def private_dict(self) -> dict[str, Any]:
        value = self.public_dict()
        value["held_out_scenarios"] = list(self.held_out)
        return value


OVERCOOKED = FamilyConfig(
    cli_name="overcooked",
    name="Overcooked",
    development=("cramped_room", "bottleneck", "coord_ring"),
    held_out=("forced_coord", "counter_circuit"),
    # The official classic Overcooked CNN PPO reference uses five million steps.
    timesteps=_integer("OVERCOOKED_TIMESTEPS", 5_000_000),
    evaluation_episodes=_integer("OVERCOOKED_EVAL_EPISODES", 64),
    trace_episodes=_integer("OVERCOOKED_TRACE_EPISODES", 8),
    reward_weight=_number("OVERCOOKED_REWARD_WEIGHT", 0.10),
    learner=LearnerConfig(
        max_steps=400, num_envs=64, rollout_steps=256, ppo_epochs=4,
        minibatches=16, learning_rate=5e-4, entropy_coef=0.01,
        actor_hidden=64, critic_hidden=64, max_grad_norm=0.5,
        learning_rate_schedule="linear", warmup_fraction=0.0,
    ),
)

SMAX = FamilyConfig(
    cli_name="smax",
    name="SMAX",
    development=("3s5z_vs_3s6z", "smacv2_5_units", "5m_vs_6m"),
    held_out=("10m_vs_11m", "smacv2_10_units"),
    # The same training budget is used for all conditions and held-out jobs.
    # Full rollout batches produce 305 updates / 4,997,120 environment steps.
    timesteps=_integer("SMAX_TIMESTEPS", 5_000_000),
    evaluation_episodes=_integer("SMAX_EVAL_EPISODES", 256),
    trace_episodes=_integer("SMAX_TRACE_EPISODES", 16),
    reward_weight=_number("SMAX_REWARD_WEIGHT", 0.01),
    learner=LearnerConfig(
        max_steps=100, num_envs=128, rollout_steps=128, ppo_epochs=4,
        minibatches=4, learning_rate=2e-3, entropy_coef=0.0,
        actor_hidden=128, critic_hidden=128,
        learning_rate_schedule="linear", warmup_fraction=0.0,
    ),
)

FAMILIES = (OVERCOOKED, SMAX)
FAMILY_BY_NAME = {family.name: family for family in FAMILIES}
FAMILY_BY_CLI = {family.cli_name: family for family in FAMILIES}

MAX_REWARD_SOURCE_LINES = _integer("MAX_REWARD_SOURCE_LINES", 500)
MAX_REWARD_AST_NODES = _integer("MAX_REWARD_AST_NODES", 8000)
MAX_MEMORY_ELEMENTS = _integer("MAX_REWARD_MEMORY_ELEMENTS", 100_000)

MODEL = os.getenv("OPENAI_REWARD_MODEL", "gpt-6.1-sol").strip()
MODEL_VERBOSITY = os.getenv("OPENAI_MODEL_VERBOSITY", "medium").strip()
REASONING_EFFORT = os.getenv("REWARD_ENGINEER_REASONING_EFFORT", "max").strip()
REASONING_MODE = os.getenv("REWARD_ENGINEER_REASONING_MODE", "pro").strip()
MAX_AGENT_TURNS = _integer("REWARD_ENGINEER_MAX_TURNS", 64)
AGENT_TIMEOUT_SECONDS = _number("REWARD_ENGINEER_TIMEOUT_SECONDS", 5400)
SHELL_TIMEOUT_SECONDS = _number("REWARD_ENGINEER_SHELL_TIMEOUT_SECONDS", 300)
MAX_DESIGN_REPAIRS = _integer("MAX_DESIGN_REPAIRS", 2, minimum=0)
# Equal optional training allowances in all conditions. A probe is a complete
# development training job, not permission to change the learner or its budget.
LEARNING_PROBE_JOBS = _integer("LEARNING_PROBE_JOBS", 2, minimum=0)
LEARNING_PROBE_TIMEOUT_SECONDS = _integer("LEARNING_PROBE_TIMEOUT_SECONDS", 3600)

EVALUATION_GREEDY = _flag("EVALUATION_GREEDY", False)
# Observational evidence only: capture after existing learner chunks without
# altering training randomness, update boundaries, or the selection score.
SNAPSHOT_FRACTIONS = (0.2, 0.6)
SNAPSHOT_EPISODES = _integer("TRAINING_SNAPSHOT_EPISODES", 2, minimum=0)

TRAINING_JOB_TIMEOUT_SECONDS = _integer("TRAINING_JOB_TIMEOUT_SECONDS", 86_400)
TRAINING_JOB_RETRIES = _integer("TRAINING_JOB_RETRIES", 0, minimum=0)

REQUIRE_GPU = _flag("REQUIRE_GPU", True)
TRAINING_JAX_PLATFORM = "cuda" if REQUIRE_GPU else "cpu"
JAX_CACHE_DIR = Path(os.getenv("JAX_COMPILATION_CACHE_DIR", str(CACHE / "jax"))).resolve()


def protocol(family: FamilyConfig) -> dict[str, Any]:
    """Return the immutable protocol for one independently runnable family."""
    return {
        "schema_version": 2,
        "study": "GARDEN probe-enabled controlled study",
        "experiment": EXPERIMENT,
        "search_replicate": SEARCH_REPLICATE,
        "family": family.private_dict(),
        "candidate_schedule": [list(pair) for pair in GENERATIONS],
        "train_seeds": list(TRAIN_SEEDS),
        "held_out_seeds": list(HELD_OUT_SEEDS),
        "training_evidence": {
            "snapshot_target_fractions": list(SNAPSHOT_FRACTIONS),
            "snapshot_episodes": SNAPSHOT_EPISODES,
            "capture_boundary": "first existing learner chunk reaching each target",
            "training_rng_independent": True,
        },
        "learning_probes": {
            "maximum_jobs_per_candidate": LEARNING_PROBE_JOBS,
            "maximum_jobs_per_search": LEARNING_PROBE_JOBS * len(CANDIDATES),
            "optional": True,
            "training_seed": TRAIN_SEEDS[0],
            "requested_timesteps_per_job": family.timesteps,
            "timeout_seconds": LEARNING_PROBE_TIMEOUT_SECONDS,
            "failed_launched_jobs_consume_budget": True,
            "automatic_retries": 0,
            "runtime_counts_toward_design_budget": True,
            "exact_match_final_development_reuse": True,
            "reuse_scope": "same candidate, reward hash, scenario, seed and complete training/evaluation configuration",
            "feedback": "raw artifacts" if EXPERIMENT == "exp1" else "fixed diagnostics" if EXPERIMENT == "exp2" else "task and component reflection",
        },
        "selection": {
            "primary": "mean normalized sparse improvement relative to native",
            "formula": "mean_i((J_i(R)-J_i(B))/max(abs(J_i(B)),1))",
            "tie_breaks": [
                "minimum scenario normalized improvement",
                "mean development sparse return",
                "candidate identifier",
            ],
        },
        "reward_modes": {
            "native": "task + native_shaping",
            "candidate": "task + family_weight * standalone_llm_reward",
            "native_available_to_candidate_runtime": False,
        },
        "candidate_uniqueness": {
            "exact_source_duplicates_allowed": False,
            "normalized_program_duplicates_allowed": False,
            "same_generation_b_sees_a_design_but_not_results": False,
            "same_generation_candidates_are_independent": True,
        },
        "held_out": {
            "visible_to_designer": False,
            "native_comparison": True,
            "new_policies": True,
            "llm_calls_after_lock": False,
            "suite_requires_all_smax_conditions_frozen": True,
        },
        "model": {
            "identifier": MODEL,
            "reasoning_mode": REASONING_MODE,
            "reasoning_effort": REASONING_EFFORT,
            "verbosity": MODEL_VERBOSITY,
            "prompt_directory": "prompts",
            "maximum_turns": MAX_AGENT_TURNS,
            "timeout_seconds": AGENT_TIMEOUT_SECONDS,
            "shell_timeout_seconds": SHELL_TIMEOUT_SECONDS,
            "maximum_repairs": MAX_DESIGN_REPAIRS,
            "shared_candidate_time_budget": True,
            "reserved_finalization_seconds": min(300.0, AGENT_TIMEOUT_SECONDS * 0.1),
            "completion": "explicit completed proposal with reward/report hashes recorded; timeout alone never commits drafts",
        },
        "evaluation": {
            "greedy_actions": EVALUATION_GREEDY,
            "new_policy_per_scenario_seed": True,
        },
        "execution": {
            "gpu_required": REQUIRE_GPU,
            "training_workers": 1,
            "training_backend": TRAINING_JAX_PLATFORM,
            "controller_validation_backend": "cpu",
            "persistent_compilation_cache": True,
        },
        "reward_validation": {
            "maximum_source_lines": MAX_REWARD_SOURCE_LINES,
            "maximum_ast_nodes": MAX_REWARD_AST_NODES,
            "maximum_memory_elements": MAX_MEMORY_ELEMENTS,
        },
        "controller": {
            "training_job_timeout_seconds": TRAINING_JOB_TIMEOUT_SECONDS,
            "training_job_retries": TRAINING_JOB_RETRIES,
        },
    }


for family in FAMILIES:
    if len(family.development) != 3 or len(family.held_out) != 2:
        raise ValueError(f"{family.name} requires exactly three development and two held-out scenarios")
    if set(family.development) & set(family.held_out):
        raise ValueError(f"{family.name} development and held-out scenarios overlap")
    batch = family.learner.num_envs * family.learner.rollout_steps
    if family.timesteps < batch:
        raise ValueError(f"{family.name} training budget is shorter than one rollout")
    if family.learner.num_envs % family.learner.minibatches:
        raise ValueError(f"{family.name} num_envs must be divisible by minibatches")
    if family.learner.learning_rate_schedule not in {"constant", "linear", "warmup_cosine"}:
        raise ValueError(f"Unsupported learning-rate schedule for {family.name}")
