"""Standalone generated-reward contract with no native-shaping channel."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

import config
from common.records import file_hash


class CandidateComponents(NamedTuple):
    full: jax.Array
    task: jax.Array
    llm: jax.Array
    raw_llm: jax.Array


@dataclass(frozen=True)
class LoadedReward:
    family: str
    path: Path
    digest: str
    source: str
    initial_memory: Callable[[int], Any]
    reward_and_memory: Callable[[Any, jax.Array, Any, jax.Array, Any], tuple[Any, Any]]
    # Optional additive, raw (before lambda) diagnostic terms. Never a payoff input.
    reward_components: Callable | None = None


def _dotted_name(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _literal(node: ast.AST) -> bool:
    try:
        value = ast.literal_eval(node)
    except (TypeError, ValueError):
        return False
    if isinstance(value, (str, int, float, bool, type(None))):
        return True
    return isinstance(value, tuple) and all(
        isinstance(item, (str, int, float, bool, type(None))) for item in value
    )


def normalized_program_hash(source: str) -> str:
    """Hash executable Python structure while ignoring comments and docstrings."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            del body[0]
    canonical = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_source(source: str, family: str) -> ast.Module:
    if family not in config.FAMILY_BY_NAME:
        raise ValueError(f"Unknown reward family: {family}")
    if len(source.splitlines()) > config.MAX_REWARD_SOURCE_LINES:
        raise ValueError("reward.py exceeds the registered source-line limit")
    tree = ast.parse(source)
    if sum(1 for _ in ast.walk(tree)) > config.MAX_REWARD_AST_NODES:
        raise ValueError("reward.py exceeds the registered AST limit")

    approved = {"jax", "jax.numpy"}
    if family == "Overcooked":
        approved.add("jaxmarl.environments.overcooked.common")
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    imported: set[str] = set()
    import_names: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if isinstance(node, ast.Import):
            if any(alias.name not in approved for alias in node.names):
                raise ValueError("Generated rewards may import only JAX and approved value enums")
            imported.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
            for alias in node.names:
                import_names[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
            continue
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module not in approved:
                raise ValueError("Generated rewards may not import environment reward implementations")
            if any(alias.name == "*" for alias in node.names):
                raise ValueError("Wildcard imports are not allowed")
            imported.update(alias.asname or alias.name for alias in node.names)
            for alias in node.names:
                import_names[alias.asname or alias.name] = f"{node.module}.{alias.name}"
            continue
        if isinstance(node, ast.FunctionDef) and not node.decorator_list:
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if value is not None and _literal(value) and all(isinstance(item, ast.Name) for item in targets):
                continue
        raise ValueError("Top level may contain only imports, immutable constants, docstrings, and functions")

    forbidden = (
        ast.ClassDef, ast.AsyncFunctionDef, ast.Lambda, ast.AsyncFor, ast.While,
        ast.Try, ast.With, ast.AsyncWith, ast.Raise, ast.Global, ast.Nonlocal,
        ast.Yield, ast.YieldFrom, ast.Await, ast.NamedExpr,
    )
    safe_builtins = {
        "abs", "all", "any", "bool", "enumerate", "float", "int", "len",
        "max", "min", "range", "round", "sum", "tuple", "zip",
    }
    top_level_nodes = set(tree.body)
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            raise ValueError(f"Unsupported reward construct: {type(node).__name__}")
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node not in top_level_nodes:
            raise ValueError("Imports must be declared at the top level")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Private and dunder attributes are unavailable")
        # JAX NumPy re-exports file I/O. Approved numerical imports must not
        # become a backdoor to files, artifacts, or held-out information.
        if isinstance(node, ast.Attribute) and node.attr in {
            "load", "save", "savez", "savez_compressed", "fromfile", "tofile", "memmap"
        }:
            raise ValueError("Generated rewards may not read or write files")
        if isinstance(node, ast.Call):
            dotted = _dotted_name(node.func)
            if dotted:
                root, separator, suffix = dotted.partition(".")
                dotted = import_names.get(root, root) + (separator + suffix if separator else "")
                if (
                    dotted.startswith("jax.debug")
                    or dotted.startswith("jax.experimental.io_callback")
                    or dotted == "jax.pure_callback"
                ):
                    raise ValueError("Callbacks and debug side effects are not allowed")
                if dotted.startswith("jax.numpy.") and dotted.rsplit(".", 1)[-1] in {
                    "load", "save", "savez", "savez_compressed", "fromfile", "tofile", "memmap"
                }:
                    raise ValueError("Generated rewards may not read or write files")
            if isinstance(node.func, ast.Name):
                if node.func.id not in functions | imported | safe_builtins:
                    raise ValueError(f"Unsupported function call: {node.func.id}")
            elif not isinstance(node.func, ast.Attribute):
                raise ValueError("Unsupported callable expression")

    expected = {
        "initial_memory": (1, "initial_memory(num_agents)"),
        "reward_and_memory": (5, "reward_and_memory(previous_state, actions, next_state, task_reward, memory)"),
    }
    if "reward_components" in functions:
        expected["reward_components"] = (
            5, "reward_components(previous_state, actions, next_state, task_reward, memory)"
        )
    for name, (count, signature) in expected.items():
        definition = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name), None)
        if definition is None:
            raise ValueError(f"reward.py must define {signature}")
        arguments = definition.args
        if len(arguments.args) != count or arguments.vararg or arguments.kwarg or arguments.kwonlyargs:
            raise ValueError(f"reward.py must define {signature}")
    return tree


def _validate_memory(memory: Any) -> None:
    total = 0
    for leaf in jax.tree_util.tree_leaves(memory):
        value = jnp.asarray(leaf)
        if value.dtype == jnp.dtype("O"):
            raise ValueError("Reward memory must contain only numeric or boolean JAX values")
        total += value.size
    if total > config.MAX_MEMORY_ELEMENTS:
        raise ValueError("Reward memory exceeds the registered element limit")


def load_reward(path: Path, family: str) -> LoadedReward:
    source = path.read_text(encoding="utf-8")
    validate_source(source, family)
    digest = file_hash(path)
    spec = importlib.util.spec_from_file_location(f"generated_{family}_{digest[:16]}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import generated reward: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    initial = getattr(module, "initial_memory", None)
    function = getattr(module, "reward_and_memory", None)
    if not callable(initial) or not callable(function):
        raise TypeError("Generated reward contract is incomplete")
    components = getattr(module, "reward_components", None)
    if components is not None and not callable(components):
        raise TypeError("reward_components must be a function when present")
    return LoadedReward(family, path, digest, source, initial, function, components)


def initial_batched_memory(reward: LoadedReward, batch_size: int, num_agents: int) -> Any:
    memory = reward.initial_memory(num_agents)
    _validate_memory(memory)
    return jax.tree_util.tree_map(
        lambda value: jnp.broadcast_to(jnp.asarray(value), (batch_size,) + jnp.shape(value)), memory
    )


def reset_memory(memory: Any, done: jax.Array, initial: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda current, fresh: jnp.where(
            done.reshape((done.shape[0],) + (1,) * (current.ndim - 1)), fresh, current
        ),
        memory,
        initial,
    )


def candidate_components(reward, weight, previous_state, actions, next_state, task_reward, memory):
    """Compute candidate payoff. Native shaping is absent by construction."""
    raw, next_memory = reward.reward_and_memory(previous_state, actions, next_state, task_reward, memory)
    raw = jnp.asarray(raw, dtype=jnp.float32).reshape((-1,))
    if raw.shape != actions.reshape((-1,)).shape:
        raise ValueError(f"reward_and_memory returned reward shape {raw.shape}; expected {actions.shape}")
    llm = jnp.asarray(weight, dtype=jnp.float32) * raw
    task = jnp.full_like(llm, jnp.asarray(task_reward, dtype=jnp.float32).reshape(()))
    return CandidateComponents(task + llm, task, llm, raw), next_memory


def component_values(reward, previous_state, actions, next_state, task_reward, memory):
    """Return optional raw reward terms for logging, never for computing payoff.

    Names and shapes must be static for JAX. The transition validator additionally
    checks finiteness and that the terms sum to reward_and_memory's raw output.
    """
    if reward.reward_components is None:
        return {}
    terms = reward.reward_components(previous_state, actions, next_state, task_reward, memory)
    if not isinstance(terms, dict) or not terms:
        raise ValueError("reward_components must return a nonempty dict")
    result = {}
    for name, value in terms.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,47}", name):
            raise ValueError("Component names must be short identifiers starting with a letter")
        value = jnp.asarray(value, dtype=jnp.float32)
        if value.shape != actions.shape:
            raise ValueError(f"Component {name} has shape {value.shape}; expected {actions.shape}")
        result[name] = value
    return result


def native_payoff(task_reward: jax.Array, native_shaping: jax.Array) -> jax.Array:
    task = jnp.asarray(task_reward, dtype=jnp.float32).reshape(())
    shaping = jnp.asarray(native_shaping, dtype=jnp.float32).reshape((-1,))
    return jnp.full_like(shaping, task) + shaping
