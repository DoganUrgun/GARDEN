"""Resolve family-specific adapters without dynamic experimental behavior."""

from __future__ import annotations

import config


def adapter_for(name: str, family_config: config.FamilyConfig | None = None):
    if name == "Overcooked":
        from Overcooked.adapter import Adapter

        return Adapter(family_config or config.OVERCOOKED)
    if name == "SMAX":
        from SMAX.adapter import Adapter

        return Adapter(family_config or config.SMAX)
    raise ValueError(f"Unknown environment family: {name}")
