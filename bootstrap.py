"""Make the project and its minimal vendored JaxMARL runtime importable."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"

for path in (ROOT, RUNTIME):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
