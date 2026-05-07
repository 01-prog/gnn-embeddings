"""Ensure the project root is on sys.path so ``src.*`` imports work when
pytest is invoked as ``pytest`` (without ``-m``)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
