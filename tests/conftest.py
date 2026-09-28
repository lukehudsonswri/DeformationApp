"""Pytest configuration: make the repo root importable as ``core``, etc.

Individual test files should NOT need their own ``sys.path.insert`` --
this fixture-free conftest handles it once for the whole test session.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
