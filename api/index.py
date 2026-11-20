"""Vercel Serverless Function entrypoint for TruthLens FastAPI service."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Remove any current directory or shadowing paths and put project ROOT at index 0
sys.path = [p for p in sys.path if Path(p).resolve() != Path.cwd()]
sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402
