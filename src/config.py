"""Central configuration and logging for ClaimIQ."""

from __future__ import annotations

import logging
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
POLICY_DIR = ROOT / "docs" / "policies"
VECTOR_DIR = ARTIFACTS_DIR / "vector_store"
LOG_LEVEL = os.getenv("CLAIMIQ_LOG_LEVEL", "INFO").upper()


def get_logger(name: str) -> logging.Logger:
    """Return a consistently configured project logger."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return logging.getLogger(name)
