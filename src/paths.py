from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_FEATURES = ROOT / "data" / "features"
PREDICTIONS = ROOT / "artifacts" / "predictions"
