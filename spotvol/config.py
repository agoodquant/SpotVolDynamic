from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "spotvol.duckdb"
REPORTS = ROOT / "reports"

SOURCES = ["yahoo", "dolthub"]   # preference order when both cover the same day
TENORS = {"1m": 30, "3m": 91, "6m": 182, "1y": 365, "2y": 730}
FALLBACK_RATE = 0.04


def universe() -> dict:
    with open(ROOT / "universe.yaml") as f:
        u = yaml.safe_load(f)
    u["all"] = list(u["symbols"]) + list(u.get("benchmarks", []))
    return u
