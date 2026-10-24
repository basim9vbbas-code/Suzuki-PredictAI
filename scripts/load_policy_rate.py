"""Validate data/raw/sbp_policy_rate.csv and load it into the SQLite database.

Run from the project root:
    python scripts\\load_policy_rate.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.market_intelligence import db  # noqa: E402
from src.market_intelligence.analysis import policy_rate_monthly  # noqa: E402
from src.market_intelligence.importers.sbp_policy_rate import (  # noqa: E402
    SERIES_ID,
    DataValidationError,
    load_policy_rate,
)

CSV_PATH = ROOT / "data" / "raw" / "sbp_policy_rate.csv"


def main() -> int:
    conn = db.connect()
    try:
        n = load_policy_rate(conn, CSV_PATH)
    except DataValidationError as exc:
        print(f"[FAIL] Validation error: {exc}")
        return 1

    changes = db.read_observations(conn, SERIES_ID)
    monthly = policy_rate_monthly(changes, end_month=pd.Timestamp.today())
    print(f"[ OK ] Stored {n} rate changes in {db.DEFAULT_DB_PATH}")
    print(f"       First change : {changes['date'].min().date()}  "
          f"({changes['value'].iloc[0]}%)")
    print(f"       Latest change: {changes['date'].max().date()}  "
          f"({changes['value'].iloc[-1]}%)")
    print(f"       Monthly rows : {len(monthly)} "
          f"({monthly['month_end'].min().date()} to {monthly['month_end'].max().date()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
