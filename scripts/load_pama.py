"""Validate data/raw/pama_history_2007_2026.xlsx and load it into SQLite.

Run from the project root:
    python scripts\\load_pama.py
"""

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.market_intelligence import db  # noqa: E402
from src.market_intelligence.importers.pama_history import load_pama  # noqa: E402
from src.market_intelligence.importers.sbp_policy_rate import DataValidationError  # noqa: E402

XLSX_PATH = ROOT / "data" / "raw" / "pama_history_2007_2026.xlsx"


def main() -> int:
    warnings.filterwarnings("ignore", message="Cannot parse header or footer")
    conn = db.connect()
    try:
        result = load_pama(conn, XLSX_PATH)
    except DataValidationError as exc:
        print(f"[FAIL] Validation error: {exc}")
        return 1

    df = result.data
    print(f"[ OK ] Stored {len(df)} rows in {db.DEFAULT_DB_PATH}")
    print(f"       Months covered : {df['month'].min()} to {df['month'].max()}")
    print(f"       Fiscal sheets  : {df['fiscal_year'].nunique()}")
    print(f"       Distinct models: {df['model'].nunique()}")
    if result.warnings:
        print(f"[WARN] {len(result.warnings)} cumulative-total mismatches in the source file:")
        for w in result.warnings[:10]:
            print("       -", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
