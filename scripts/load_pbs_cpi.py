"""Validate a PBS 'Monthly Review on Price Indices' PDF and load it into SQLite.

Run from the project root (default file shown):
    python scripts\\load_pbs_cpi.py
    python scripts\\load_pbs_cpi.py data\\raw\\pbs_cpi_review_2026_09.pdf

Each run ADDS the months in that PDF to what is already stored.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.market_intelligence import db  # noqa: E402
from src.market_intelligence.importers.pbs_cpi import load_pbs_review  # noqa: E402
from src.market_intelligence.importers.sbp_policy_rate import DataValidationError  # noqa: E402

DEFAULT_PDF = ROOT / "data" / "raw" / "pbs_cpi_review_2026_08.pdf"


def main() -> int:
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF
    conn = db.connect()
    try:
        review = load_pbs_review(conn, pdf)
    except DataValidationError as exc:
        print(f"[FAIL] Validation error: {exc}")
        return 1

    d = review.data
    n_months = d["obs_date"].nunique()
    print(f"[ OK ] Loaded {len(d)} values from {pdf.name}")
    print(f"       Review month : {review.review_month:%B %Y}")
    print(f"       Months       : {n_months} ({d['obs_date'].min():%Y-%m} to {d['obs_date'].max():%Y-%m})")
    print(f"       National CPI index: {review.national_index['current']} "
          f"(previous month {review.national_index['previous']}, year ago {review.national_index['year_ago']})")
    total = conn.execute(
        "SELECT COUNT(DISTINCT obs_date) FROM indicator_observations "
        "WHERE series_id = 'pbs_cpi_national_yoy'"
    ).fetchone()[0]
    print(f"       Months now stored for national CPI: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
