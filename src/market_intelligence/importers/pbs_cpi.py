"""Importer for the PBS 'Monthly Review on Price Indices' PDF.

Reads Table 1.a (general inflation, base 2015-16): year-on-year and
month-on-month percentage changes for national, urban and rural CPI, the food
and non-food groups, SPI and WPI, over roughly the last 22 months.

Published figures are rounded to one decimal. They are stored as published.
Each monthly review is loaded on top of earlier ones (rows are upserted), so
loading a new month's PDF extends the history instead of replacing it.
"""

from __future__ import annotations

import calendar
import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from .sbp_policy_rate import DataValidationError

# Column order of Table 1.a after the period label: (group, YoY, MoM) pairs.
GROUPS = [
    ("cpi_national", "CPI national"),
    ("cpi_urban", "CPI urban"),
    ("cpi_rural", "CPI rural"),
    ("food_urban", "Food urban"),
    ("food_rural", "Food rural"),
    ("nonfood_urban", "Non-food urban"),
    ("nonfood_rural", "Non-food rural"),
    ("spi", "Sensitive Price Indicator (quintile 1)"),
    ("wpi", "Wholesale Price Index"),
]
NUMBER = r"-?\d+(?:\.\d+)?"
_ROW = re.compile(
    r"^\s*(Jan|Feb|Mar|Apr|May|June?|July?|Aug|Sept?|Oct|Nov|Dec)-(\d{2})\s+"
    rf"((?:{NUMBER}\s+){{17}}{NUMBER})\s*$"
)
_INDEX_ROW = re.compile(rf"^\s*General\s+100\.00\s+({NUMBER})\s+({NUMBER})\s+({NUMBER})\s+({NUMBER})\s+({NUMBER})")
_TITLE = re.compile(r"^\s*(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\s*$")
_MONTH_NUM = {m.lower()[:3]: i for i, m in enumerate(calendar.month_name) if m}

MIN_PCT, MAX_PCT = -30.0, 80.0
CONSISTENCY_TOLERANCE_PP = 0.15  # published values are rounded to 0.1


def series_id(group: str, kind: str) -> str:
    return f"pbs_{group}_{kind}"  # kind: yoy | mom


@dataclass
class PbsReview:
    data: pd.DataFrame  # series_id, obs_date (month end), value
    review_month: pd.Timestamp  # month the review covers (month end)
    national_index: dict[str, float]  # {'current','previous','year_ago'}


def _month_end(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=month, day=calendar.monthrange(year, month)[1])


def _read_text(path: Path) -> list[str]:
    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # noqa: BLE001 - pypdf raises many types
        raise DataValidationError(f"Cannot open PDF {path.name}: {exc}") from exc
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).splitlines()


def read_pbs_review(path: str | Path) -> PbsReview:
    path = Path(path)
    if not path.exists():
        raise DataValidationError(f"File not found: {path}")
    lines = _read_text(path)

    # Which month does this review cover? (title page: 'August 2026')
    title = next((m for l in lines if (m := _TITLE.match(l))), None)
    if not title:
        raise DataValidationError("Could not find the review month on the title page")
    review_month = _month_end(int(title.group(2)), _MONTH_NUM[title.group(1).lower()[:3]])

    # Table 1.a rows: 'Aug-26 11.1 1.2 ...' (18 numbers).
    rows = []
    for line in lines:
        m = _ROW.match(line)
        if not m:
            continue
        mon = _MONTH_NUM[m.group(1).lower()[:3]]
        year = 2000 + int(m.group(2))
        rows.append((_month_end(year, mon), [float(x) for x in m.group(3).split()]))
    if len(rows) < 12:
        raise DataValidationError(
            f"Found only {len(rows)} Table 1.a rows; the PDF layout may have changed"
        )

    dates = [d for d, _ in rows]
    if len(set(dates)) != len(dates):
        raise DataValidationError("Duplicate months in Table 1.a")
    if dates != sorted(dates):
        raise DataValidationError("Table 1.a months are not in chronological order")
    expected = pd.date_range(dates[0], dates[-1], freq="ME")
    if list(expected) != dates:
        missing = sorted(set(expected) - set(dates))
        raise DataValidationError(f"Gaps in Table 1.a months: {[d.strftime('%Y-%m') for d in missing]}")
    if dates[-1] != review_month:
        raise DataValidationError(
            f"Last table month {dates[-1]:%Y-%m} does not match the review month {review_month:%Y-%m}"
        )

    records = []
    for d, nums in rows:
        for i, (group, _) in enumerate(GROUPS):
            for kind, value in (("yoy", nums[2 * i]), ("mom", nums[2 * i + 1])):
                if not MIN_PCT <= value <= MAX_PCT:
                    raise DataValidationError(
                        f"{group} {kind} {d:%Y-%m} = {value} is outside {MIN_PCT}..{MAX_PCT}"
                    )
                records.append((series_id(group, kind), d, value))
    df = pd.DataFrame(records, columns=["series_id", "obs_date", "value"])

    # Independent cross-check: national index levels (Table 1) must agree with
    # the published national YoY/MoM for the review month.
    idx = next((m for l in lines if (m := _INDEX_ROW.match(l))), None)
    if not idx:
        raise DataValidationError("Could not find the national CPI index row (Table 1)")
    current, previous, year_ago = (float(idx.group(i)) for i in (1, 2, 3))
    yoy_calc = (current / year_ago - 1) * 100
    mom_calc = (current / previous - 1) * 100
    last = df[df["obs_date"] == review_month].set_index("series_id")["value"]
    for kind, calc in (("yoy", yoy_calc), ("mom", mom_calc)):
        published = last[series_id("cpi_national", kind)]
        if abs(published - calc) > CONSISTENCY_TOLERANCE_PP:
            raise DataValidationError(
                f"National CPI {kind}: Table 1.a says {published}, but the index "
                f"levels imply {calc:.2f}. The table may have been misread."
            )

    return PbsReview(
        data=df,
        review_month=review_month,
        national_index={"current": current, "previous": previous, "year_ago": year_ago},
    )


def load_pbs_review(conn: sqlite3.Connection, path: str | Path) -> PbsReview:
    """Validate the PDF and upsert its rows. Earlier months are kept."""
    path = Path(path)
    review = read_pbs_review(path)  # raises before touching the database

    with conn:
        for group, label in GROUPS:
            for kind, kind_label in (("yoy", "year-on-year"), ("mom", "month-on-month")):
                conn.execute(
                    "INSERT OR REPLACE INTO indicator_series "
                    "(series_id, name, unit, source, frequency, description) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        series_id(group, kind),
                        f"{label}, {kind_label} change",
                        "percent",
                        "Pakistan Bureau of Statistics, Monthly Review on Price Indices",
                        "monthly",
                        "Base 2015-16. Published to one decimal place.",
                    ),
                )
        conn.executemany(
            "INSERT OR REPLACE INTO indicator_observations (series_id, obs_date, value) "
            "VALUES (?, ?, ?)",
            [
                (r.series_id, r.obs_date.strftime("%Y-%m-%d"), r.value)
                for r in review.data.itertuples(index=False)
            ],
        )
        conn.execute(
            "INSERT INTO import_log "
            "(series_id, source_file, file_sha256, row_count, imported_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                series_id("cpi_national", "yoy"),
                path.name,
                hashlib.sha256(path.read_bytes()).hexdigest(),
                len(review.data),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
    return review
