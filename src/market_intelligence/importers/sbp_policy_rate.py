"""Importer for the SBP policy-rate CSV downloaded from SBP EasyData.

The file records rate CHANGES only (one row per change), not one value per
month. Validation is strict: any problem raises DataValidationError, so bad
data never reaches the database silently.
"""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SERIES_ID = "sbp_policy_rate"
REQUIRED_COLUMNS = [
    "Series Key",
    "Series",
    "Observation Date",
    "Observation Value",
    "Unit",
    "Observation Status",
]
DATE_FORMAT = "%d-%b-%Y"  # e.g. 28-Apr-2026
MIN_RATE, MAX_RATE = 0.0, 50.0  # plausibility bounds, in percent


class DataValidationError(ValueError):
    """Raised when an input file fails validation."""


@dataclass
class PolicyRateFile:
    data: pd.DataFrame  # columns: date, value (sorted oldest to newest)
    series_key: str
    series_name: str
    unit: str


def read_policy_rate_csv(path: str | Path) -> PolicyRateFile:
    path = Path(path)
    if not path.exists():
        raise DataValidationError(f"File not found: {path}")

    raw = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise DataValidationError(f"Missing columns: {missing}")
    if raw.empty:
        raise DataValidationError("File has no rows")

    keys = raw["Series Key"].dropna().unique()
    if len(keys) != 1:
        raise DataValidationError(
            f"Expected exactly one series, found {len(keys)}: {list(keys)}"
        )

    bad_status = raw.loc[raw["Observation Status"] != "Normal", "Observation Status"]
    if not bad_status.empty:
        raise DataValidationError(
            f"Rows with non-'Normal' status: {sorted(bad_status.unique())}"
        )

    units = raw["Unit"].dropna().unique()
    if len(units) != 1 or str(units[0]).lower() != "percent":
        raise DataValidationError(f"Expected unit 'Percent', found {list(units)}")

    dates = pd.to_datetime(
        raw["Observation Date"], format=DATE_FORMAT, errors="coerce"
    )
    if dates.isna().any():
        bad = raw.loc[dates.isna(), "Observation Date"].tolist()
        raise DataValidationError(f"Unparseable dates (expected DD-Mon-YYYY): {bad}")

    values = pd.to_numeric(raw["Observation Value"], errors="coerce")
    if values.isna().any():
        raise DataValidationError("Non-numeric or missing observation values")
    if ((values < MIN_RATE) | (values > MAX_RATE)).any():
        raise DataValidationError(
            f"Rate outside plausible range {MIN_RATE}-{MAX_RATE}%: "
            f"{values[(values < MIN_RATE) | (values > MAX_RATE)].tolist()}"
        )

    df = pd.DataFrame({"date": dates, "value": values.astype(float)})
    if df["date"].duplicated().any():
        dup = df.loc[df["date"].duplicated(), "date"].dt.date.tolist()
        raise DataValidationError(f"Duplicate dates: {dup}")

    df = df.sort_values("date").reset_index(drop=True)
    return PolicyRateFile(
        data=df,
        series_key=str(keys[0]),
        series_name=str(raw["Series"].iloc[0]),
        unit="percent",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_policy_rate(conn: sqlite3.Connection, path: str | Path) -> int:
    """Validate the CSV and replace the stored series. Returns rows stored.

    Re-running with the same or an updated file is safe: the series is replaced
    inside one transaction, so a failure leaves the old data untouched.
    """
    path = Path(path)
    parsed = read_policy_rate_csv(path)  # raises before touching the database

    with conn:  # one transaction
        conn.execute(
            "INSERT OR REPLACE INTO indicator_series "
            "(series_id, name, unit, source, frequency, description) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                SERIES_ID,
                parsed.series_name,
                parsed.unit,
                "State Bank of Pakistan (EasyData)",
                "irregular (recorded at each rate change)",
                f"Series key {parsed.series_key}. One row per policy-rate change.",
            ),
        )
        conn.execute(
            "DELETE FROM indicator_observations WHERE series_id = ?", (SERIES_ID,)
        )
        conn.executemany(
            "INSERT INTO indicator_observations (series_id, obs_date, value) "
            "VALUES (?, ?, ?)",
            [
                (SERIES_ID, d.strftime("%Y-%m-%d"), v)
                for d, v in zip(parsed.data["date"], parsed.data["value"])
            ],
        )
        conn.execute(
            "INSERT INTO import_log "
            "(series_id, source_file, file_sha256, row_count, imported_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                SERIES_ID,
                path.name,
                _sha256(path),
                len(parsed.data),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
    return len(parsed.data)
