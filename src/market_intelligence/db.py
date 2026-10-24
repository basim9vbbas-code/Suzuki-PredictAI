"""SQLite storage for market-intelligence indicators.

Design: raw observations are stored exactly as the source published them.
Derived values (monthly forward-fill, changes) are computed on read in
analysis.py, never stored, so a bug in a calculation cannot corrupt the data.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "processed" / "market_intelligence.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS indicator_series (
    series_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    unit        TEXT NOT NULL,
    source      TEXT NOT NULL,
    frequency   TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS indicator_observations (
    series_id TEXT NOT NULL REFERENCES indicator_series(series_id),
    obs_date  TEXT NOT NULL,          -- ISO date, YYYY-MM-DD
    value     REAL NOT NULL,
    PRIMARY KEY (series_id, obs_date)
);

CREATE TABLE IF NOT EXISTS import_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id   TEXT NOT NULL REFERENCES indicator_series(series_id),
    source_file TEXT NOT NULL,
    file_sha256 TEXT NOT NULL,
    row_count   INTEGER NOT NULL,
    imported_at TEXT NOT NULL         -- ISO timestamp, UTC
);
"""


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open the database, creating the file and tables if they do not exist."""
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def read_observations(conn: sqlite3.Connection, series_id: str) -> pd.DataFrame:
    """Return stored observations for a series as a DataFrame (date, value)."""
    rows = conn.execute(
        "SELECT obs_date, value FROM indicator_observations "
        "WHERE series_id = ? ORDER BY obs_date",
        (series_id,),
    ).fetchall()
    df = pd.DataFrame(
        [(r["obs_date"], r["value"]) for r in rows], columns=["date", "value"]
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def read_series_info(conn: sqlite3.Connection, series_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM indicator_series WHERE series_id = ?", (series_id,)
    ).fetchone()
    return dict(row) if row else None


def read_last_import(conn: sqlite3.Connection, series_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM import_log WHERE series_id = ? ORDER BY id DESC LIMIT 1",
        (series_id,),
    ).fetchone()
    return dict(row) if row else None
