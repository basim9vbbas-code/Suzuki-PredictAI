"""Importer for the PAMA production-and-sales history workbook.

The workbook has one sheet per fiscal year (July to June), 2007-08 onward.
Each sheet is a set of blocks (passenger cars, jeeps/pickups, trucks, buses,
tractors, motorcycles). Every block has a header row of 12 month labels, then
pairs of rows per model: a 'Prod' row and a 'Sale' row.

Layout differs between years (model names, column positions, two- or
four-digit years in month labels), so nothing is hard-coded by position: the
parser finds the header row of each block and reads values under it.

PAMA reports cover PAMA members only (Kia, Changan, MG and others excluded).
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import warnings as py_warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

from .sbp_policy_rate import DataValidationError

SERIES_ID = "pama_monthly"

_MONTH_LABEL = re.compile(r"^\s*([A-Za-z]+)\.?\s*'\s*(\d{2}|\d{4})\s*$")
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_FISCAL_SHEET = re.compile(r"^(\d{4})-(\d{2})$")

_PROD_LABELS = {"prod", "prod."}
_SALE_LABELS = {"sale", "sale."}

# Canonical names for labels that vary in spelling or case.
_NAME_FIXES = {
    "suzuki ravi": "Suzuki Ravi",
    "fronx suzuki": "Suzuki Fronx",
    "suzuki fronx": "Suzuki Fronx",
}

COLUMNS = [
    "fiscal_year", "section", "subsection", "model", "row_type",
    "metric", "month", "value",
]


@dataclass
class PamaParseResult:
    data: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def _clean(text: object) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _parse_month_label(label: str, fiscal_start_year: int, sheet: str) -> pd.Timestamp:
    m = _MONTH_LABEL.match(label)
    mon = _MONTHS.get(m.group(1)[:3].lower()) if m else None
    if not m or mon is None:
        raise DataValidationError(f"Sheet {sheet}: unreadable month label {label!r}")
    year = int(m.group(2))
    if year < 100:
        year += 2000
    expected = fiscal_start_year if mon >= 7 else fiscal_start_year + 1
    if year != expected:
        raise DataValidationError(
            f"Sheet {sheet}: month label {label!r} is year {year}, but fiscal "
            f"year {sheet} requires {expected} for that month"
        )
    return pd.Timestamp(year=year, month=mon, day=1)


def _section_for(heading: str) -> tuple[str | None, str | None]:
    """Map a heading row to (section, subsection); None means 'not a heading we know'."""
    h = _clean(heading).lower()
    if h in {"production & sale data of vehicles"}:
        return None, None
    if h == "passenger cars":
        return "passenger_cars", None
    if h in {"electric car", "electric vehicle"}:
        return "passenger_cars", "electric"
    if "1300" in h and "above" in h:
        return "passenger_cars", "1300cc_and_above"
    if h == "1000 cc":
        return "passenger_cars", "1000cc"
    if h in {"800 cc & below 1000cc", "below 1000cc"}:
        return "passenger_cars", "below_1000cc"
    if "jeep" in h or "lcv" in h or "pick" in h:
        return "jeeps_pickups_lcv", h
    if h == "trucks":
        return "trucks", None
    if h == "buses":
        return "buses", None
    if h == "farm tractors":
        return "farm_tractors", None
    if h.startswith("motorcycles"):
        return "motorcycles_3wheelers", None
    return None, None


def _canonical_model(label: str, section: str | None) -> str:
    name = _clean(label)
    fixed = _NAME_FIXES.get(name.lower())
    if fixed:
        return fixed
    if section == "motorcycles_3wheelers" and name.lower() == "suzuki":
        return "Suzuki (motorcycles)"
    return name


def _to_number(value: object, where: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise DataValidationError(f"{where}: boolean where a number was expected")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text in {"", "-", "–", "—"}:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        raise DataValidationError(f"{where}: non-numeric value {value!r}") from None


def _parse_sheet(ws, sheet: str) -> tuple[list[dict], list[str]]:
    fy = _FISCAL_SHEET.match(sheet)
    if not fy:
        raise DataValidationError(f"Unexpected sheet name {sheet!r} (expected YYYY-YY)")
    fiscal_start = int(fy.group(1))

    records: list[dict] = []
    warnings: list[str] = []
    month_cols: dict[int, pd.Timestamp] = {}
    cumulative_col: int | None = None
    section: str | None = None
    subsection: str | None = None
    last_model: str | None = None

    for rownum, row in enumerate(ws.iter_rows(values_only=True), start=1):
        cells = list(row)

        # 1. Month header row: starts a new block.
        found = {
            j: v for j, v in enumerate(cells)
            if isinstance(v, str) and _MONTH_LABEL.match(v)
        }
        if found:
            if len(found) != 12:
                raise DataValidationError(
                    f"Sheet {sheet} row {rownum}: header has {len(found)} month "
                    "labels, expected 12"
                )
            month_cols = {
                j: _parse_month_label(v, fiscal_start, sheet) for j, v in found.items()
            }
            cumulative_col = next(
                (j for j, v in enumerate(cells)
                 if isinstance(v, str) and v.strip().lower() == "cumulative"),
                None,
            )
            label = cells[1] if len(cells) > 1 else None
            if label:  # e.g. '1300cc and Above' sits on the header row
                sec, sub = _section_for(label)
                if sec:
                    section, subsection = sec, sub
            last_model = None
            continue

        label = cells[1] if len(cells) > 1 else None
        metric_cell = cells[2] if len(cells) > 2 else None
        metric_txt = _clean(metric_cell).lower() if metric_cell is not None else ""

        # 2. Heading row (label but no metric and no numbers).
        if metric_cell is None:
            if isinstance(label, str) and label.strip():
                sec, sub = _section_for(label)
                if sec:
                    if sec != section:
                        subsection = None
                    section = sec
                    if sub is not None or sec == "passenger_cars":
                        subsection = sub if sub is not None else subsection
            continue

        # 3. Data row.
        if metric_txt in _PROD_LABELS:
            metric = "production"
        elif metric_txt in _SALE_LABELS:
            metric = "sales"
        else:
            continue  # stray marker rows (e.g. an ellipsis)

        if not month_cols:
            raise DataValidationError(
                f"Sheet {sheet} row {rownum}: data row before any month header"
            )
        if isinstance(label, str) and label.strip():
            last_model = _canonical_model(label, section)
        if last_model is None:
            raise DataValidationError(f"Sheet {sheet} row {rownum}: data row without a model name")

        low = last_model.lower()
        row_type = "total" if low.startswith(("sub-total", "subtotal", "total")) else "model"

        month_values: list[float] = []
        for j, month in month_cols.items():
            v = _to_number(cells[j] if j < len(cells) else None,
                           f"Sheet {sheet} row {rownum} ({last_model})")
            if v is None:
                continue
            month_values.append(v)
            records.append({
                "fiscal_year": sheet, "section": section or "unknown",
                "subsection": subsection or "", "model": last_model,
                "row_type": row_type, "metric": metric,
                "month": month.strftime("%Y-%m"), "value": v,
            })

        if cumulative_col is not None and cumulative_col < len(cells):
            cum = _to_number(cells[cumulative_col], f"Sheet {sheet} row {rownum}")
            if cum is not None and abs(cum - sum(month_values)) > 0.5:
                warnings.append(
                    f"{sheet} row {rownum} {last_model} {metric}: months sum to "
                    f"{sum(month_values):.0f} but file's cumulative says {cum:.0f}"
                )
    return records, warnings


def read_pama_workbook(path: str | Path) -> PamaParseResult:
    path = Path(path)
    if not path.exists():
        raise DataValidationError(f"File not found: {path}")
    with py_warnings.catch_warnings():
        # The workbook has an unreadable print header; it is irrelevant to the data.
        py_warnings.filterwarnings("ignore", message="Cannot parse header or footer")
        wb = load_workbook(path, data_only=True)

    records: list[dict] = []
    warnings: list[str] = []
    for sheet in wb.sheetnames:
        recs, warns = _parse_sheet(wb[sheet], sheet)
        records.extend(recs)
        warnings.extend(warns)
    if not records:
        raise DataValidationError("No production/sales rows found")

    df = pd.DataFrame(records, columns=COLUMNS)
    key = ["fiscal_year", "section", "subsection", "model", "metric", "month"]
    dup = df[df.duplicated(key, keep=False)]
    if not dup.empty:
        sample = dup[key].head(5).to_dict("records")
        raise DataValidationError(f"{len(dup)} duplicate rows, for example {sample}")
    if (df["value"] < 0).any():
        raise DataValidationError("Negative production or sales values found")
    return PamaParseResult(data=df, warnings=warnings)


def load_pama(conn: sqlite3.Connection, path: str | Path) -> PamaParseResult:
    """Validate the workbook and replace the stored PAMA table."""
    path = Path(path)
    result = read_pama_workbook(path)  # raises before touching the database
    df = result.data

    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO indicator_series "
            "(series_id, name, unit, source, frequency, description) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                SERIES_ID,
                "PAMA vehicle production and sales",
                "units",
                "Pakistan Automotive Manufacturers Association",
                "monthly",
                "PAMA members only (excludes Kia, Changan, MG and others).",
            ),
        )
        conn.execute("DELETE FROM pama_monthly")
        conn.executemany(
            "INSERT INTO pama_monthly "
            "(fiscal_year, section, subsection, model, row_type, metric, month, value) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [tuple(r) for r in df[COLUMNS].itertuples(index=False, name=None)],
        )
        conn.execute(
            "INSERT INTO import_log "
            "(series_id, source_file, file_sha256, row_count, imported_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                SERIES_ID,
                path.name,
                hashlib.sha256(path.read_bytes()).hexdigest(),
                len(df),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
    return result
