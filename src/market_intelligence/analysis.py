"""Calculations on stored indicators. Takes DataFrames in, returns DataFrames.

No database or dashboard code here, so every function is easy to test.
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd


def policy_rate_monthly(
    changes: pd.DataFrame, end_month: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Turn rate-change events into one row per month.

    The value for a month is the rate in effect on the LAST day of that month.
    Months before the first recorded change are not invented.

    Columns returned:
        month_end      last day of the month
        rate_pct       policy rate in effect at month end
        change_1m_pp   change vs previous month, in percentage points
        change_12m_pp  change vs 12 months earlier, in percentage points
    """
    if changes.empty:
        raise ValueError("No observations to convert")

    s = changes.sort_values("date").set_index("date")["value"]
    first = s.index.min().to_period("M")
    last = (pd.Timestamp(end_month) if end_month is not None else s.index.max())
    last = last.to_period("M")
    if last < first:
        raise ValueError("end_month is before the first observation")

    month_ends = pd.period_range(first, last, freq="M").to_timestamp(how="end").normalize()
    filled = s.reindex(s.index.union(month_ends)).ffill().reindex(month_ends)

    out = pd.DataFrame({"month_end": month_ends, "rate_pct": filled.to_numpy()})
    out["change_1m_pp"] = out["rate_pct"].diff(1)
    out["change_12m_pp"] = out["rate_pct"].diff(12)
    return out


def freshness(
    last_change: pd.Timestamp,
    imported_at: str | datetime | None,
    today: date | None = None,
    max_import_age_days: int = 35,
) -> dict:
    """Describe how current the stored data is.

    Important: the source file lists only rate CHANGES, so 'no change since X'
    cannot be proven from the file alone. The warning is based on how long ago
    the file was downloaded, not on the date of the last change.
    """
    today = today or date.today()
    days_since_change = (today - last_change.date()).days

    days_since_import = None
    if imported_at is not None:
        imported = pd.Timestamp(imported_at).date()
        days_since_import = (today - imported).days

    stale = days_since_import is None or days_since_import > max_import_age_days
    return {
        "last_change_date": last_change.date(),
        "days_since_last_change": days_since_change,
        "days_since_import": days_since_import,
        "stale": stale,
    }


def active_window(series: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Trim a monthly series to the span between its first and last non-zero value.

    PAMA files list models as zero in years when they were not on sale, so
    leading and trailing zeros mean 'not sold', not 'zero demand'. Zeros
    inside the window are kept (they can be real stoppages).
    """
    s = series.sort_values("month").reset_index(drop=True)
    nonzero = s.index[s[value_col] > 0]
    if len(nonzero) == 0:
        return s.iloc[0:0]
    return s.loc[nonzero.min(): nonzero.max()].reset_index(drop=True)


def annual_totals(pama: pd.DataFrame) -> pd.DataFrame:
    """Sum monthly values per fiscal year (July-June) for each model and metric."""
    out = (
        pama.groupby(["fiscal_year", "model", "metric"], as_index=False)["value"].sum()
        .pivot(index=["fiscal_year", "model"], columns="metric", values="value")
        .reset_index()
    )
    out.columns.name = None
    return out


def real_policy_rate(monthly_rate: pd.DataFrame, cpi_yoy: pd.DataFrame) -> pd.DataFrame:
    """Policy rate minus year-on-year CPI inflation, in percentage points.

    monthly_rate: output of policy_rate_monthly (month_end, rate_pct, ...).
    cpi_yoy: DataFrame with columns date (month end) and value (CPI YoY, %).

    This is an ex-post real rate: it uses inflation that has already happened,
    not what lenders expect. Only months present in BOTH inputs are returned.
    """
    infl = cpi_yoy.rename(columns={"date": "month_end", "value": "cpi_yoy_pct"})[
        ["month_end", "cpi_yoy_pct"]
    ]
    out = monthly_rate[["month_end", "rate_pct"]].merge(infl, on="month_end", how="inner")
    out["real_rate_pp"] = out["rate_pct"] - out["cpi_yoy_pct"]
    return out.sort_values("month_end").reset_index(drop=True)
