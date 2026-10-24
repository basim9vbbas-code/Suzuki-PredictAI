from pathlib import Path

import pandas as pd
import pytest

from src.market_intelligence import db
from src.market_intelligence.analysis import freshness, policy_rate_monthly
from src.market_intelligence.importers.sbp_policy_rate import (
    SERIES_ID,
    DataValidationError,
    load_policy_rate,
    read_policy_rate_csv,
)

HEADER = (
    "Dataset,Series Key,Series,Observation Date,Observation Value,Unit,"
    "Observation Status,Observation Status Comment\n"
)


def make_csv(tmp_path: Path, rows: list[tuple[str, str]], status="Normal", unit="Percent"):
    lines = [
        f"Policy Rates,KEY1,SBP Policy (Target) Rate,{d},{v},{unit},{status},\n"
        for d, v in rows
    ]
    p = tmp_path / "rate.csv"
    p.write_text(HEADER + "".join(lines))
    return p


GOOD = [("28-Jan-2025", "12.0"), ("06-May-2025", "11.0"), ("16-Dec-2025", "10.5")]


def test_reads_valid_file_sorted_oldest_first(tmp_path):
    parsed = read_policy_rate_csv(make_csv(tmp_path, list(reversed(GOOD))))
    assert parsed.data["date"].is_monotonic_increasing
    assert parsed.data["value"].tolist() == [12.0, 11.0, 10.5]
    assert parsed.unit == "percent"


def test_rejects_duplicate_dates(tmp_path):
    with pytest.raises(DataValidationError, match="Duplicate"):
        read_policy_rate_csv(make_csv(tmp_path, GOOD + [("06-May-2025", "11.0")]))


def test_rejects_bad_date(tmp_path):
    with pytest.raises(DataValidationError, match="dates"):
        read_policy_rate_csv(make_csv(tmp_path, [("2025/01/28", "12.0")]))


def test_rejects_non_numeric_value(tmp_path):
    with pytest.raises(DataValidationError, match="numeric"):
        read_policy_rate_csv(make_csv(tmp_path, [("28-Jan-2025", "abc")]))


def test_rejects_implausible_rate(tmp_path):
    with pytest.raises(DataValidationError, match="plausible"):
        read_policy_rate_csv(make_csv(tmp_path, [("28-Jan-2025", "120")]))


def test_rejects_non_normal_status(tmp_path):
    with pytest.raises(DataValidationError, match="status"):
        read_policy_rate_csv(make_csv(tmp_path, GOOD, status="Provisional"))


def test_rejects_missing_column(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("Series Key,Series\nKEY1,x\n")
    with pytest.raises(DataValidationError, match="Missing columns"):
        read_policy_rate_csv(p)


def test_rejects_missing_file(tmp_path):
    with pytest.raises(DataValidationError, match="not found"):
        read_policy_rate_csv(tmp_path / "nope.csv")


def test_monthly_forward_fill_uses_rate_in_effect_at_month_end(tmp_path):
    changes = read_policy_rate_csv(make_csv(tmp_path, GOOD)).data
    m = policy_rate_monthly(changes, end_month=pd.Timestamp("2026-02-15"))
    rate = dict(zip(m["month_end"].dt.strftime("%Y-%m"), m["rate_pct"]))
    assert rate["2025-01"] == 12.0          # change on the 28th counts for January
    assert rate["2025-04"] == 12.0          # unchanged through April
    assert rate["2025-05"] == 11.0          # change on 6 May
    assert rate["2025-12"] == 10.5          # change on 16 Dec
    assert rate["2026-02"] == 10.5          # extended to end_month
    assert m["month_end"].iloc[0] == pd.Timestamp("2025-01-31")
    assert "2024-12" not in rate            # nothing invented before first change


def test_monthly_changes_in_percentage_points(tmp_path):
    changes = read_policy_rate_csv(make_csv(tmp_path, GOOD)).data
    m = policy_rate_monthly(changes, end_month=pd.Timestamp("2026-01-31"))
    row_may = m[m["month_end"] == pd.Timestamp("2025-05-31")].iloc[0]
    assert row_may["change_1m_pp"] == pytest.approx(-1.0)
    row_jan26 = m[m["month_end"] == pd.Timestamp("2026-01-31")].iloc[0]
    assert row_jan26["change_12m_pp"] == pytest.approx(-1.5)  # 12.0 -> 10.5


def test_load_is_idempotent_and_logs_imports(tmp_path):
    csv = make_csv(tmp_path, GOOD)
    conn = db.connect(tmp_path / "t.db")
    assert load_policy_rate(conn, csv) == 3
    assert load_policy_rate(conn, csv) == 3
    stored = db.read_observations(conn, SERIES_ID)
    assert len(stored) == 3                   # no duplicates after second load
    n_logs = conn.execute("SELECT COUNT(*) FROM import_log").fetchone()[0]
    assert n_logs == 2
    info = db.read_series_info(conn, SERIES_ID)
    assert info["unit"] == "percent"


def test_failed_import_leaves_existing_data_untouched(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    load_policy_rate(conn, make_csv(tmp_path, GOOD))
    bad = tmp_path / "bad.csv"
    bad.write_text(HEADER + "P,K,S,28-Jan-2025,999,Percent,Normal,\n")
    with pytest.raises(DataValidationError):
        load_policy_rate(conn, bad)
    assert len(db.read_observations(conn, SERIES_ID)) == 3


def test_freshness_flags_old_import():
    last = pd.Timestamp("2026-04-28")
    old = freshness(last, "2026-04-30T10:00:00+00:00", today=pd.Timestamp("2026-10-25").date())
    assert old["stale"] is True
    new = freshness(last, "2026-10-20T10:00:00+00:00", today=pd.Timestamp("2026-10-25").date())
    assert new["stale"] is False


def test_real_file_if_present():
    real = Path(__file__).resolve().parents[1] / "data" / "raw" / "sbp_policy_rate.csv"
    if not real.exists():
        pytest.skip("real SBP file not in data/raw")
    parsed = read_policy_rate_csv(real)
    assert len(parsed.data) == 39
    assert parsed.data["date"].min() == pd.Timestamp("2015-05-25")
    assert parsed.data["value"].iloc[-1] == 11.5
