from pathlib import Path

import pandas as pd
import pytest

from src.market_intelligence import db
from src.market_intelligence.analysis import policy_rate_monthly, real_policy_rate
from src.market_intelligence.importers import pbs_cpi
from src.market_intelligence.importers.pbs_cpi import (
    GROUPS,
    load_pbs_review,
    read_pbs_review,
    series_id,
)
from src.market_intelligence.importers.sbp_policy_rate import DataValidationError

ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def fake_lines(title="December 2025", months=None, last_yoy=10.0, last_mom=0.9,
               index_row=True, base=5.0):
    """Lines shaped like the PDF text. Index 110 / 109 / 100 implies YoY 10.0, MoM 0.9."""
    months = months or [(2025, m) for m in range(1, 13)]
    lines = [f"   {title} ", "Table 1.a General Inflation (%) (Base 2015-16)"]
    for i, (y, m) in enumerate(months):
        is_last = i == len(months) - 1
        nums = [base + i * 0.1] * 18
        if is_last:
            nums[0], nums[1] = last_yoy, last_mom
        lines.append(f"{ABBR[m - 1]}-{y % 100:02d} " + " ".join(f"{n:.1f}" for n in nums) + " ")
    if index_row:
        lines.append("General 100.00 110.00 109.00 100.00 0.92 10.00 0.92 10.00")
    return lines


def patch(monkeypatch, tmp_path, lines):
    monkeypatch.setattr(pbs_cpi, "_read_text", lambda p: lines)
    f = tmp_path / "review.pdf"
    f.write_bytes(b"x")
    return f


def test_parses_all_groups_and_both_kinds(monkeypatch, tmp_path):
    r = read_pbs_review(patch(monkeypatch, tmp_path, fake_lines()))
    assert len(r.data) == 12 * 18
    assert r.data.series_id.nunique() == len(GROUPS) * 2
    assert r.review_month == pd.Timestamp("2025-12-31")
    last = r.data[r.data.obs_date == r.review_month].set_index("series_id").value
    assert last[series_id("cpi_national", "yoy")] == 10.0
    assert last[series_id("cpi_national", "mom")] == 0.9


def test_column_order_matches_table(monkeypatch, tmp_path):
    lines = fake_lines()
    row = ["Dec-25"] + [str(float(i)) for i in range(1, 19)]
    lines[-2] = " ".join(row)
    lines[-2] = lines[-2].replace(" 1.0 2.0", " 10.0 0.9", 1)
    r = read_pbs_review(patch(monkeypatch, tmp_path, lines)).data
    last = r[r.obs_date == pd.Timestamp("2025-12-31")].set_index("series_id").value
    assert last[series_id("cpi_urban", "yoy")] == 3.0
    assert last[series_id("cpi_rural", "mom")] == 6.0
    assert last[series_id("spi", "yoy")] == 15.0
    assert last[series_id("wpi", "mom")] == 18.0


def test_rejects_gap_in_months(monkeypatch, tmp_path):
    months = [(2024, 12)] + [(2025, m) for m in range(1, 13) if m != 6]  # 12 rows, June missing
    with pytest.raises(DataValidationError, match="Gaps"):
        read_pbs_review(patch(monkeypatch, tmp_path, fake_lines(months=months)))


def test_rejects_wrong_review_month(monkeypatch, tmp_path):
    with pytest.raises(DataValidationError, match="does not match"):
        read_pbs_review(patch(monkeypatch, tmp_path, fake_lines(title="November 2025")))


def test_rejects_too_few_rows(monkeypatch, tmp_path):
    months = [(2025, m) for m in range(6, 13)]
    with pytest.raises(DataValidationError, match="layout may have changed"):
        read_pbs_review(patch(monkeypatch, tmp_path, fake_lines(months=months)))


def test_rejects_implausible_value(monkeypatch, tmp_path):
    with pytest.raises(DataValidationError, match="outside"):
        read_pbs_review(patch(monkeypatch, tmp_path, fake_lines(base=500.0)))


def test_rejects_table_that_disagrees_with_index_levels(monkeypatch, tmp_path):
    with pytest.raises(DataValidationError, match="misread"):
        read_pbs_review(patch(monkeypatch, tmp_path, fake_lines(last_yoy=12.0)))


def test_rejects_missing_index_row(monkeypatch, tmp_path):
    with pytest.raises(DataValidationError, match="index row"):
        read_pbs_review(patch(monkeypatch, tmp_path, fake_lines(index_row=False)))


def test_rejects_missing_file(tmp_path):
    with pytest.raises(DataValidationError, match="not found"):
        read_pbs_review(tmp_path / "none.pdf")


def test_loading_a_later_review_extends_history_and_updates_revisions(monkeypatch, tmp_path):
    conn = db.connect(tmp_path / "t.db")
    f = patch(monkeypatch, tmp_path, fake_lines())
    load_pbs_review(conn, f)
    # Next review: Feb-25 to Jan-26, with a revised value for Feb-25.
    months = [(2025, m) for m in range(2, 13)] + [(2026, 1)]
    lines = fake_lines(title="January 2026", months=months, base=7.0)
    monkeypatch.setattr(pbs_cpi, "_read_text", lambda p: lines)
    load_pbs_review(conn, f)
    s = db.read_observations(conn, series_id("cpi_national", "yoy")).set_index("date")["value"]
    assert len(s) == 13                                  # Jan-25 kept, Jan-26 added
    assert s.loc["2025-01-31"] == 5.0                    # untouched older month
    assert s.loc["2025-02-28"] == 7.0                    # revised by newer review
    n = conn.execute("SELECT COUNT(*) FROM import_log").fetchone()[0]
    assert n == 2


def test_real_policy_rate_matches_by_month_and_subtracts():
    changes = pd.DataFrame({"date": pd.to_datetime(["2025-01-10", "2025-03-05"]), "value": [12.0, 10.0]})
    monthly = policy_rate_monthly(changes, end_month=pd.Timestamp("2025-04-30"))
    cpi = pd.DataFrame({"date": pd.to_datetime(["2025-02-28", "2025-03-31", "2025-06-30"]),
                        "value": [5.0, 7.0, 9.0]})
    out = real_policy_rate(monthly, cpi)
    assert out["month_end"].dt.strftime("%Y-%m").tolist() == ["2025-02", "2025-03"]
    assert out["real_rate_pp"].tolist() == [7.0, 3.0]


REAL = Path(__file__).resolve().parents[1] / "data" / "raw" / "pbs_cpi_review_2026_08.pdf"


@pytest.mark.skipif(not REAL.exists(), reason="real PBS PDF not in data/raw")
def test_real_pdf():
    r = read_pbs_review(REAL)
    assert len(r.data) == 22 * 18
    assert r.review_month == pd.Timestamp("2026-08-31")
    assert r.national_index == {"current": 300.5, "previous": 296.97, "year_ago": 270.35}
    v = r.data[r.data.series_id == series_id("cpi_national", "yoy")].set_index("obs_date").value
    assert v.loc["2024-11-30"] == 4.9
    assert v.loc["2026-03-31"] == 7.3
    assert v.loc["2026-04-30"] == 10.9
    assert v.loc["2026-08-31"] == 11.1
    wpi = r.data[r.data.series_id == series_id("wpi", "yoy")].set_index("obs_date").value
    assert wpi.loc["2026-08-31"] == 11.8
