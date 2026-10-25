import warnings
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import Workbook

from src.market_intelligence import db
from src.market_intelligence.analysis import active_window, annual_totals
from src.market_intelligence.importers.pama_history import (
    load_pama,
    read_pama_workbook,
)
from src.market_intelligence.importers.sbp_policy_rate import DataValidationError

warnings.filterwarnings("ignore", message="Cannot parse header or footer")

MONTHS_2P = ["July'24", "Aug'24", "Sept'24", "Oct'24", "Nov'24", "Dec'24",
             "Jan'25", "Feb'25", "Mar'25", "Apr'25", "May'25", "June'25"]
MONTHS_4P = [m.replace("'24", "'2024").replace("'25", "'2025") for m in MONTHS_2P]


def build(path: Path, sheet="2024-25", months=MONTHS_2P, alto=None, cumulative=None,
          extra_rows=()):
    alto = alto if alto is not None else [0, 0, 5, 10, 20, 0, 4, 3, 2, 1, 0, 0]
    wb = Workbook()
    ws = wb.active
    ws.title = sheet
    ws.append([None, "PRODUCTION & SALE DATA OF VEHICLES"])
    ws.append([None, "PASSENGER CARS"])
    ws.append([None, "1300cc and Above", None, None] + months + ["Cumulative"])
    ws.append([None, "Suzuki Swift", "Prod"] + [None] + [1] * 12 + [12])
    ws.append([None, None, "Sale"] + [None] + [2] * 12 + [24])
    ws.append([None, "Sub-Total", "Prod"] + [None] + [1] * 12 + [12])
    ws.append([None, None, "Sale"] + [None] + [2] * 12 + [24])
    ws.append([None, "Below 1000cc"])
    total = sum(alto) if cumulative is None else cumulative
    ws.append([None, "Suzuki Alto", "Prod."] + [None] + [x + 1 for x in alto] + [total + 12])
    ws.append([None, None, "Sale"] + [None] + alto + [total])
    for r in extra_rows:
        ws.append(r)
    wb.save(path)
    return path


def test_parses_both_two_and_four_digit_year_labels(tmp_path):
    for months in (MONTHS_2P, MONTHS_4P):
        df = read_pama_workbook(build(tmp_path / "a.xlsx", months=months)).data
        alto = df[(df.model == "Suzuki Alto") & (df.metric == "sales")]
        assert len(alto) == 12
        assert alto["month"].min() == "2024-07" and alto["month"].max() == "2025-06"


def test_sales_row_inherits_model_from_production_row(tmp_path):
    df = read_pama_workbook(build(tmp_path / "a.xlsx")).data
    swift = df[df.model == "Suzuki Swift"]
    assert set(swift.metric) == {"production", "sales"}
    assert swift[swift.metric == "sales"]["value"].eq(2).all()


def test_subsection_and_totals_are_labelled(tmp_path):
    df = read_pama_workbook(build(tmp_path / "a.xlsx")).data
    assert df[df.model == "Suzuki Swift"].subsection.iloc[0] == "1300cc_and_above"
    assert df[df.model == "Suzuki Alto"].subsection.iloc[0] == "below_1000cc"
    assert df[df.model == "Sub-Total"].row_type.eq("total").all()
    assert df[df.model == "Suzuki Alto"].row_type.eq("model").all()


def test_wrong_year_in_month_label_is_rejected(tmp_path):
    bad = [m.replace("'25", "'26") for m in MONTHS_2P]
    with pytest.raises(DataValidationError, match="requires"):
        read_pama_workbook(build(tmp_path / "a.xlsx", months=bad))


def test_header_with_missing_month_is_rejected(tmp_path):
    with pytest.raises(DataValidationError, match="expected 12"):
        read_pama_workbook(build(tmp_path / "a.xlsx", months=MONTHS_2P[:11]))


def test_non_numeric_value_is_rejected(tmp_path):
    bad = [None, "Suzuki Cultus", "Prod"] + [None] + ["abc"] + [1] * 11 + [11]
    with pytest.raises(DataValidationError, match="non-numeric"):
        read_pama_workbook(build(tmp_path / "a.xlsx", extra_rows=[bad]))


def test_negative_value_is_rejected(tmp_path):
    with pytest.raises(DataValidationError, match="Negative"):
        read_pama_workbook(build(tmp_path / "a.xlsx", alto=[-1] + [0] * 11))


def test_duplicate_rows_are_rejected(tmp_path):
    dup = [None, "Suzuki Alto", "Prod."] + [None] + [1] * 12 + [12]
    with pytest.raises(DataValidationError, match="duplicate"):
        read_pama_workbook(build(tmp_path / "a.xlsx", extra_rows=[dup]))


def test_cumulative_mismatch_is_reported_not_hidden(tmp_path):
    result = read_pama_workbook(build(tmp_path / "a.xlsx", cumulative=999))
    assert any("cumulative" in w for w in result.warnings)


def test_bad_sheet_name_is_rejected(tmp_path):
    with pytest.raises(DataValidationError, match="sheet name"):
        read_pama_workbook(build(tmp_path / "a.xlsx", sheet="Sheet1"))


def test_load_is_idempotent(tmp_path):
    xlsx = build(tmp_path / "a.xlsx")
    conn = db.connect(tmp_path / "t.db")
    load_pama(conn, xlsx)
    load_pama(conn, xlsx)
    n = conn.execute("SELECT COUNT(*) FROM pama_monthly").fetchone()[0]
    assert n == len(read_pama_workbook(xlsx).data)
    alto = db.read_pama(conn, models=["Suzuki Alto"], metric="sales")
    assert alto["value"].sum() == 45


def test_active_window_trims_not_sold_zeros_but_keeps_inner_zeros():
    s = pd.DataFrame({"month": pd.date_range("2024-01-01", periods=8, freq="MS"),
                      "value": [0, 0, 5, 0, 7, 0, 0, 0]})
    out = active_window(s)
    assert out["value"].tolist() == [5, 0, 7]
    assert active_window(s.assign(value=0)).empty


def test_annual_totals_sums_by_fiscal_year(tmp_path):
    df = read_pama_workbook(build(tmp_path / "a.xlsx")).data
    t = annual_totals(df[df.model == "Suzuki Alto"])
    assert t.iloc[0]["sales"] == 45 and t.iloc[0]["production"] == 57


REAL = Path(__file__).resolve().parents[1] / "data" / "raw" / "pama_history_2007_2026.xlsx"


@pytest.fixture(scope="module")
def df():
    return read_pama_workbook(REAL).data


@pytest.mark.skipif(not REAL.exists(), reason="real PAMA workbook not in data/raw")
class TestRealWorkbook:

    def test_covers_all_fiscal_years(self, df):
        assert df["month"].min() == "2007-07" and df["month"].max() == "2026-06"
        assert df["fiscal_year"].nunique() == 19
        assert (df["section"] != "unknown").all()

    def test_matches_pama_pdf_for_jul_to_sep_2025(self, df):
        def vals(model, metric):
            sel = df[(df.model == model) & (df.metric == metric)
                     & (df.fiscal_year == "2025-26")
                     & df.month.isin(["2025-07", "2025-08", "2025-09"])]
            return sel.sort_values("month")["value"].tolist()

        assert vals("Suzuki Alto", "production") == [4808, 6238, 6319]
        assert vals("Suzuki Alto", "sales") == [2327, 4193, 4718]
        assert vals("Suzuki Swift", "sales") == [522, 1473, 1403]
        assert vals("Suzuki Cultus", "sales") == [239, 497, 427]
        assert vals("Suzuki Every", "sales") == [230, 376, 721]
        assert vals("TOTAL CARS", "sales") == [7135, 10057, 12126]

    def test_alto_is_two_separate_eras(self, df):
        alto = df[(df.model == "Suzuki Alto") & (df.metric == "sales")].set_index("month")["value"]
        assert alto.loc["2012-07":"2012-12"].sum() == 0
        assert alto.loc["2018-07":"2019-05"].sum() == 0
        assert alto.loc["2019-06"] == 1685
        assert alto.loc["2019-06":"2026-06"].gt(0).sum() >= 80
