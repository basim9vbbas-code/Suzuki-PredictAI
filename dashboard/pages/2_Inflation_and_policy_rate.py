"""Page 3: PBS inflation and the SBP policy rate."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.market_intelligence import db  # noqa: E402
from src.market_intelligence.analysis import policy_rate_monthly, real_policy_rate  # noqa: E402
from src.market_intelligence.importers.pbs_cpi import GROUPS, series_id  # noqa: E402
from src.market_intelligence.importers.sbp_policy_rate import SERIES_ID as RATE_ID  # noqa: E402

st.title("Inflation and the policy rate")

conn = db.connect()
cpi = db.read_observations(conn, series_id("cpi_national", "yoy"))
if cpi.empty:
    st.error("No inflation data in the database yet.")
    st.code("python scripts\\load_pbs_cpi.py")
    st.stop()

labels = dict(GROUPS)
choice = st.multiselect(
    "Inflation measures", list(labels), default=["cpi_national", "cpi_urban", "cpi_rural"],
    format_func=lambda k: labels[k],
)
kind = st.radio("Change", ["Year-on-year", "Month-on-month"], horizontal=True)
kind_key = "yoy" if kind == "Year-on-year" else "mom"

fig = go.Figure()
for g in choice:
    s = db.read_observations(conn, series_id(g, kind_key))
    fig.add_trace(go.Scatter(x=s["date"], y=s["value"], mode="lines+markers", name=labels[g]))
fig.update_layout(yaxis_title="Percent", xaxis_title="Month", hovermode="x unified")
st.plotly_chart(fig, width="stretch")

st.subheader("Policy rate against inflation")
rate_changes = db.read_observations(conn, RATE_ID)
if rate_changes.empty:
    st.info("Load the policy rate first: python scripts\\load_policy_rate.py")
else:
    monthly = policy_rate_monthly(rate_changes, end_month=cpi["date"].max())
    both = real_policy_rate(monthly, cpi)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=both["month_end"], y=both["rate_pct"], mode="lines", line_shape="hv", name="Policy rate"))
    fig2.add_trace(go.Scatter(x=both["month_end"], y=both["cpi_yoy_pct"], mode="lines", name="CPI inflation (YoY)"))
    fig2.add_trace(go.Bar(x=both["month_end"], y=both["real_rate_pp"], name="Real rate (rate minus inflation)", opacity=0.4))
    fig2.update_layout(yaxis_title="Percent / percentage points", hovermode="x unified")
    st.plotly_chart(fig2, width="stretch")
    latest = both.iloc[-1]
    st.metric(
        f"Real policy rate, {latest['month_end']:%b %Y}",
        f"{latest['real_rate_pp']:+.1f} pp",
        help="Policy rate minus year-on-year CPI. Uses inflation that has already happened, not expectations.",
    )
    st.caption(
        "The policy rate comes from a file that lists changes only. If SBP has moved the "
        "rate since the last change on record, the real rate shown here is out of date."
    )

st.caption(
    f"Source: Pakistan Bureau of Statistics, Monthly Review on Price Indices (base 2015-16), "
    f"published to one decimal. Data covers {cpi['date'].min():%b %Y} to {cpi['date'].max():%b %Y}. "
    "Longer history needs the PBS historical series, which is not loaded yet."
)
