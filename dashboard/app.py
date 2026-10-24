"""Market Intelligence dashboard, page 1: SBP policy rate.

Run from the project root:
    streamlit run dashboard\\app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.market_intelligence import db  # noqa: E402
from src.market_intelligence.analysis import freshness, policy_rate_monthly  # noqa: E402
from src.market_intelligence.importers.sbp_policy_rate import SERIES_ID  # noqa: E402

st.set_page_config(page_title="Suzuki PredictAI - Market Intelligence", layout="wide")
st.title("Suzuki PredictAI - Market Intelligence")
st.header("SBP policy rate")

conn = db.connect()
changes = db.read_observations(conn, SERIES_ID)
info = db.read_series_info(conn, SERIES_ID)
last_import = db.read_last_import(conn, SERIES_ID)

if changes.empty:
    st.error("No policy-rate data in the database yet.")
    st.code("python scripts\\load_policy_rate.py")
    st.stop()

monthly = policy_rate_monthly(changes, end_month=pd.Timestamp.today())
fresh = freshness(
    changes["date"].max(), last_import["imported_at"] if last_import else None
)

if fresh["stale"]:
    age = fresh["days_since_import"]
    st.warning(
        f"Data file was loaded {age} days ago. The source lists rate changes "
        "only, so a newer SBP decision would not appear until you download the "
        "file again from SBP EasyData and re-run the loader."
    )
else:
    st.success("Data file was loaded recently.")

latest = changes.iloc[-1]
previous = changes.iloc[-2]
c1, c2, c3 = st.columns(3)
c1.metric("Current rate (per file)", f"{latest['value']:.2f}%",
          f"{latest['value'] - previous['value']:+.2f} pp at last change")
c2.metric("Last change recorded", latest["date"].strftime("%d %b %Y"))
c3.metric("Rate changes on record", len(changes))

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=monthly["month_end"], y=monthly["rate_pct"], mode="lines",
    line_shape="hv", name="Rate in effect at month end",
))
fig.add_trace(go.Scatter(
    x=changes["date"], y=changes["value"], mode="markers", name="Rate change",
    marker=dict(size=8),
))
fig.update_layout(yaxis_title="Percent", xaxis_title="Date", hovermode="x unified")
st.plotly_chart(fig, width="stretch")

with st.expander("Monthly table"):
    table = monthly.copy()
    table["month_end"] = table["month_end"].dt.strftime("%Y-%m")
    st.dataframe(table.sort_values("month_end", ascending=False), hide_index=True)

st.caption(
    f"Source: {info['source']}. File: {last_import['source_file'] if last_import else 'n/a'}. "
    "Months before the first recorded change (May 2015) are not shown."
)
