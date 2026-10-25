"""Page 2: PAMA production and sales for Suzuki passenger cars (demand proxy)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.market_intelligence import db  # noqa: E402
from src.market_intelligence.analysis import active_window, annual_totals  # noqa: E402

st.title("Suzuki demand proxy (PAMA)")

conn = db.connect()
models_all = db.list_pama_models(conn)
if not models_all:
    st.error("No PAMA data in the database yet.")
    st.code("python scripts\\load_pama.py")
    st.stop()

suzuki = [m for m in models_all if m.startswith("Suzuki")]
default = ["Suzuki Alto"] if "Suzuki Alto" in suzuki else suzuki[:1]
chosen = st.multiselect("Suzuki models", suzuki, default=default)
metric = st.radio("Measure", ["sales", "production"], horizontal=True)
if not chosen:
    st.info("Pick at least one model.")
    st.stop()

pama = db.read_pama(conn, models=chosen)

fig = go.Figure()
for model in chosen:
    s = pama[(pama["model"] == model) & (pama["metric"] == metric)]
    s = active_window(s[["month", "value"]])
    if s.empty:
        continue
    fig.add_trace(go.Scatter(x=s["month"], y=s["value"], mode="lines", name=model))
fig.update_layout(yaxis_title=f"Units ({metric})", xaxis_title="Month", hovermode="x unified")
st.plotly_chart(fig, width="stretch")

st.info(
    "Each line starts at the model's first month with sales and ends at its last. "
    "Zeros before or after that mean the model was not on sale. "
    "The Alto sold in 2007-2012 and the Alto relaunched in June 2019 are different "
    "vehicles on one name; the old Alto has no sales from July 2012 to May 2019."
)

with st.expander("Annual totals by fiscal year (July to June)"):
    st.dataframe(annual_totals(pama), hide_index=True)

st.caption(
    "Source: PAMA monthly production and sales, as supplied in pama_history_2007_2026.xlsx. "
    "PAMA members only; Kia, Changan, MG and others are excluded. "
    "This is national, monthly data and a proxy for Suzuki's own sales, not a substitute."
)
