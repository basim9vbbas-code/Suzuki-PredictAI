"""Suzuki Pakistan Enterprise Intelligence & Showroom.

Corrections compared with the previous version:
  * Model errors are shown, never hidden behind a fixed number.
  * Inputs are aligned to the columns the model was trained on.
  * Each entry in the model dropdown is a different fitted model, or is not shown.
  * Evaluation numbers are computed here from data, with time-ordered splits.
  * Feature importances are shown only when the model really provides them.
  * The NLP tab shows real model output or a real error, never an invented score.
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
MASTER_CSV = BASE_DIR / "master_market_intelligence_features.csv"
SAVED_MODEL = BASE_DIR / "src" / "suzuki_alto_model.pkl"
NLP_CSV = BASE_DIR / "processed_nlp_sentiment_features.csv"

TARGET = "suzuki_alto_sales_units"
DATE_COL = "date"
FEATURES = [
    "petrol_pkr_per_liter",
    "kibor_6m_pct",
    "auto_finance_avg_rate_pct",
    "total_duty_rate_pct",
    "sro_event_flag",
]
ADMIN_PASSCODE = os.environ.get("SUZUKI_ADMIN_PASSCODE", "admin123")

st.set_page_config(
    page_title="Suzuki Pakistan Enterprise Intelligence & Showroom",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stMetric { background-color: #1a1c24; padding: 15px; border-radius: 8px;
                border: 1px solid #2d3748; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background-color: #1a1c24; border-radius: 6px;
                color: #cbd5e0; padding: 10px 16px; font-weight: 500; }
    .stTabs [aria-selected="true"] { background-color: #3182ce !important; color: white !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

for key, default in (("logged_in", False), ("user_role", None)):
    if key not in st.session_state:
        st.session_state[key] = default


# ----------------------------------------------------------------- data + models
def _mtime(path: Path) -> float:
    """File modification time; used so caches refresh when a file is replaced."""
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data
def _read_master(mtime: float) -> pd.DataFrame | None:
    if not MASTER_CSV.exists():
        return None
    df = pd.read_csv(MASTER_CSV)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df = df.sort_values(DATE_COL).reset_index(drop=True)
    return df


def load_master_data() -> pd.DataFrame | None:
    return _read_master(_mtime(MASTER_CSV))


def has_training_columns(df: pd.DataFrame | None) -> bool:
    return df is not None and all(c in df.columns for c in FEATURES + [TARGET, DATE_COL])


def make_models() -> dict:
    return {
        "Ridge Regularized Linear": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=0
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=0),
    }


@st.cache_resource
def fit_models(df: pd.DataFrame) -> dict:
    """Fit each model on all rows of the master data. Different models, different results."""
    fitted = {}
    for name, model in make_models().items():
        model.fit(df[FEATURES], df[TARGET])
        fitted[name] = model
    return fitted


def load_saved_model():
    return _load_saved_model(_mtime(SAVED_MODEL))


@st.cache_resource
def _load_saved_model(mtime: float):
    if not SAVED_MODEL.exists():
        return None, None
    try:
        return joblib.load(SAVED_MODEL), None
    except Exception as exc:  # noqa: BLE001 - shown to the user
        return None, f"Could not load {SAVED_MODEL.name}: {exc}"


def expected_columns(model) -> list[str]:
    cols = getattr(model, "feature_names_in_", None)
    if cols is None and hasattr(model, "steps"):  # pipeline
        cols = getattr(model.steps[-1][1], "feature_names_in_", None)
        if cols is None:
            cols = getattr(model.steps[0][1], "feature_names_in_", None)
    return list(cols) if cols is not None else list(FEATURES)


def build_input(model, scenario: dict, df_master: pd.DataFrame | None):
    """Return (DataFrame in the model's column order, list of columns filled from history)."""
    cols = expected_columns(model)
    filled = []
    row = {}
    for c in cols:
        if c in scenario:
            row[c] = scenario[c]
        elif df_master is not None and c in df_master.columns:
            row[c] = df_master[c].iloc[-1]  # latest known value
            filled.append(c)
        else:
            raise ValueError(
                f"The model needs a column '{c}' that the app cannot supply. "
                f"Model columns: {cols}"
            )
    return pd.DataFrame([row], columns=cols), filled


def seasonal_baseline(df: pd.DataFrame) -> float | None:
    """Real seasonal-naive forecast: sales 12 months before the next month."""
    if df is None or TARGET not in df.columns or DATE_COL not in df.columns:
        return None
    s = df.set_index(DATE_COL)[TARGET]
    target_month = s.index.max() + pd.DateOffset(months=1)
    ref = target_month - pd.DateOffset(months=12)
    match = s[(s.index.year == ref.year) & (s.index.month == ref.month)]
    return float(match.iloc[0]) if len(match) else None


@st.cache_data
def run_backtest(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Time-ordered cross-validation. Models are refitted on each training fold only."""
    data = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    lag12 = data[TARGET].shift(12)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    errors = {name: [] for name in list(make_models()) + ["Seasonal naive (same month last year)"]}
    actuals_by_model = {name: [] for name in errors}
    for train_idx, test_idx in tscv.split(data):
        x_tr, y_tr = data.loc[train_idx, FEATURES], data.loc[train_idx, TARGET]
        for name, model in make_models().items():
            model.fit(x_tr, y_tr)
            pred = model.predict(data.loc[test_idx, FEATURES])
            errors[name].extend(pred - data.loc[test_idx, TARGET].to_numpy())
            actuals_by_model[name].extend(data.loc[test_idx, TARGET].to_numpy())
        usable = [i for i in test_idx if not np.isnan(lag12[i])]
        name = "Seasonal naive (same month last year)"
        errors[name].extend(lag12[usable].to_numpy() - data.loc[usable, TARGET].to_numpy())
        actuals_by_model[name].extend(data.loc[usable, TARGET].to_numpy())
    rows = []
    for name, errs in errors.items():
        if not errs:
            continue
        e = np.array(errs)
        a = np.array(actuals_by_model[name])
        nonzero = a != 0
        rows.append({
            "Model": name,
            "Test points": len(e),
            "MAE (units)": float(np.abs(e).mean()),
            "RMSE (units)": float(np.sqrt((e ** 2).mean())),
            "MAPE (%)": float((np.abs(e[nonzero]) / a[nonzero]).mean() * 100),
        })
    return pd.DataFrame(rows).sort_values("MAE (units)").reset_index(drop=True)


def importance_table(model) -> pd.DataFrame | None:
    names = expected_columns(model)
    inner = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(inner, "feature_importances_"):
        values, label = inner.feature_importances_, "Importance weight"
    elif hasattr(inner, "coef_"):
        values, label = np.abs(np.ravel(inner.coef_)), "Coefficient magnitude (standardised inputs)"
    else:
        return None
    if len(values) != len(names):
        return None
    return pd.DataFrame({"Indicator": names, label: values}).sort_values(label, ascending=False)


# ----------------------------------------------------------------- sidebar session
if st.session_state["logged_in"]:
    st.sidebar.markdown(f"**Current Session:** `{st.session_state['user_role'].upper()}`")
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.session_state["user_role"] = None
        st.rerun()

# ----------------------------------------------------------------- login gate
if not st.session_state["logged_in"]:
    st.title("Suzuki Pakistan Portal - Authentication")
    st.markdown("Please select your operational access role to proceed.")
    left, right = st.columns(2)
    with left:
        st.subheader("Admin Executive Portal")
        st.caption("Analytics, forecasting, backtesting and NLP inspection.")
        passcode = st.text_input("Enter Admin Passcode:", type="password", key="admin_pwd")
        if st.button("Authenticate as Admin"):
            if passcode == ADMIN_PASSCODE:
                st.session_state["logged_in"] = True
                st.session_state["user_role"] = "admin"
                st.rerun()
            else:
                st.error("Invalid admin passcode.")
    with right:
        st.subheader("Customer Showroom")
        st.caption("Vehicle catalog, specifications and financing calculator.")
        if st.button("Open Customer Showroom"):
            st.session_state["logged_in"] = True
            st.session_state["user_role"] = "customer"
            st.rerun()

# ----------------------------------------------------------------- admin portal
elif st.session_state["user_role"] == "admin":
    st.title("Suzuki Executive Portal - Demand Analytics & ML Intelligence")
    df_master = load_master_data()
    st.markdown(
        "Scenario forecasting of monthly Suzuki Alto sales from macroeconomic indicators. "
        "Geographic scope and units follow the data in `master_market_intelligence_features.csv`."
    )

    st.sidebar.markdown("---")
    st.sidebar.header("Scenario Simulation Controls")
    petrol = st.sidebar.slider("Petrol Price (PKR/Liter)", 200.0, 350.0, 280.0)
    kibor = st.sidebar.slider("6-Month KIBOR Rate (%)", 8.0, 25.0, 11.5)
    duty = st.sidebar.slider("Total Effective Duty Rate (%)", 15.0, 30.0, 22.0)
    sro = st.sidebar.selectbox("Regulatory SRO Active Flag", [0, 1], index=0)
    spread = st.sidebar.number_input("Auto-finance spread over KIBOR (pp)", 0.0, 10.0, 2.5, 0.1)
    auto_finance = kibor + spread

    scenario = {
        "petrol_pkr_per_liter": petrol,
        "kibor_6m_pct": kibor,
        "auto_finance_avg_rate_pct": auto_finance,
        "total_duty_rate_pct": duty,
        "sro_event_flag": sro,
    }

    tab1, tab2, tab3, tab4 = st.tabs([
        "Live Demand Forecaster",
        "Model Evaluation & Backtest",
        "Transformer NLP & Policy Sentiment",
        "Historical Market Data",
    ])

    # ---- Tab 1: forecaster
    with tab1:
        st.subheader("Machine Learning Model Inference & Scenario Simulation")

        options: dict = {}
        if has_training_columns(df_master) and len(df_master) >= 24:
            options.update(fit_models(df_master))
        saved, saved_error = load_saved_model()
        if saved is not None:
            options[f"Saved model file ({type(saved).__name__})"] = saved
        elif saved_error:
            st.warning(saved_error)
        baseline_value = seasonal_baseline(df_master)

        names = list(options)
        if baseline_value is not None:
            names.append("Seasonal naive (same month last year)")

        if not names:
            st.error(
                "No model is available. Provide `master_market_intelligence_features.csv` with the "
                f"columns {FEATURES + [TARGET, DATE_COL]} (at least 24 rows), or a valid "
                "`src/suzuki_alto_model.pkl`."
            )
        else:
            choice = st.selectbox("Select model:", names)
            if choice.startswith("Seasonal naive"):
                predicted = baseline_value
                model = None
                st.info(
                    "The seasonal naive forecast ignores the sliders by design: it repeats the same "
                    "month of last year. It is the baseline the other models must beat."
                )
            else:
                model = options[choice]
                try:
                    x, filled = build_input(model, scenario, df_master)
                    predicted = float(model.predict(x)[0])
                except Exception as exc:  # noqa: BLE001 - never hide a model failure
                    st.error(f"Model prediction failed: {exc}")
                    st.stop()
                if filled:
                    st.caption(
                        f"Columns not controlled by the sliders were held at their latest value "
                        f"in the data: {', '.join(filled)}."
                    )

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted monthly sales", f"{predicted:,.0f} units")
            c2.metric("Effective auto financing rate", f"{auto_finance:.2f}%")
            c3.metric("Regulatory SRO status", "Active" if sro == 1 else "Inactive")
            st.caption(
                "A point estimate with no confidence range. Test errors for each model are on the "
                "evaluation tab; compare the prediction with them before relying on it."
            )

            if model is not None:
                table = importance_table(model)
                if table is not None:
                    with st.expander("Inspect model feature importance"):
                        st.dataframe(table, width="stretch", hide_index=True)

    # ---- Tab 2: evaluation
    with tab2:
        st.subheader("Time-Ordered Cross-Validation Performance")
        if not has_training_columns(df_master) or len(df_master) < 36:
            st.info(
                "The backtest needs `master_market_intelligence_features.csv` with the columns "
                f"{FEATURES + [TARGET, DATE_COL]} and at least 36 monthly rows."
            )
        else:
            with st.spinner("Running time-ordered backtest..."):
                results = run_backtest(df_master)
            st.dataframe(
                results.style.format({
                    "MAE (units)": "{:,.1f}", "RMSE (units)": "{:,.1f}", "MAPE (%)": "{:.1f}",
                }),
                width="stretch", hide_index=True,
            )
            best = results.iloc[0]["Model"]
            st.markdown(
                f"**Lowest average error:** {best}. These numbers are computed here from your data. "
                "Each fold trains only on months before the months it is tested on."
            )
            st.caption(
                f"{len(df_master)} monthly rows, 5 expanding-window folds. With few rows the "
                "differences between models can be noise; treat small gaps as ties."
            )

    # ---- Tab 3: NLP
    with tab3:
        st.subheader("Transformer NLP Sentiment & Policy Analysis")
        st.markdown(
            "General-purpose English sentiment model "
            "(`distilbert-base-uncased-finetuned-sst-2-english`). It was trained on movie-review "
            "sentences, not on finance or policy text, so treat its output as indicative only."
        )
        if NLP_CSV.exists():
            st.dataframe(pd.read_csv(NLP_CSV), width="stretch")
        else:
            st.warning("`processed_nlp_sentiment_features.csv` not found.")

        st.markdown("---")
        st.subheader("Live Transformer Text Inspector")
        text = st.text_input(
            "Enter a headline or policy announcement:",
            value="Commercial banks report surge in auto loan approvals as inflation rate stabilizes.",
        )
        if st.button("Execute Live Sentiment Analysis"):
            try:
                from transformers import pipeline

                @st.cache_resource
                def get_classifier():
                    return pipeline(
                        "sentiment-analysis",
                        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                        device=-1,
                    )

                with st.spinner("Running the model (the first run downloads it)..."):
                    result = get_classifier()(text[:512])[0]
                label, conf = result["label"], float(result["score"])
                a, b, c = st.columns(3)
                a.metric("Classification", label)
                b.metric("Confidence score", f"{conf:.4f}")
                c.metric("Signed polarity", f"{conf if label == 'POSITIVE' else -conf:.4f}")
            except Exception as exc:  # noqa: BLE001
                st.error(
                    f"The sentiment model could not run: {exc}. Install it with "
                    "`pip install transformers` and a CPU build of PyTorch, then try again."
                )

    # ---- Tab 4: history
    with tab4:
        st.subheader("Historical Market & Macroeconomic Trends")
        if df_master is None or DATE_COL not in df_master.columns:
            st.info("`master_market_intelligence_features.csv` not found or has no date column.")
        else:
            plot_df = df_master.set_index(DATE_COL)
            left, right = st.columns(2)
            if TARGET in plot_df.columns:
                left.markdown("**Alto sales (units)**")
                left.line_chart(plot_df[[TARGET]])
            if "petrol_pkr_per_liter" in plot_df.columns:
                right.markdown("**Petrol price (PKR/liter)**")
                right.line_chart(plot_df[["petrol_pkr_per_liter"]])

# ----------------------------------------------------------------- customer showroom
elif st.session_state["user_role"] == "customer":
    st.title("Suzuki Digital Showroom")
    st.markdown("Explore specifications, indicative prices and an installment estimate.")
    st.caption(
        "Prices below were entered by the project team and are not checked against Pak Suzuki's "
        "current price list. Confirm with an authorised dealer."
    )

    EXTERIOR_IMG = "https://suzukipakistan.com/api/media/file/image-21.png"
    INTERIOR_IMG = "https://cache4.pakwheels.com/ad_pictures/1519/suzuki-alto-vxl-2-2022-151943072.webp"

    catalog = {
        "Suzuki Alto VX (660cc Manual)": {
            "Price": "PKR 2,331,000", "Numeric Price": 2331000,
            "Engine": "660cc R06A Petrol Engine", "Transmission": "5-Speed Manual",
            "Fuel Economy": "18 - 22 km/L",
            "Specifications": ["Power Steering", "Dual Airbags", "Engine Immobilizer",
                               "Manual Air Conditioning"],
        },
        "Suzuki Alto VXR (660cc Manual)": {
            "Price": "PKR 2,707,000", "Numeric Price": 2707000,
            "Engine": "660cc R06A Petrol Engine", "Transmission": "5-Speed Manual",
            "Fuel Economy": "18 - 22 km/L",
            "Specifications": ["Anti-lock Braking System (ABS)", "Dual Airbags", "Power Windows",
                               "Keyless Entry", "Audio System"],
        },
        "Suzuki Alto VXL AGS (660cc Auto)": {
            "Price": "PKR 3,045,000", "Numeric Price": 3045000,
            "Engine": "660cc R06A Petrol Engine", "Transmission": "Auto Gear Shift (AGS)",
            "Fuel Economy": "19 - 23 km/L",
            "Specifications": ["Auto Gear Shift Transmission", "ABS with EBD",
                               "Retractable Side Mirrors", "Touchscreen Infotainment", "Dual Airbags"],
        },
    }

    selected = st.selectbox("Select Suzuki Alto variant:", list(catalog))
    v = catalog[selected]
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(selected)
        st.markdown(f"**Indicative retail price:** `{v['Price']}`")
        tab_ext, tab_int = st.tabs(["Exterior", "Interior"])
        for tab, url, label in ((tab_ext, EXTERIOR_IMG, "Exterior"), (tab_int, INTERIOR_IMG, "Interior")):
            with tab:
                try:
                    st.image(url, caption=f"{selected} - {label}", width="stretch")
                except Exception:  # noqa: BLE001 - external image may be unavailable
                    st.info("Image could not be loaded from its source.")
        st.markdown("---")
        st.markdown("**Technical specifications:**")
        st.write(f"- Engine: {v['Engine']}")
        st.write(f"- Transmission: {v['Transmission']}")
        st.write(f"- Fuel economy: {v['Fuel Economy']}")
        st.markdown("**Standard features:**")
        for spec in v["Specifications"]:
            st.write(f"- {spec}")

    with col2:
        st.subheader("Auto Financing Calculator")
        down_pct = st.slider("Down payment (%)", 20, 50, 30)
        years = st.selectbox("Financing tenure (years)", [1, 2, 3, 4, 5], index=2)
        markup = st.slider("Estimated bank markup rate (%)", 10.0, 25.0, 14.0)

        price = v["Numeric Price"]
        down = price * down_pct / 100.0
        principal = price - down
        n = years * 12
        r = markup / 100.0 / 12
        installment = principal * r * (1 + r) ** n / ((1 + r) ** n - 1) if r > 0 else principal / n

        st.metric("Required down payment", f"PKR {int(down):,}")
        st.metric("Principal financed", f"PKR {int(principal):,}")
        st.metric("Estimated monthly installment", f"PKR {int(installment):,}/month")
        st.caption("An estimate using a standard reducing-balance formula. Banks' terms and fees differ.")

    st.markdown("---")
    st.subheader("Dealership Booking & Payment Procedures")
    p1, p2, p3 = st.columns(3)
    p1.info("Bank Pay Order / Demand Draft\n\nPayable to the manufacturer at authorised dealerships.")
    p2.success("Commercial Bank Auto Financing\n\nAvailable through partner banks, subject to their terms.")
    p3.warning("Cash Settlement\n\nSettled at the authorised dealership.")