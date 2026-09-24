"""
Correllia CPI Inflation Forecast -- customer pilot demo.

Run with:
    streamlit run app.py
"""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from cpi_lstm_v4 import IMPORTANCES_PATH, PREDICTIONS_PATH, TARGET, WALK_FORWARD_PATH
from cpi_inference import latest_forecast, load_model

st.set_page_config(
    page_title="Correllia | CPI Inflation Forecast",
    page_icon="\U0001F4C8",
    layout="wide",
)

# ---------------------------------------------------------------- styling --

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        padding: 1rem 1rem 0.6rem 1rem;
    }
    .pilot-badge {
        display: inline-block;
        background: #6C4FF0;
        color: white;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        padding: 0.18rem 0.6rem;
        border-radius: 999px;
        margin-left: 0.6rem;
        vertical-align: middle;
    }
    .section-caption { color: rgba(128,128,128,0.9); font-size: 0.92rem; margin-top: -0.6rem; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data --

@st.cache_resource(show_spinner="Loading model...")
def get_model_ckpt():
    _, ckpt = load_model()
    return ckpt


@st.cache_resource(show_spinner="Loading model and running inference...")
def get_forecast_default():
    return latest_forecast()


@st.cache_data(show_spinner="Running inference on your data...")
def get_forecast_from_upload(csv_bytes: bytes):
    return latest_forecast(csv_path=io.BytesIO(csv_bytes))


@st.cache_data
def get_predictions():
    return pd.read_csv(PREDICTIONS_PATH, parse_dates=["anchor_date", "target_date"])


@st.cache_data
def get_importances():
    return pd.read_csv(IMPORTANCES_PATH)


@st.cache_data
def get_walk_forward():
    return pd.read_csv(WALK_FORWARD_PATH)


try:
    ckpt = get_model_ckpt()
except FileNotFoundError:
    st.error(
        "No trained model found. Run `python cpi_lstm_v4.py` first to train "
        "the model and generate the artifacts this demo reads."
    )
    st.stop()

predictions = get_predictions()
importances = get_importances()
walk_forward = get_walk_forward()


# ---------------------------------------------------------------- header --

st.markdown(
    f"### Correllia &nbsp;&middot;&nbsp; CPI Inflation Forecast "
    f"<span class='pilot-badge'>PILOT DEMO</span>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='section-caption'>LSTM forecast of the U.S. CPI (CPIAUCSL).</div>",
    unsafe_allow_html=True,
)
st.write("")

uploaded_csv = st.file_uploader(
    "Upload your own CSV to generate a custom projection (must be formatted like the training data)",
    type=["csv"],
)
with st.expander("What does the CSV need to look like?"):
    st.markdown(
        f"- A `sasdate` column with monthly dates\n"
        f"- A `{TARGET}` column (the CPI level)\n"
        f"- The {len(ckpt['selected_cols'])} feature columns this model was trained on: "
        f"`{'`, `'.join(ckpt['selected_cols'])}`\n"
        f"- At least {ckpt['seq_len'] + 1} months of rows\n\n"
        f"If you don't upload a file, the bundled FRED-MD dataset is used."
    )

if uploaded_csv is not None:
    try:
        result = get_forecast_from_upload(uploaded_csv.getvalue())
        data_source_label = f"your uploaded file (**{uploaded_csv.name}**)"
    except Exception as e:
        st.error(f"Couldn't generate a forecast from this file: {e}")
        st.stop()
else:
    result = get_forecast_default()
    data_source_label = "the bundled FRED-MD dataset"

history = result["history"]
forecasts = result["forecasts"]
as_of_date = result["as_of_date"]
as_of_level = result["as_of_level"]

horizons = forecasts["horizon"].tolist()
month_names = {h: row.target_date.strftime("%B %Y") for h, row in zip(horizons, forecasts.itertuples())}

# per-horizon backtest RMSE (level), used as an uncertainty band on the chart
wf_rmse = {
    h: float(walk_forward[f"lstm_rmse_h{h}"].mean())
    for h in horizons
    if f"lstm_rmse_h{h}" in walk_forward.columns
}

st.markdown(
    f"<div class='section-caption'>Using {data_source_label}, data through "
    f"<b>{as_of_date.strftime('%B %Y')}</b>.</div>",
    unsafe_allow_html=True,
)
st.write("")

# ------------------------------------------------------------- forecast --

cols = st.columns(1 + len(horizons))

with cols[0]:
    st.metric("Current CPI level", f"{as_of_level:.2f}", help=f"As of {as_of_date.strftime('%B %Y')}")

for i, h in enumerate(horizons):
    row = forecasts[forecasts["horizon"] == h].iloc[0]
    with cols[i + 1]:
        st.metric(
            f"+{h} month{'s' if h > 1 else ''} — {month_names[h]}",
            f"{row['pred_level']:.2f}",
            delta=f"{row['pred_delta']:+.2f} ({row['pred_mom_pct']:+.2f}% MoM)",
        )

st.write("")

fig = go.Figure()

recent = history.tail(36)
fig.add_trace(go.Scatter(
    x=recent["date"], y=recent["level"],
    mode="lines", name="Actual CPI",
    line=dict(color="#2E5EAA", width=2.5),
))

fc_x = [as_of_date] + forecasts["target_date"].tolist()
fc_y = [as_of_level] + forecasts["pred_level"].tolist()
fig.add_trace(go.Scatter(
    x=fc_x, y=fc_y,
    mode="lines+markers", name="Forecast",
    line=dict(color="#6C4FF0", width=2.5, dash="dash"),
    marker=dict(size=8),
))

if wf_rmse:
    upper = [as_of_level] + [
        row.pred_level + wf_rmse.get(row.horizon, 0.0) for row in forecasts.itertuples()
    ]
    lower = [as_of_level] + [
        row.pred_level - wf_rmse.get(row.horizon, 0.0) for row in forecasts.itertuples()
    ]
    fig.add_trace(go.Scatter(
        x=fc_x + fc_x[::-1], y=upper + lower[::-1],
        fill="toself", fillcolor="rgba(108,79,240,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip", showlegend=True, name="Backtest RMSE band",
    ))

fig.update_layout(
    height=420,
    margin=dict(l=10, r=10, t=30, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    xaxis_title=None, yaxis_title="CPIAUCSL",
    template="plotly_white",
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Shaded band is the average level RMSE for that horizon from the walk-forward "
    "backtest below — a rough sense of typical forecast error, not a formal confidence interval."
)

st.divider()

# ----------------------------------------------------------- track record --

st.markdown("#### Model track record (held-out test period)")
st.markdown(
    "<div class='section-caption'>Predicted vs. actual CPI level on data the model "
    "never trained on, shown separately for each forecast horizon.</div>",
    unsafe_allow_html=True,
)
st.write("")

tabs = st.tabs([f"+{h} month{'s' if h > 1 else ''}" for h in horizons])

for h, tab in zip(horizons, tabs):
    with tab:
        sub = predictions[predictions["horizon"] == h].sort_values("target_date")

        err = sub["pred_level_lstm"] - sub["actual_level"]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mae = float(np.mean(np.abs(err)))

        base_err = sub["pred_level_baseline"] - sub["actual_level"]
        base_rmse = float(np.sqrt(np.mean(base_err ** 2)))

        m1, m2, m3 = st.columns(3)
        m1.metric("LSTM RMSE (level)", f"{rmse:.3f}")
        m2.metric("LSTM MAE (level)", f"{mae:.3f}")
        m3.metric(
            "vs. naive baseline",
            f"{base_rmse:.3f} RMSE",
            delta=f"{rmse - base_rmse:+.3f}",
            delta_color="inverse",
        )

        tfig = go.Figure()
        tfig.add_trace(go.Scatter(
            x=sub["target_date"], y=sub["actual_level"],
            mode="lines", name="Actual",
            line=dict(color="#2E5EAA", width=2),
        ))
        tfig.add_trace(go.Scatter(
            x=sub["target_date"], y=sub["pred_level_lstm"],
            mode="lines", name="Predicted (LSTM)",
            line=dict(color="#6C4FF0", width=2),
        ))
        tfig.update_layout(
            height=340,
            margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            template="plotly_white",
            hovermode="x unified",
        )
        st.plotly_chart(tfig, use_container_width=True)

st.divider()

# ----------------------------------------------------------------- two-up --

left, right = st.columns([3, 2])

with left:
    st.markdown("#### Backtest: LSTM vs. naive baseline")
    st.markdown(
        "<div class='section-caption'>Expanding-window walk-forward backtest across "
        "5 rolling time periods, so results aren’t an artifact of one lucky split.</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    bar_fig = go.Figure()
    bar_fig.add_trace(go.Bar(
        x=[f"+{h}mo" for h in horizons],
        y=[walk_forward[f"baseline_rmse_h{h}"].mean() for h in horizons],
        name="Naive baseline", marker_color="#B0B0B0",
    ))
    bar_fig.add_trace(go.Bar(
        x=[f"+{h}mo" for h in horizons],
        y=[walk_forward[f"lstm_rmse_h{h}"].mean() for h in horizons],
        name="LSTM", marker_color="#6C4FF0",
    ))
    bar_fig.update_layout(
        barmode="group",
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="Avg. RMSE across folds (level)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(bar_fig, use_container_width=True)

with right:
    st.markdown("#### Top macro drivers")
    st.markdown(
        "<div class='section-caption'>Features the model relies on most "
        "(random forest importance, feature selection stage).</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    top_imp = importances.sort_values("importance", ascending=True).tail(12)
    imp_fig = go.Figure(go.Bar(
        x=top_imp["importance"], y=top_imp["feature"],
        orientation="h", marker_color="#2E5EAA",
    ))
    imp_fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        template="plotly_white",
    )
    st.plotly_chart(imp_fig, use_container_width=True)

st.divider()
st.caption(
    "Correllia pilot demo — for evaluation purposes only. Forecasts are generated by an "
    "LSTM trained on historical FRED-MD data and are not investment or economic advice."
)
