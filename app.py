import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path


st.set_page_config(
    page_title="AI Climate Digital Twin - India",
    layout="wide"
)


# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    csv_path = Path(__file__).parent / "rainfall_prediction_results.csv"
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


df = load_data()


# -----------------------------
# Helper Functions
# -----------------------------
def rainfall_risk(rain):
    if rain < 2.5:
        return "Low/No Rain"
    elif rain < 15:
        return "Moderate Rain"
    elif rain < 35:
        return "Heavy Rain"
    else:
        return "Extreme Rain"


def risk_color_order():
    return {
        "Low/No Rain": "#2ecc71",
        "Moderate Rain": "#f1c40f",
        "Heavy Rain": "#e67e22",
        "Extreme Rain": "#e74c3c"
    }


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Climate Twin Controls")

available_dates = sorted(df["date"].dt.date.unique())

selected_date = st.sidebar.selectbox(
    "Select Date",
    available_dates,
    index=len(available_dates) - 1
)

map_type = st.sidebar.radio(
    "Map Layer",
    [
        "Actual Rainfall",
        "Predicted Rainfall",
        "Prediction Error",
        "Risk Category",
        "What-if Scenario"
    ]
)

rainfall_change = st.sidebar.slider(
    "What-if Rainfall Change (%)",
    min_value=-50,
    max_value=100,
    value=20,
    step=5
)

temperature_change = st.sidebar.slider(
    "What-if Temperature Change (°C)",
    min_value=0.0,
    max_value=5.0,
    value=2.0,
    step=0.5
)


# -----------------------------
# Filter Selected Date
# -----------------------------
map_df = df[df["date"].dt.date == selected_date].copy()

map_df["scenario_rainfall"] = (
    map_df["predicted_rainfall_tuned"] * (1 + rainfall_change / 100)
)

map_df["scenario_risk"] = map_df["scenario_rainfall"].apply(rainfall_risk)


# -----------------------------
# Header
# -----------------------------
st.title("AI-Powered Digital Twin of India's Climate")
st.subheader("Rainfall Prediction, Risk Mapping, and What-if Scenario Simulation")

st.write(
    """
    This prototype demonstrates a climate digital twin using gridded rainfall observations,
    AI-based next-day rainfall prediction, risk classification, and scenario simulation.
    """
)


# -----------------------------
# Metrics
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Selected Date",
        str(selected_date)
    )

with col2:
    st.metric(
        "Avg Actual Rainfall",
        f"{map_df['actual_next_day_rainfall'].mean():.2f} mm"
    )

with col3:
    st.metric(
        "Avg Predicted Rainfall",
        f"{map_df['predicted_rainfall_tuned'].mean():.2f} mm"
    )

with col4:
    st.metric(
        "Avg Error",
        f"{map_df['error'].mean():.2f} mm"
    )


# -----------------------------
# Select Map Column
# -----------------------------
if map_type == "Actual Rainfall":
    color_col = "actual_next_day_rainfall"
    title = "Actual Next-Day Rainfall"
    color_scale = "Blues"

elif map_type == "Predicted Rainfall":
    color_col = "predicted_rainfall_tuned"
    title = "AI Predicted Rainfall"
    color_scale = "Blues"

elif map_type == "Prediction Error":
    color_col = "error"
    title = "Prediction Error"
    color_scale = "Reds"

elif map_type == "Risk Category":
    title = "Rainfall Risk Category"
    color_col = "risk"
    color_scale = None

else:
    color_col = "scenario_rainfall"
    title = f"What-if Scenario: Rainfall {rainfall_change:+d}%, Temperature +{temperature_change}°C"
    color_scale = "Blues"


# -----------------------------
# Map
# -----------------------------
st.subheader(title)

if map_type == "Risk Category":
    fig = px.scatter_mapbox(
        map_df,
        lat="lat",
        lon="lon",
        color="risk",
        color_discrete_map=risk_color_order(),
        hover_data={
            "lat": True,
            "lon": True,
            "actual_next_day_rainfall": ":.2f",
            "predicted_rainfall_tuned": ":.2f",
            "error": ":.2f",
            "risk": True
        },
        zoom=4,
        height=650
    )

elif map_type == "What-if Scenario":
    fig = px.scatter_mapbox(
        map_df,
        lat="lat",
        lon="lon",
        color="scenario_rainfall",
        color_continuous_scale="Blues",
        hover_data={
            "lat": True,
            "lon": True,
            "predicted_rainfall_tuned": ":.2f",
            "scenario_rainfall": ":.2f",
            "scenario_risk": True
        },
        zoom=4,
        height=650
    )

else:
    fig = px.scatter_mapbox(
        map_df,
        lat="lat",
        lon="lon",
        color=color_col,
        color_continuous_scale=color_scale,
        hover_data={
            "lat": True,
            "lon": True,
            "actual_next_day_rainfall": ":.2f",
            "predicted_rainfall_tuned": ":.2f",
            "error": ":.2f",
            "risk": True
        },
        zoom=4,
        height=650
    )


# -----------------------------
# Marker Visibility
# -----------------------------
fig.update_traces(
    marker=dict(
        size=5,
        opacity=0.72
    )
)


# -----------------------------
# Darker No-label Basemap
# -----------------------------
fig.update_layout(
    mapbox_style="white-bg",
    mapbox_layers=[
        {
            "below": "traces",
            "sourcetype": "raster",
            "source": [
                "https://a.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}.png",
                "https://b.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}.png",
                "https://c.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}.png",
                "https://d.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}.png"
            ],
            "opacity": 1
        }
    ],
    margin={"r": 0, "t": 0, "l": 0, "b": 0}
)

st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Scenario Summary
# -----------------------------
if map_type == "What-if Scenario":
    st.subheader("Scenario Impact Summary")

    risk_counts = map_df["scenario_risk"].value_counts().reset_index()
    risk_counts.columns = ["Risk Category", "Grid Points"]

    c1, c2 = st.columns(2)

    with c1:
        st.dataframe(risk_counts, use_container_width=True)

    with c2:
        fig_bar = px.bar(
            risk_counts,
            x="Risk Category",
            y="Grid Points",
            title="Scenario Risk Distribution",
            color="Risk Category",
            color_discrete_map=risk_color_order()
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# -----------------------------
# Data Table
# -----------------------------
st.subheader("Climate Twin State Table")

show_cols = [
    "date",
    "lat",
    "lon",
    "actual_next_day_rainfall",
    "predicted_rainfall_tuned",
    "error",
    "risk"
]

if map_type == "What-if Scenario":
    show_cols += ["scenario_rainfall", "scenario_risk"]

st.dataframe(
    map_df[show_cols].head(5000),
    use_container_width=True
)


# -----------------------------
# Download Button
# -----------------------------
csv = map_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download selected date data as CSV",
    data=csv,
    file_name=f"climate_twin_{selected_date}.csv",
    mime="text/csv"
)


# -----------------------------
# Model Notes
# -----------------------------
st.subheader("Model Validation Summary")

st.write(
    """
    The rainfall prediction model was trained using spatial coordinates, seasonal features,
    and lag-based rainfall features. A persistence baseline was used for comparison.
    
    In time-based validation, the baseline performed better on MAE during dry/low-rainfall
    periods, while the AI model reduced RMSE, indicating better handling of larger rainfall
    error events. This is relevant for climate adaptation because extreme rainfall deviations
    are more important for disaster-risk planning.
    """
)

st.info(
    "PoC scope: rainfall-only climate twin. Temperature and sectoral impact layers can be added in the next version."
)