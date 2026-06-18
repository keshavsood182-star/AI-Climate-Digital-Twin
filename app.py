import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="AI Rainfall-Risk Digital Twin — Punjab + Haryana",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Styling
# ============================================================
def inject_lofi_station_theme():
    custom_css = """
    <style>
        [data-testid="stAppViewContainer"] {
            background: transparent !important;
        }
        .stApp {
            position: relative;
            background: transparent !important;
            color: #ebf7ff;
        }
        [data-testid="stHeader"] {
            background: rgba(5, 12, 20, 0.18) !important;
            backdrop-filter: blur(10px);
            z-index: 999 !important;
        }
        .block-container {
            position: relative !important;
            z-index: 10 !important;
            max-width: 1180px;
            padding-top: 2rem;
        }
        section.main {
            position: relative !important;
            z-index: 10 !important;
        }
        [data-testid="stVerticalBlock"] {
            position: relative;
            z-index: 10;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #eef9ff !important;
        }
        p, span, div, label {
            color: rgba(235, 247, 255, 0.88);
        }
        .station-label,
        .status-pill,
        .metric-label {
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.78rem;
            font-weight: 700;
            color: rgba(210, 235, 255, 0.78);
        }
        .hero-card {
            background: linear-gradient(
                135deg,
                rgba(8, 20, 34, 0.78),
                rgba(18, 44, 68, 0.58)
            );
            border: 1px solid rgba(180, 225, 255, 0.18);
            border-radius: 28px;
            backdrop-filter: blur(18px);
            box-shadow:
                0 24px 80px rgba(0, 0, 0, 0.28),
                inset 0 1px 0 rgba(255, 255, 255, 0.08);
            padding: 2.2rem;
            margin-bottom: 1.6rem;
        }
        .hero-title {
            font-family: "Inter", "Segoe UI", "SF Pro Display", "Helvetica Neue", Arial, sans-serif;
            color: #eef9ff;
            font-size: 2.6rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin: 0.3rem 0 1rem;
            line-height: 1.05;
            text-shadow: 0 0 24px rgba(120, 210, 255, 0.18);
        }
        .hero-subtitle {
            color: rgba(225, 242, 255, 0.82);
            line-height: 1.75;
            font-size: 1rem;
            max-width: 80rem;
        }
        .glass-card,
        .dashboard-card,
        .info-card,
        .metric-card,
        .prediction-card,
        .hero-card {
            background: linear-gradient(
                135deg,
                rgba(8, 20, 34, 0.78),
                rgba(18, 44, 68, 0.58)
            );
            border: 1px solid rgba(180, 225, 255, 0.18);
            border-radius: 24px;
            backdrop-filter: blur(18px);
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
            color: #edf8ff !important;
        }
        .metric-card {
            min-height: 110px;
        }
        .metric-label {
            margin-bottom: 0.45rem;
        }
        .metric-value {
            color: #f4fbff;
            font-size: 1.6rem;
            font-weight: 800;
        }
        .small-muted {
            color: rgba(230, 245, 255, 0.78);
            font-size: 0.95rem;
        }
        .note-box,
        .good-box,
        .bad-box {
            padding: 1rem 1.1rem;
            margin: 1rem 0;
        }
        .note-box {
            border-color: rgba(180, 225, 255, 0.16);
        }
        .good-box {
            border-color: rgba(100, 220, 170, 0.18);
        }
        .bad-box {
            border-color: rgba(240, 160, 160, 0.22);
        }
        .stButton > button {
            background: rgba(8, 18, 30, 0.82);
            color: #eaf7ff !important;
            border: 1px solid rgba(170, 220, 255, 0.22);
            border-radius: 14px;
            padding: 0.7rem 1.1rem;
            font-weight: 700;
            backdrop-filter: blur(12px);
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
        }
        .stButton > button:hover {
            background: rgba(24, 56, 84, 0.88);
            border-color: rgba(180, 230, 255, 0.36);
            color: #ffffff !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(8, 18, 30, 0.48);
            border-radius: 999px;
            color: rgba(230, 245, 255, 0.72);
            border: 1px solid rgba(180, 225, 255, 0.12);
            padding: 0.6rem 1rem;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(120, 190, 255, 0.22);
            color: #ffffff;
            border-color: rgba(180, 230, 255, 0.34);
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.45rem 0.75rem;
            border-radius: 999px;
            background: rgba(8, 18, 30, 0.64);
            border: 1px solid rgba(180, 225, 255, 0.18);
            color: #eaf7ff;
            font-weight: 700;
            font-size: 0.85rem;
            backdrop-filter: blur(12px);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg,
                rgba(5, 12, 22, 0.92),
                rgba(10, 26, 42, 0.88)
            );
            border-right: 1px solid rgba(180, 225, 255, 0.12);
        }
        [data-testid="stSidebar"] * {
            color: rgba(235, 247, 255, 0.88);
        }
        [data-testid="stMetric"] {
            background: rgba(8, 18, 30, 0.58);
            border: 1px solid rgba(180, 225, 255, 0.14);
            border-radius: 20px;
            padding: 1rem;
            backdrop-filter: blur(12px);
        }
        [data-testid="stMetricLabel"] {
            color: rgba(210, 235, 255, 0.72);
        }
        [data-testid="stMetricValue"] {
            color: #f4fbff;
            font-weight: 800;
        }
        .map-card,
        .chart-card {
            background: rgba(245, 250, 255, 0.88);
            border-radius: 24px;
            padding: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.42);
            color: #101828;
        }
        .lofi-weather-bg {
            position: fixed;
            inset: 0;
            width: 100vw;
            height: 100vh;
            z-index: -3;
            pointer-events: none;
            overflow: hidden;
            background: #07111f;
        }
        .lofi-video-bg {
            position: fixed;
            inset: 0;
            width: 100vw;
            height: 100vh;
            z-index: -3;
            pointer-events: none;
            overflow: hidden;
            background: #07111f;
        }
        .lofi-video-bg video {
            width: 100%;
            height: 100%;
            object-fit: cover;
            opacity: 0.45;
            filter: brightness(0.72) saturate(0.88) contrast(1.08);
        }
        .lofi-video-overlay {
            position: fixed;
            inset: 0;
            z-index: -2;
            pointer-events: none;
            background:
                radial-gradient(circle at center, rgba(255,255,255,0.08), rgba(0,0,0,0.30)),
                linear-gradient(rgba(5,15,25,0.18), rgba(5,15,25,0.45));
        }
        .lofi-scanlines {
            position: fixed;
            inset: 0;
            z-index: -1;
            pointer-events: none;
            opacity: 0.10;
            background: repeating-linear-gradient(
                to bottom,
                rgba(255,255,255,0.10) 0px,
                rgba(255,255,255,0.10) 1px,
                transparent 2px,
                transparent 5px
            );
            mix-blend-mode: overlay;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

inject_lofi_station_theme()


# ============================================================
# Paths
# ============================================================
# ============================================================
APP_DIR = Path(__file__).resolve().parent


def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def resolve_project_dirs():
    data_dir = first_existing([
        APP_DIR / "DEV_HANDOFF" / "data",
        APP_DIR / "data",
        APP_DIR.parent / "DEV_HANDOFF" / "data",
    ])
    models_dir = first_existing([
        APP_DIR / "DEV_HANDOFF" / "models",
        APP_DIR / "models",
        APP_DIR.parent / "DEV_HANDOFF" / "models",
    ])
    docs_dir = first_existing([
        APP_DIR / "DEV_HANDOFF" / "docs",
        APP_DIR / "docs",
        APP_DIR.parent / "DEV_HANDOFF" / "docs",
    ])
    return {
        "data": data_dir or APP_DIR / "DEV_HANDOFF" / "data",
        "models": models_dir or APP_DIR / "DEV_HANDOFF" / "models",
        "docs": docs_dir or APP_DIR / "DEV_HANDOFF" / "docs",
    }


DIRS = resolve_project_dirs()


# ============================================================
# File constants
# ============================================================
DASHBOARD_CSV = "v2_2025_dashboard_predictions.csv"
SIMULATOR_CSV = "v2_2025_simulator_feature_dataset.csv"
FEATURE_CONFIG_FILE = "v2_feature_config.json"
V1_V2_METRICS_FILE = "v1_vs_v2_metrics_summary.csv"
HEAVY_METRICS_FILE = "heavy_rain_detector_metrics.csv"
RAIN_MODEL_FILE = "v2_rain_probability_classifier_xgb.joblib"
AMOUNT_MODEL_FILE = "v2_rainfall_amount_regressor_xgb.joblib"
HEAVY_MODEL_FILE = "v2_heavy_rain_detector_xgb.joblib"

RISK_ORDER = [
    "No / Low Rain",
    "Low / No Rain Risk",
    "Moderate Rain",
    "High Risk",
    "Severe Risk",
]

RISK_COLOR_MAP = {
    "No / Low Rain": "#d8f6ff",
    "Low / No Rain Risk": "#d8f6ff",
    "Moderate Rain": "#ffe8a3",
    "High Risk": "#ffb36b",
    "Severe Risk": "#ff6b6b",
    "Unknown": "#dddddd",
}

COLUMN_ALIASES = {
    "date": ["date", "Date"],
    "lat": ["lat", "latitude", "Latitude"],
    "lon": ["lon", "longitude", "Longitude"],
    "pred_rain": [
        "predicted_rainfall_soft_hybrid_v2",
        "soft_hybrid_rainfall_v2",
        "predicted_rainfall_amount_v2",
        "predicted_rainfall",
        "predicted_rainfall_mm",
        "rainfall_prediction",
    ],
    "rain_prob": [
        "rain_probability_v2",
        "rain_probability",
        "predicted_rain_probability",
        "rain_prob",
        "rain_proba",
    ],
    "heavy_prob": [
        "heavy_rain_probability_v2",
        "heavy_rain_probability",
        "heavy_probability",
        "heavy_prob",
        "heavy_proba",
    ],
    "actual": [
        "target_rainfall",
        "actual_next_day_rainfall",
        "actual_rainfall",
        "rainfall",
        "today_actual_rainfall",
    ],
    "risk": [
        "final_dashboard_risk_v2",
        "dashboard_risk_v2",
        "final_dashboard_risk",
        "risk_category_v2",
        "risk_category",
        "predicted_risk_category_v2",
        "actual_risk_category",
    ],
    "heavy_alert": [
        "heavy_alert_level_v2",
        "heavy_alert_level",
        "heavy_alert",
        "alert_level",
    ],
}

DEFAULT_FEATURES = [
    "lat",
    "lon",
    "month",
    "day_of_year",
    "season",
    "north_rain",
    "south_rain",
    "east_rain",
    "west_rain",
    "neighbor_rain_mean",
    "neighbor_rain_max",
    "neighbor_rain_sum",
    "rain_lag_1",
    "rain_lag_2",
    "rain_lag_3",
    "rain_lag_7",
    "rain_lag_14",
    "rain_roll_3",
    "rain_roll_7",
    "rain_roll_14",
    "temp_mean",
    "temp_max",
    "temp_min",
    "dewpoint_mean",
    "relative_humidity",
    "temp_mean_lag_1",
    "dewpoint_mean_lag_1",
    "relative_humidity_lag_1",
    "temp_mean_roll_7",
    "dewpoint_mean_roll_7",
    "relative_humidity_roll_7",
]


# ============================================================
# Helper functions
# ============================================================
def find_col(df: pd.DataFrame, key: str, required: bool = False) -> Optional[str]:
    for alias in COLUMN_ALIASES.get(key, [key]):
        if alias in df.columns:
            return alias
    if required:
        raise KeyError(f"Missing required column for {key}. Tried: {COLUMN_ALIASES.get(key, [key])}")
    return None


def normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_col = find_col(df, "date", required=True)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if date_col != "date":
        df = df.rename(columns={date_col: "date"})
    return df


def safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def clean_probability(value) -> float:
    try:
        value = float(value)
    except Exception:
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def format_pct(value) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{100 * float(value):.1f}%"


def format_mm(value) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.2f} mm"


def comparison_delta(new_value: float, old_value: float, suffix: str = "") -> str:
    new_value = 0.0 if pd.isna(new_value) else float(new_value)
    old_value = 0.0 if pd.isna(old_value) else float(old_value)
    delta = new_value - old_value
    return f"{delta:+.2f}{suffix}"


def classify_risk(predicted_rainfall: float, rain_probability: float = 0.0, heavy_probability: float = 0.0) -> str:
    predicted_rainfall = 0.0 if pd.isna(predicted_rainfall) else float(predicted_rainfall)
    rain_probability = 0.0 if pd.isna(rain_probability) else float(rain_probability)
    heavy_probability = 0.0 if pd.isna(heavy_probability) else float(heavy_probability)
    if heavy_probability >= 0.70 or predicted_rainfall >= 35:
        return "Severe Risk"
    if heavy_probability >= 0.50 or predicted_rainfall >= 15:
        return "High Risk"
    if predicted_rainfall >= 2.5 or rain_probability >= 0.50:
        return "Moderate Rain"
    return "No / Low Rain"


def heavy_alert_level(heavy_probability: float) -> str:
    heavy_probability = 0.0 if pd.isna(heavy_probability) else float(heavy_probability)
    if heavy_probability >= 0.70:
        return "🔴 Severe Heavy-Rain Alert"
    if heavy_probability >= 0.50:
        return "🟠 High Heavy-Rain Watch"
    if heavy_probability >= 0.30:
        return "🟡 Moderate Watch"
    return "🟢 No Heavy-Rain Alert"


# ============================================================
# Reactive ambience helpers
# ============================================================
SOUND_DIR = APP_DIR / "assets" / "sounds"
MANUAL_SOUND_OPTIONS = {
    "Calm": "calm.mp3",
    "Birds": "birds.mp3",
    "Light rain": "light_rain.mp3",
    "Rain + wind": "rain_wind.mp3",
    "Heavy rain": "heavy_rain.mp3",
    "Thunderstorm": "thunder.mp3",
    "Dry wind": "dry_wind.mp3",
}


def get_reactive_sound(predicted_rainfall_mm, risk_level, heatwave_risk=None):
    """Select a sound file name from prediction and risk inputs."""
    risk_level = str(risk_level).lower()

    if heatwave_risk is not None and str(heatwave_risk).lower() in ["high", "extreme", "yes", "true"]:
        return "dry_wind.mp3"

    if predicted_rainfall_mm >= 50 or "extreme" in risk_level or "flood" in risk_level:
        return "thunder.mp3"

    elif predicted_rainfall_mm >= 20 or "high" in risk_level:
        return "heavy_rain.mp3"

    elif predicted_rainfall_mm >= 5 or "moderate" in risk_level:
        return "rain_wind.mp3"

    elif predicted_rainfall_mm > 0:
        return "light_rain.mp3"

    else:
        return "calm.mp3"


def get_representative_risk_label(df: pd.DataFrame, risk_col: str) -> str:
    """Choose the most severe risk level present in the selected subset."""
    if df.empty or risk_col not in df.columns:
        return "No / Low Rain"
    for risk in reversed(RISK_ORDER):
        if (df[risk_col] == risk).any():
            return risk
    return str(df[risk_col].mode().iloc[0]) if not df[risk_col].mode().empty else "No / Low Rain"


def get_ambience_sound_file(sound_mode: str, manual_choice: Optional[str], predicted_rainfall_mm: float, risk_label: str, heatwave_risk=None) -> Optional[str]:
    if sound_mode == "Off":
        return None
    if sound_mode == "Manual":
        return MANUAL_SOUND_OPTIONS.get(manual_choice, "calm.mp3")
    return get_reactive_sound(predicted_rainfall_mm, risk_label, heatwave_risk)


def get_sound_path(sound_file: Optional[str]) -> Optional[Path]:
    if sound_file is None:
        return None
    return SOUND_DIR / sound_file


def sound_file_exists(sound_file: Optional[str]) -> bool:
    sound_path = get_sound_path(sound_file)
    return sound_path is not None and sound_path.exists()


AMBITION_ANIMATION_MAP = {
    "calm.mp3": {
        "label": "Clear / calm",
        "animation": "cloudy",
        "subtitle": "A calm soundscape with low predicted rainfall.",
    },
    "birds.mp3": {
        "label": "Birdsong in nature",
        "animation": "bird",
        "subtitle": "A gentle natural ambience for low-risk conditions.",
    },
    "light_rain.mp3": {
        "label": "Light rain",
        "animation": "light-rain",
        "subtitle": "Light drops and soft rainfall animation.",
    },
    "rain_wind.mp3": {
        "label": "Rain with wind",
        "animation": "rain-wind",
        "subtitle": "Steadier rain with wind movement.",
    },
    "heavy_rain.mp3": {
        "label": "Heavy rain",
        "animation": "heavy-rain",
        "subtitle": "Strong rainfall animation for a higher-risk event.",
    },
    "thunder.mp3": {
        "label": "Thunderstorm",
        "animation": "thunder",
        "subtitle": "Stormy conditions with thunder and heavier precipitation.",
    },
    "dry_wind.mp3": {
        "label": "Dry wind",
        "animation": "dry-wind",
        "subtitle": "Dry and windy ambience for heatwave or clear conditions.",
    },
}


def get_sidebar_ambience_sound_file(
    dashboard_df: Optional[pd.DataFrame],
    sound_mode: str,
    manual_choice: Optional[str],
    selected_date: Optional[str] = None,
) -> Optional[str]:
    if sound_mode == "Off":
        return None
    if sound_mode == "Manual":
        return MANUAL_SOUND_OPTIONS.get(manual_choice, "calm.mp3")
    if dashboard_df is None or dashboard_df.empty:
        return "calm.mp3"

    cols = get_prediction_columns(dashboard_df)
    if cols["pred_rain"] is None or cols["risk"] is None:
        return "calm.mp3"

    if selected_date is None:
        available_dates = sorted(dashboard_df["date"].dt.date.unique())
        selected_date = available_dates[-1] if available_dates else None

    if selected_date is None:
        return "calm.mp3"

    day_df = dashboard_df[dashboard_df["date"].dt.date == selected_date].copy()
    if day_df.empty:
        day_df = dashboard_df.copy()

    day_df[cols["pred_rain"]]= safe_numeric(day_df[cols["pred_rain"]])
    predicted_mean = day_df[cols["pred_rain"]].mean() if not day_df.empty else 0.0
    risk_label = get_representative_risk_label(day_df, cols["risk"])
    return get_reactive_sound(predicted_mean, risk_label, heatwave_risk=None)


def get_ambience_animation_state(sound_file: Optional[str], enabled: bool) -> Dict[str, str]:
    if not enabled or sound_file is None:
        return {
            "label": "Ambience off",
            "animation": "cloudy",
            "subtitle": "Ambient visuals are muted. Use the sidebar or homepage button to restore sound.",
        }
    return AMBITION_ANIMATION_MAP.get(sound_file, AMBITION_ANIMATION_MAP["calm.mp3"])



BACKGROUND_URLS = {
    # Kept only for reference. The current app uses CSS-only lofi backgrounds so it never depends on remote video hotlinking.
    "calm": "",
    "cloudy": "",
    "light_rain": "",
    "moderate_rain": "",
    "heavy_rain": "",
    "thunderstorm": "",
    "dry_wind": "",
}


LOFI_STATE_LABELS = {
    "calm": "Calm Climate",
    "cloudy": "Cloudy",
    "light_rain": "Light Rain",
    "moderate_rain": "Rain With Wind",
    "heavy_rain": "Heavy Rain",
    "thunderstorm": "Thunderstorm",
    "dry_wind": "Dry Wind",
}


LOFI_GRADIENTS = {
    "calm": "linear-gradient(135deg, #102538 0%, #183d52 45%, #355d6d 100%)",
    "cloudy": "linear-gradient(135deg, #111f2e 0%, #263a4a 45%, #506476 100%)",
    "light_rain": "linear-gradient(135deg, #0f2235 0%, #1d4257 50%, #3d6778 100%)",
    "moderate_rain": "linear-gradient(135deg, #091727 0%, #15344a 50%, #2b5265 100%)",
    "heavy_rain": "linear-gradient(135deg, #050c16 0%, #10283d 55%, #203f52 100%)",
    "thunderstorm": "linear-gradient(135deg, #030711 0%, #101a2b 50%, #1d3145 100%)",
    "dry_wind": "linear-gradient(135deg, #2d1f16 0%, #7c5130 45%, #d1995a 100%)",
}


def _make_rain_lines(weather_state: str) -> str:
    counts = {
        "calm": 0,
        "cloudy": 0,
        "light_rain": 22,
        "moderate_rain": 42,
        "heavy_rain": 72,
        "thunderstorm": 88,
        "dry_wind": 0,
    }
    count = counts.get(weather_state, 18)
    drops = []
    for i in range(count):
        left = (i * 37) % 100
        delay = ((i * 0.17) % 2.2)
        duration = 1.8 if weather_state == "light_rain" else 1.15 if weather_state in ["moderate_rain", "heavy_rain"] else 0.95
        height = 42 if weather_state == "light_rain" else 62 if weather_state == "moderate_rain" else 86
        opacity = 0.28 if weather_state == "light_rain" else 0.42 if weather_state == "moderate_rain" else 0.58
        drops.append(
            f'<i class="rain-line" style="left:{left}%; animation-delay:{delay:.2f}s; animation-duration:{duration:.2f}s; height:{height}px; opacity:{opacity};"></i>'
        )
    return "\n".join(drops)


def _make_window_grid(cols: int = 4, rows: int = 5) -> str:
    cells = []
    for i in range(cols * rows):
        # Deterministic pattern so the city looks alive without JS/randomness.
        lit = " lit" if (i * 7) % 11 in [0, 1, 4] else ""
        cells.append(f'<span class="window{lit}"></span>')
    return "".join(cells)


def render_cute_lofi_background(weather_state: str) -> None:
    """Render a CSS-only lofi climate-station background that cannot cover the UI.

    This version does NOT inject fixed HTML layers. It paints the atmosphere on
    .stApp pseudo-elements and forces Streamlit content above the background.
    """
    weather_state = str(weather_state or "calm")
    weather_state = weather_state if weather_state in LOFI_GRADIENTS else "calm"
    label = LOFI_STATE_LABELS.get(weather_state, "Climate Ambience")
    gradient = LOFI_GRADIENTS.get(weather_state, LOFI_GRADIENTS["calm"])

    # State-specific intensity values.
    if weather_state == "light_rain":
        rain_opacity, rain_speed, storm_opacity = 0.22, "1.8s", 0.0
    elif weather_state == "moderate_rain":
        rain_opacity, rain_speed, storm_opacity = 0.34, "1.25s", 0.0
    elif weather_state == "heavy_rain":
        rain_opacity, rain_speed, storm_opacity = 0.48, "0.85s", 0.0
    elif weather_state == "thunderstorm":
        rain_opacity, rain_speed, storm_opacity = 0.56, "0.70s", 0.22
    else:
        rain_opacity, rain_speed, storm_opacity = 0.0, "1.8s", 0.0

    wind_opacity = 0.30 if weather_state == "dry_wind" else 0.0
    sun_opacity = 0.72 if weather_state in ["calm", "dry_wind"] else 0.24
    cloud_opacity = 0.42 if weather_state in ["calm", "cloudy", "light_rain"] else 0.58

    # A compact pure-CSS illustration: sky, clouds, moon/sun, city silhouettes,
    # rain/wind overlays, scanlines, and vignette. Because these are pseudo-elements
    # of .stApp with z-index:0, the actual Streamlit UI remains at z-index:10+.
    custom_css = f"""
    <style>
    .stApp {{
        position: relative !important;
        min-height: 100vh !important;
        overflow-x: hidden !important;
        background: {gradient} !important;
        color: #ebf7ff !important;
    }}

    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        background:
            radial-gradient(circle at 84% 16%, rgba(255, 226, 155, {sun_opacity}), rgba(255, 208, 132, 0.16) 4%, transparent 10%),
            radial-gradient(ellipse at 20% 17%, rgba(231, 246, 255, {cloud_opacity}), rgba(231, 246, 255, 0.18) 8%, transparent 18%),
            radial-gradient(ellipse at 54% 10%, rgba(231, 246, 255, {cloud_opacity * 0.82}), rgba(231, 246, 255, 0.16) 8%, transparent 18%),
            radial-gradient(ellipse at 56% 25%, rgba(231, 246, 255, {cloud_opacity * 0.65}), rgba(231, 246, 255, 0.12) 7%, transparent 16%),
            radial-gradient(circle at 18% 26%, rgba(95, 205, 255, 0.10), transparent 26%),
            radial-gradient(circle at 78% 18%, rgba(255, 196, 128, 0.08), transparent 24%);
        filter: blur(0.2px);
        animation: lofiCloudFloat 18s ease-in-out infinite alternate;
    }}

    .stApp::after {{
        content: "";
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        opacity: 1;
        background-image:
            repeating-linear-gradient(102deg, transparent 0 22px, rgba(195, 232, 255, {rain_opacity}) 23px 25px, transparent 26px 44px),
            repeating-linear-gradient(90deg, transparent 0 42px, rgba(255, 226, 166, {wind_opacity}) 43px 130px, transparent 131px 220px),
            linear-gradient(to top, rgba(4, 11, 18, 0.94) 0 15vh, transparent 15vh),
            linear-gradient(to top, rgba(8, 19, 31, 0.96) 0 23vh, transparent 23vh),
            linear-gradient(to top, rgba(8, 19, 31, 0.94) 0 31vh, transparent 31vh),
            linear-gradient(to top, rgba(8, 19, 31, 0.95) 0 25vh, transparent 25vh),
            linear-gradient(to top, rgba(8, 19, 31, 0.93) 0 34vh, transparent 34vh),
            linear-gradient(to top, rgba(8, 19, 31, 0.95) 0 20vh, transparent 20vh),
            linear-gradient(to top, rgba(8, 19, 31, 0.95) 0 28vh, transparent 28vh),
            radial-gradient(ellipse at 18% 88%, rgba(255, 185, 90, 0.18), transparent 26%),
            radial-gradient(circle at center, rgba(255,255,255,0.02), rgba(0,0,0,0.36)),
            linear-gradient(rgba(200, 235, 255, {storm_opacity}), rgba(200, 235, 255, 0)),
            repeating-linear-gradient(to bottom, rgba(255,255,255,0.09) 0px, rgba(255,255,255,0.09) 1px, transparent 2px, transparent 5px);
        background-size:
            115px 115px,
            360px 100vh,
            100vw 40vh,
            8vw 35vh,
            11vw 38vh,
            9vw 35vh,
            12vw 40vh,
            9vw 30vh,
            10vw 34vh,
            100vw 24vh,
            100vw 100vh,
            100vw 100vh,
            100vw 100vh;
        background-position:
            0 -20vh,
            0 0,
            0 bottom,
            8vw bottom,
            18vw bottom,
            31vw bottom,
            42vw bottom,
            56vw bottom,
            68vw bottom,
            0 bottom,
            center,
            center,
            0 0;
        background-repeat:
            repeat,
            repeat-x,
            no-repeat,
            no-repeat,
            no-repeat,
            no-repeat,
            no-repeat,
            no-repeat,
            no-repeat,
            no-repeat,
            no-repeat,
            no-repeat,
            repeat;
        animation: lofiRainFall {rain_speed} linear infinite, lofiLightning 7s linear infinite;
        mix-blend-mode: normal;
    }}

    /* Keep the actual Streamlit UI above the CSS-painted background. */
    [data-testid="stAppViewContainer"],
    section.main,
    .main,
    .block-container,
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"],
    [data-testid="element-container"] {{
        position: relative !important;
        z-index: 10 !important;
        background: transparent !important;
    }}

    [data-testid="stHeader"] {{
        position: relative !important;
        z-index: 50 !important;
        background: rgba(5, 12, 20, 0.18) !important;
        backdrop-filter: blur(10px);
    }}

    [data-testid="stSidebar"] {{
        position: relative !important;
        z-index: 60 !important;
        background: linear-gradient(180deg, rgba(5, 12, 22, 0.94), rgba(10, 26, 42, 0.90)) !important;
        border-right: 1px solid rgba(180, 225, 255, 0.12) !important;
    }}

    .hero-card,
    .glass-card,
    .dashboard-card,
    .info-card,
    .metric-card,
    .prediction-card,
    .note-box,
    .good-box,
    .bad-box,
    [data-testid="stMetric"],
    .stTabs {{
        position: relative !important;
        z-index: 20 !important;
    }}

    .lofi-status-pill-fixed {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        border-radius: 999px;
        padding: 0.45rem 0.85rem;
        margin-bottom: 0.75rem;
        background: rgba(8, 18, 30, 0.62);
        border: 1px solid rgba(180, 225, 255, 0.18);
        backdrop-filter: blur(12px);
        color: rgba(235, 247, 255, 0.92) !important;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }}

    @keyframes lofiRainFall {{
        from {{ background-position: 0 -20vh, 0 0, 0 bottom, 8vw bottom, 18vw bottom, 31vw bottom, 42vw bottom, 56vw bottom, 68vw bottom, 0 bottom, center, center, 0 0; }}
        to {{ background-position: -70px 110vh, 360px 0, 0 bottom, 8vw bottom, 18vw bottom, 31vw bottom, 42vw bottom, 56vw bottom, 68vw bottom, 0 bottom, center, center, 0 0; }}
    }}

    @keyframes lofiCloudFloat {{
        0% {{ transform: translateX(-1.2vw) translateY(0); }}
        100% {{ transform: translateX(1.8vw) translateY(0.8vh); }}
    }}

    @keyframes lofiLightning {{
        0%, 88%, 100% {{ filter: brightness(1); }}
        89% {{ filter: brightness({1 + storm_opacity}); }}
        90% {{ filter: brightness(1); }}
        91% {{ filter: brightness({1 + storm_opacity * 0.7}); }}
        92% {{ filter: brightness(1); }}
    }}

    @media (prefers-reduced-motion: reduce) {{
        .stApp::before,
        .stApp::after {{
            animation-duration: 0.001ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.001ms !important;
        }}
    }}
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown(
        f'<div class="lofi-status-pill-fixed">Current ambience: {label}</div>',
        unsafe_allow_html=True,
    )


def get_video_mime_type(video_url):
    # Kept for compatibility with older code. CSS-only background no longer uses remote videos.
    video_url = str(video_url).lower()
    if video_url.endswith(".webm"):
        return "video/webm"
    if video_url.endswith(".mp4"):
        return "video/mp4"
    return "video/mp4"


def get_fallback_gradient(weather_state):
    weather_state = str(weather_state or "calm")
    return LOFI_GRADIENTS.get(weather_state, LOFI_GRADIENTS["calm"])


def render_css_weather_background(weather_state):
    render_cute_lofi_background(weather_state)


def render_remote_lofi_background(weather_state):
    # Remote Pixabay links were unreliable in <video> backgrounds, so this now renders a deploy-safe CSS lofi scene.
    render_cute_lofi_background(weather_state)


def map_animation_to_background_state(animation_class: str) -> str:
    mapping = {
        "cloudy": "cloudy",
        "bird": "calm",
        "light-rain": "light_rain",
        "rain-wind": "moderate_rain",
        "heavy-rain": "heavy_rain",
        "thunder": "thunderstorm",
        "dry-wind": "dry_wind",
        "calm": "calm",
    }
    return mapping.get(animation_class, "calm")


def get_prediction_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "date": find_col(df, "date", required=True),
        "lat": find_col(df, "lat", required=True),
        "lon": find_col(df, "lon", required=True),
        "pred_rain": find_col(df, "pred_rain"),
        "rain_prob": find_col(df, "rain_prob"),
        "heavy_prob": find_col(df, "heavy_prob"),
        "actual": find_col(df, "actual"),
        "risk": find_col(df, "risk"),
        "heavy_alert": find_col(df, "heavy_alert"),
    }


def standardize_dashboard_df(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_date_column(df)
    cols = get_prediction_columns(df)
    df = df.copy()
    if cols["lat"] != "lat":
        df = df.rename(columns={cols["lat"]: "lat"})
    if cols["lon"] != "lon":
        df = df.rename(columns={cols["lon"]: "lon"})
    cols = get_prediction_columns(df)
    if cols["rain_prob"] is None:
        df["rain_probability_v2"] = np.nan
        cols["rain_prob"] = "rain_probability_v2"
    if cols["pred_rain"] is None:
        df["predicted_rainfall_soft_hybrid_v2"] = np.nan
        cols["pred_rain"] = "predicted_rainfall_soft_hybrid_v2"
    if cols["heavy_prob"] is None:
        df["heavy_rain_probability_v2"] = np.nan
        cols["heavy_prob"] = "heavy_rain_probability_v2"
    df[cols["rain_prob"]]= safe_numeric(df[cols["rain_prob"]])
    df[cols["pred_rain"]]= safe_numeric(df[cols["pred_rain"]])
    df[cols["heavy_prob"]]= safe_numeric(df[cols["heavy_prob"]])
    if cols["risk"] is None:
        df["final_dashboard_risk_v2"] = df.apply(
            lambda row: classify_risk(
                row[cols["pred_rain"]],
                row[cols["rain_prob"]],
                row[cols["heavy_prob"]],
            ),
            axis=1,
        )
        cols["risk"] = "final_dashboard_risk_v2"
    if cols["heavy_alert"] is None:
        df["heavy_alert_level_v2"] = df[cols["heavy_prob"]].apply(heavy_alert_level)
        cols["heavy_alert"] = "heavy_alert_level_v2"
    return df


def predict_proba_safe(model, X, model_name="model"):
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            if hasattr(proba, "ndim") and proba.ndim == 2 and proba.shape[1] > 1:
                return float(proba[0, 1])
            return float(proba[0])

        preds = model.predict(X)
        return float(preds[0])

    except Exception as e:
        st.error(f"{model_name} prediction failed: {e}")
        st.write(f"{model_name} input shape:", X.shape)
        st.write(f"{model_name} input columns:", list(X.columns))
        st.write(f"{model_name} first input row:")
        st.dataframe(X.head(1), use_container_width=True)
        return 0.0


def make_feature_frame(row: pd.Series, feature_cols: List[str]) -> pd.DataFrame:
    values = {col: [float(row[col]) if col in row.index and not pd.isna(row[col]) else 0.0] for col in feature_cols}
    return pd.DataFrame(values)[feature_cols]


def predict_with_models(
    row: pd.Series,
    feature_cols: List[str],
    rain_model,
    amount_model,
    heavy_model,
) -> Dict[str, float]:
    X = make_feature_frame(row, feature_cols)
    rain_probability = clean_probability(
        predict_proba_safe(rain_model, X, "Rain probability model")
    )
    rainfall_amount = 0.0
    try:
        rainfall_amount = float(amount_model.predict(X)[0])
    except Exception as e:
        st.error(f"Rainfall amount regressor prediction failed: {e}")
        st.write("Rainfall amount regressor input shape:", X.shape)
        st.write("Rainfall amount regressor input columns:", list(X.columns))
        st.write("Rainfall amount regressor first input row:")
        st.dataframe(X.head(1), use_container_width=True)
        rainfall_amount = 0.0
    rainfall_amount = max(0.0, rainfall_amount)
    heavy_probability = clean_probability(
        predict_proba_safe(heavy_model, X, "Heavy-rain detector model")
    )
    return {
        "rain_probability": rain_probability,
        "rainfall_amount_raw": rainfall_amount,
        "predicted_rainfall": max(0.0, rain_probability * rainfall_amount),
        "heavy_probability": heavy_probability,
    }


def apply_simulation_changes(
    row: pd.Series,
    temp_delta: float,
    dewpoint_delta: float,
    humidity_delta: float,
    recent_rain_multiplier: float,
    neighbor_rain_multiplier: float,
) -> pd.Series:
    row = row.copy()
    temp_cols = [
        "temp_mean",
        "temp_max",
        "temp_min",
        "temp_mean_lag_1",
        "temp_mean_roll_7",
    ]
    dew_cols = [
        "dewpoint_mean",
        "dewpoint_mean_lag_1",
        "dewpoint_mean_roll_7",
    ]
    humidity_cols = [
        "relative_humidity",
        "relative_humidity_lag_1",
        "relative_humidity_roll_7",
    ]
    recent_rain_cols = [
        "rain_lag_1",
        "rain_lag_2",
        "rain_lag_3",
        "rain_lag_7",
        "rain_lag_14",
        "rain_roll_3",
        "rain_roll_7",
        "rain_roll_14",
    ]
    neighbor_rain_cols = [
        "north_rain",
        "south_rain",
        "east_rain",
        "west_rain",
        "neighbor_rain_mean",
        "neighbor_rain_max",
        "neighbor_rain_sum",
    ]
    for col in temp_cols:
        if col in row.index:
            row[col] = float(row[col] if not pd.isna(row[col]) else 0.0) + temp_delta
    for col in dew_cols:
        if col in row.index:
            row[col] = float(row[col] if not pd.isna(row[col]) else 0.0) + dewpoint_delta
    for col in humidity_cols:
        if col in row.index:
            row[col] = float(np.clip(float(row[col] if not pd.isna(row[col]) else 0.0) + humidity_delta, 0, 100))
    for col in recent_rain_cols:
        if col in row.index:
            row[col] = max(0.0, float(row[col] if not pd.isna(row[col]) else 0.0) * recent_rain_multiplier)
    for col in neighbor_rain_cols:
        if col in row.index:
            row[col] = max(0.0, float(row[col] if not pd.isna(row[col]) else 0.0) * neighbor_rain_multiplier)
    return row


def make_default_v1_v2_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Metric": "MAE", "V1": 3.4139, "V2": 3.0757},
            {"Metric": "RMSE", "V1": 8.6830, "V2": 8.3608},
            {"Metric": "R²", "V1": 0.1726, "V2": 0.2329},
            {"Metric": "Risk Accuracy", "V1": 0.63, "V2": 0.69},
            {"Metric": "Weighted F1", "V1": 0.68, "V2": 0.74},
            {"Metric": "Macro F1", "V1": 0.30, "V2": 0.34},
        ]
    )


def make_default_heavy_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Metric": "Target", "V2": "target_rainfall >= 35 mm"},
            {"Metric": "ROC AUC", "V2": 0.9181},
            {"Metric": "Average Precision", "V2": 0.1538},
            {"Metric": "Heavy precision", "V2": 0.10},
            {"Metric": "Heavy recall", "V2": 0.76},
            {"Metric": "Heavy F1", "V2": 0.18},
            {"Metric": "Accuracy", "V2": 0.88},
        ]
    )


def load_csv_or_none(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> pd.DataFrame:
    path = DIRS["data"] / DASHBOARD_CSV
    if not path.exists():
        raise FileNotFoundError(f"Dashboard CSV not found at: {path}")
    return standardize_dashboard_df(pd.read_csv(path))


@st.cache_data(show_spinner=False)
def load_simulator_data() -> pd.DataFrame:
    path = DIRS["data"] / SIMULATOR_CSV
    if not path.exists():
        raise FileNotFoundError(f"Simulator CSV not found at: {path}")
    return normalize_date_column(pd.read_csv(path))


@st.cache_resource(show_spinner=False)
def load_models():
    rain_path = DIRS["models"] / RAIN_MODEL_FILE
    amount_path = DIRS["models"] / AMOUNT_MODEL_FILE
    heavy_path = DIRS["models"] / HEAVY_MODEL_FILE
    missing = [str(p) for p in (rain_path, amount_path, heavy_path) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing model files:\n" + "\n".join(missing))
    return joblib.load(rain_path), joblib.load(amount_path), joblib.load(heavy_path)


@st.cache_data(show_spinner=False)
def load_feature_columns(sample_columns: List[str]) -> List[str]:
    config_path = DIRS["docs"] / FEATURE_CONFIG_FILE
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            if isinstance(config, dict):
                for value in config.values():
                    if isinstance(value, list) and all(isinstance(x, str) for x in value):
                        return [c for c in value if c in sample_columns]
        except Exception:
            pass
    available = [c for c in DEFAULT_FEATURES if c in sample_columns]
    return available if available else [c for c in sample_columns if c not in ["date", "lat", "lon"]][:30]


@st.cache_data(show_spinner=False)
def load_metrics_tables():
    v1_v2 = load_csv_or_none(DIRS["docs"] / V1_V2_METRICS_FILE)
    heavy = load_csv_or_none(DIRS["docs"] / HEAVY_METRICS_FILE)
    if v1_v2 is None or "Metric" not in v1_v2.columns:
        v1_v2 = make_default_v1_v2_metrics()
    if heavy is None or "Metric" not in heavy.columns:
        heavy = make_default_heavy_metrics()
    return v1_v2, heavy


def render_metric_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="small-muted">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("## 🌦️ Project controls")
    st.markdown(
        """
        <div class="small-muted">
        This prototype uses IMD rainfall as ground truth and NASA POWER climate inputs. It is built for judge-friendly visualization of risk, not for operational forecasting.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.markdown("### 📁 Data paths")
    st.caption(f"Data: `{DIRS['data']}`")
    st.caption(f"Models: `{DIRS['models']}`")
    st.caption(f"Docs: `{DIRS['docs']}`")
    st.divider()
    st.markdown("### ⚠️ Limitation")
    st.info(
        "Heavy rainfall is rare and imbalanced, so exact amount prediction remains difficult. "
        "We use a separate heavy-rain detector as an early-warning layer."
    )


# ============================================================
# Load data
# ============================================================
dashboard_df = None
simulation_df = None
model_bundle = None
load_errors = []

try:
    dashboard_df = load_dashboard_data()
except Exception as e:
    load_errors.append(f"Dashboard data load failed: {e}")

try:
    simulation_df = load_simulator_data()
except Exception as e:
    load_errors.append(f"Simulator data load failed: {e}")

try:
    model_bundle = load_models()
except Exception as e:
    load_errors.append(f"Model load failed: {e}")

metrics_v1_v2, metrics_heavy = load_metrics_tables()

# Sidebar ambience controls require loaded forecast data so that audio and animation state can react globally.
with st.sidebar:
    st.divider()
    st.markdown("### Reactive Climate Ambience")
    st.caption("Soundscape reacts to predicted rainfall and risk level.")
    if "weather_sound_enabled" not in st.session_state:
        st.session_state["weather_sound_enabled"] = True
    toggle_func = getattr(st, "toggle", st.checkbox)
    weather_sound_enabled = toggle_func(
        "Enable reactive weather sounds",
        value=st.session_state["weather_sound_enabled"],
        key="weather_sound_sidebar_toggle",
    )
    if weather_sound_enabled != st.session_state["weather_sound_enabled"]:
        st.session_state["weather_sound_enabled"] = weather_sound_enabled
    sound_mode = st.radio(
        "Sound mode",
        ["Auto-reactive", "Manual", "Off"],
        index=0,
        key="sound_mode",
    )
    manual_sound_choice = None
    if sound_mode == "Manual":
        manual_sound_choice = st.selectbox(
            "Select manual ambience",
            list(MANUAL_SOUND_OPTIONS.keys()),
            index=0,
            key="manual_sound_choice",
        )
    else:
        st.caption("Auto-reactive mode uses selected dashboard prediction and risk values.")

    if weather_sound_enabled and sound_mode != "Off":
        selected_date_key = st.session_state.get("selected_date")
        selected_sound_file = get_sidebar_ambience_sound_file(
            dashboard_df,
            sound_mode,
            manual_sound_choice,
            selected_date=selected_date_key,
        )
        sound_path = get_sound_path(selected_sound_file)
        if sound_path is not None and sound_path.exists():
            st.audio(
                str(sound_path),
                format="audio/mpeg",
                loop=True,
                autoplay=True,
            )
        else:
            st.caption(f"Reactive ambience file missing: assets/sounds/{selected_sound_file}")


# Calculate the current ambience before rendering the visible UI.
weather_sound_file = get_sidebar_ambience_sound_file(
    dashboard_df,
    st.session_state.get("sound_mode", "Auto-reactive"),
    st.session_state.get("manual_sound_choice", "Calm"),
    selected_date=st.session_state.get("selected_date"),
)
weather_animation = get_ambience_animation_state(
    weather_sound_file,
    st.session_state.get("weather_sound_enabled", True),
)
weather_state = map_animation_to_background_state(weather_animation["animation"])
render_remote_lofi_background(weather_state)

# ============================================================
# Main interface
# ============================================================
st.markdown(
    """
    <div class="hero-card">
        <div class="station-label">LIVE CLIMATE MONITORING STATION</div>
        <div class="hero-title">AI Rainfall-Risk Digital Twin</div>
        <div class="hero-subtitle">
            Punjab + Haryana prototype using IMD rainfall ground truth and NASA POWER meteorological inputs.
            The system predicts next-day rain probability, estimated rainfall amount, rainfall-risk category,
            and heavy-rain early-warning probability.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if load_errors:
    with st.expander("Load status and fallback notes"):
        for message in load_errors:
            st.warning(message)


tabs = st.tabs(["🗺️ Prediction Dashboard", "🏠 Home / Overview", "🧪 What-if Simulator", "📊 V1 vs V2 Results"])


# ============================================================
# Home tab
# ============================================================
with tabs[1]:
    st.markdown("### Project overview")
    st.write(
        "This rain-risk digital twin prototype combines historical IMD rainfall targets with NASA POWER climate features to score tomorrow's rainfall risk in Punjab and Haryana. "
        "It is built for insight, storyboarding, and early-warning visualization rather than exact millimeter forecasting."
    )
    if "weather_sound_enabled" not in st.session_state:
        st.session_state["weather_sound_enabled"] = True

    def toggle_weather_sound():
        st.session_state["weather_sound_enabled"] = not st.session_state["weather_sound_enabled"]

    sound_button_label = (
        "🔇 Mute Weather Sound"
        if st.session_state["weather_sound_enabled"]
        else "🔊 Enable Weather Sound"
    )
    st.button(
        sound_button_label,
        key="weather_sound_toggle_btn",
        on_click=toggle_weather_sound,
    )
    homepage_sound_file = get_sidebar_ambience_sound_file(
        dashboard_df,
        st.session_state.get("sound_mode", "Auto-reactive"),
        st.session_state.get("manual_sound_choice", "Calm"),
        selected_date=st.session_state.get("selected_date"),
    )
    homepage_animation = get_ambience_animation_state(
        homepage_sound_file,
        st.session_state["weather_sound_enabled"],
    )
    st.caption(f"Current ambience: {homepage_animation['label']}")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="info-card">
                <h4>📍 Region</h4>
                <p>Punjab + Haryana</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="info-card">
                <h4>🌐 Data sources</h4>
                <p>IMD rainfall + NASA POWER climate variables</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="info-card">
                <h4>🧠 Model outputs</h4>
                <p>Rain probability · Rain amount · Risk category · Heavy alert</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Model pipeline")
    p1, p2, p3 = st.columns([1.2, 0.9, 1.2])
    p1.markdown(
        """
        <div class="mini-card">
            <strong>IMD + NASA POWER</strong><br>
            Ground truth rainfall and meteorological input features.
        </div>
        """,
        unsafe_allow_html=True,
    )
    p2.markdown(
        """
        <div class="mini-card">
            <strong>V2 AI models</strong><br>
            Rain probability, amount regression, and heavy-rain detector.
        </div>
        """,
        unsafe_allow_html=True,
    )
    p3.markdown(
        """
        <div class="mini-card">
            <strong>Dashboard outputs</strong><br>
            Rain risk, predicted rainfall, and heavy warning.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Key V2 improvements")
    improvements = [
        ("MAE", "3.41 → 3.08"),
        ("RMSE", "8.68 → 8.36"),
        ("R²", "0.17 → 0.23"),
        ("Risk Accuracy", "0.63 → 0.69"),
        ("Weighted F1", "0.68 → 0.74"),
    ]
    cols = st.columns(5)
    for col, (label, value) in zip(cols, improvements):
        col.markdown(
            f"""
            <div class=\"metric-card\">
                <div class=\"metric-label\">{label}</div>
                <div class=\"metric-value\">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Why this prototype matters")
    st.write(
        "This dashboard is designed for judge-friendly storytelling: it highlights how AI can fuse climate inputs with rainfall history to generate next-day risk signals and early warnings across Punjab and Haryana."
    )


# ============================================================
# Prediction Dashboard tab
# ============================================================
with tabs[0]:
    st.markdown("### Next-day rainfall-risk map")
    if dashboard_df is None:
        st.error("Prediction dashboard data is unavailable. Please check that the dashboard CSV exists under DEV_HANDOFF/data.")
    else:
        cols = get_prediction_columns(dashboard_df)
        dashboard_df = dashboard_df.copy()
        dashboard_df[cols["rain_prob"]]= safe_numeric(dashboard_df[cols["rain_prob"]])
        dashboard_df[cols["pred_rain"]]= safe_numeric(dashboard_df[cols["pred_rain"]])
        dashboard_df[cols["heavy_prob"]]= safe_numeric(dashboard_df[cols["heavy_prob"]])
        dashboard_df[cols["risk"]]= dashboard_df[cols["risk"]].fillna("Unknown")
        dashboard_df[cols["heavy_alert"]]= dashboard_df[cols["heavy_alert"]].fillna("No Heavy-Rain Alert")

        available_dates = sorted(dashboard_df["date"].dt.date.unique())
        selected_date = st.selectbox(
            "Select forecast date",
            available_dates,
            index=len(available_dates) - 1,
            key="selected_date",
        )
        show_only_alerts = st.checkbox("Show only high/severe risk points", value=False)
        min_heavy_prob = st.slider(
            "Minimum heavy-rain probability",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            help="This filters existing map points. It does not recompute rainfall predictions.",
        )

        st.caption(
            "Note: These controls filter already-generated predictions. "
            "Use the What-if Simulator tab to change climate inputs and rerun model predictions."
        )

        day_df = dashboard_df[dashboard_df["date"].dt.date == selected_date].copy()
        if show_only_alerts:
            day_df = day_df[day_df[cols["risk"]].isin(["High Risk", "Severe Risk"])]
        day_df = day_df[day_df[cols["heavy_prob"]] >= min_heavy_prob]

        avg_rain_probability = day_df[cols["rain_prob"]].mean() if not day_df.empty else np.nan
        avg_predicted_rainfall = day_df[cols["pred_rain"]].mean() if not day_df.empty else np.nan
        max_heavy_probability = day_df[cols["heavy_prob"]].max() if not day_df.empty else np.nan
        high_severe_points = int(day_df[cols["risk"]].isin(["High Risk", "Severe Risk"]).sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f"""
            <div class=\"metric-card\">
                <div class=\"metric-label\">Avg rain probability</div>
                <div class=\"metric-value\">{format_pct(avg_rain_probability)}</div>
            </div>
            """, unsafe_allow_html=True)
        c2.markdown(f"""
            <div class=\"metric-card\">
                <div class=\"metric-label\">Avg predicted rainfall</div>
                <div class=\"metric-value\">{format_mm(avg_predicted_rainfall)}</div>
            </div>
            """, unsafe_allow_html=True)
        c3.markdown(f"""
            <div class=\"metric-card\">
                <div class=\"metric-label\">Max heavy-rain probability</div>
                <div class=\"metric-value\">{format_pct(max_heavy_probability)}</div>
            </div>
            """, unsafe_allow_html=True)
        c4.markdown(f"""
            <div class=\"metric-card\">
                <div class=\"metric-label\">High / severe points</div>
                <div class=\"metric-value\">{high_severe_points}</div>
            </div>
            """, unsafe_allow_html=True)

        if day_df.empty:
            st.warning(
                "No points match the selected date and filters. "
                "Lower the minimum heavy-rain probability slider or uncheck high/severe-only filter."
            )
        else:
            day_df["_map_size"] = np.where(
                day_df[cols["risk"]].isin(["Severe Risk"]),
                14,
                np.where(
                    day_df[cols["risk"]].isin(["High Risk"]),
                    11,
                    np.where(
                        day_df[cols["risk"]].isin(["Moderate Rain"]),
                        8,
                        4
                    )
                )
            )
            hover_data = {
                cols["lat"]: ":.3f",
                cols["lon"]: ":.3f",
                cols["pred_rain"]: ":.2f",
                cols["rain_prob"]: ":.2f",
                cols["heavy_prob"]: ":.2f",
                cols["risk"]: True,
                cols["heavy_alert"]: True,
            }
            if cols["actual"] and cols["actual"] in day_df.columns:
                hover_data[cols["actual"]] = ":.2f"

            fig = px.scatter_mapbox(
                day_df,
                lat=cols["lat"],
                lon=cols["lon"],
                color=cols["risk"],
                size="_map_size",
                size_max=14,
                color_discrete_map=RISK_COLOR_MAP,
                category_orders={cols["risk"]: RISK_ORDER},
                hover_data=hover_data,
                zoom=6,
                height=620,
                center={"lat": float(day_df[cols["lat"]].mean()), "lon": float(day_df[cols["lon"]].mean())},
            )
            fig.update_layout(
                mapbox_style="open-street-map",
                margin={"l": 0, "r": 0, "t": 20, "b": 0},
                legend_title_text="Risk category",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            fig.update_traces(
                marker=dict(
                    opacity=0.45
                )
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Show top risk grid points"):
                columns_to_show = ["date", cols["lat"], cols["lon"], cols["pred_rain"], cols["rain_prob"], cols["heavy_prob"], cols["risk"], cols["heavy_alert"]]
                if cols["actual"] and cols["actual"] in day_df.columns:
                    columns_to_show.append(cols["actual"])
                columns_to_show = [c for c in columns_to_show if c in day_df.columns]
                st.dataframe(
                    day_df.sort_values(cols["heavy_prob"], ascending=False)[columns_to_show].head(30),
                    use_container_width=True,
                    hide_index=True,
                )


# ============================================================
# What-if Simulator tab
# ============================================================
with tabs[2]:
    st.markdown("### Model-based what-if simulator")
    st.markdown(
        """
        <div class="note-box">
        <strong>Model-based what-if simulator, not a physics-based climate simulator.</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if simulation_df is None or model_bundle is None:
        st.error("Simulator cannot run because the simulator dataset or models are unavailable.")
    else:
        simulation_df = simulation_df.copy()
        lat_col = find_col(simulation_df, "lat", required=True)
        lon_col = find_col(simulation_df, "lon", required=True)
        if lat_col != "lat":
            simulation_df = simulation_df.rename(columns={lat_col: "lat"})
        if lon_col != "lon":
            simulation_df = simulation_df.rename(columns={lon_col: "lon"})

        simulation_df["month"] = simulation_df["date"].dt.month
        simulation_df["day_of_year"] = simulation_df["date"].dt.dayofyear
        simulation_df["season"] = ((simulation_df["month"] % 12) // 3).astype(int)

        feature_columns = load_feature_columns(list(simulation_df.columns))
        with st.expander("Debug: simulator feature columns"):
            st.write("Number of loaded feature columns:", len(feature_columns))
            st.write("Loaded feature columns:", feature_columns)
            st.write("Simulator dataframe shape:", simulation_df.shape)
            st.write("Simulator dataframe columns:", list(simulation_df.columns))
        missing_features = [col for col in feature_columns if col not in simulation_df.columns]
        if missing_features:
            st.error("Some model feature columns are missing from the simulator dataset.")
            st.write("Missing features:", missing_features)
        sim_dates = sorted(simulation_df["date"].dt.date.unique())
        sim_date = st.selectbox("Select baseline date", sim_dates, index=len(sim_dates) - 1)
        sim_day_df = simulation_df[simulation_df["date"].dt.date == sim_date].copy()
        sim_day_df["grid_label"] = sim_day_df["lat"].round(3).astype(str) + ", " + sim_day_df["lon"].round(3).astype(str)
        location_label = st.selectbox("Select grid point (lat, lon)", sim_day_df["grid_label"].tolist())
        base_row = sim_day_df[sim_day_df["grid_label"] == location_label].iloc[0]

        st.markdown("#### Scenario controls")
        a, b, c = st.columns(3)
        with a:
            temp_delta = st.slider("Temperature change (°C)", -5.0, 5.0, 0.0, 0.5)
        with b:
            dewpoint_delta = st.slider("Dewpoint change (°C)", -5.0, 5.0, 0.0, 0.5)
        with c:
            humidity_delta = st.slider("Humidity change (%)", -30.0, 30.0, 0.0, 2.5)
        d, e = st.columns(2)
        with d:
            recent_rain_multiplier = st.slider("Recent rainfall multiplier", 0.0, 3.0, 1.0, 0.1)
        with e:
            neighbor_rain_multiplier = st.slider("Neighbor rainfall multiplier", 0.0, 3.0, 1.0, 0.1)

        rain_model, amount_model, heavy_model = model_bundle
        baseline_preds = predict_with_models(base_row, feature_columns, rain_model, amount_model, heavy_model)
        scenario_row = apply_simulation_changes(
            base_row,
            temp_delta=temp_delta,
            dewpoint_delta=dewpoint_delta,
            humidity_delta=humidity_delta,
            recent_rain_multiplier=recent_rain_multiplier,
            neighbor_rain_multiplier=neighbor_rain_multiplier,
        )
        changed_model_features = []
        for col in feature_columns:
            if col in base_row.index and col in scenario_row.index:
                old = base_row[col]
                new = scenario_row[col]
                if pd.isna(old) and pd.isna(new):
                    continue
                if old != new:
                    try:
                        delta = float(new) - float(old)
                    except Exception:
                        delta = None
                    changed_model_features.append(
                        {
                            "Feature": col,
                            "Baseline": old,
                            "Scenario": new,
                            "Delta": delta,
                        }
                    )
        with st.expander("Debug: changed model features"):
            if changed_model_features:
                st.dataframe(pd.DataFrame(changed_model_features), use_container_width=True, hide_index=True)
            else:
                st.warning("No actual model features changed. The sliders may not correspond to the trained model feature columns.")
        scenario_preds = predict_with_models(scenario_row, feature_columns, rain_model, amount_model, heavy_model)

        baseline_risk = classify_risk(
            baseline_preds["predicted_rainfall"],
            baseline_preds["rain_probability"],
            baseline_preds["heavy_probability"],
        )
        scenario_risk = classify_risk(
            scenario_preds["predicted_rainfall"],
            scenario_preds["rain_probability"],
            scenario_preds["heavy_probability"],
        )
        baseline_alert = heavy_alert_level(baseline_preds["heavy_probability"])
        scenario_alert = heavy_alert_level(scenario_preds["heavy_probability"])

        st.markdown("#### Baseline vs scenario")
        comparison = pd.DataFrame(
            [
                {
                    "Output": "Rain probability",
                    "Baseline": format_pct(baseline_preds["rain_probability"]),
                    "Scenario": format_pct(scenario_preds["rain_probability"]),
                    "Change": comparison_delta(
                        scenario_preds["rain_probability"] * 100,
                        baseline_preds["rain_probability"] * 100,
                        " pp",
                    ),
                },
                {
                    "Output": "Predicted rainfall",
                    "Baseline": format_mm(baseline_preds["predicted_rainfall"]),
                    "Scenario": format_mm(scenario_preds["predicted_rainfall"]),
                    "Change": comparison_delta(
                        scenario_preds["predicted_rainfall"],
                        baseline_preds["predicted_rainfall"],
                        " mm",
                    ),
                },
                {
                    "Output": "Heavy-rain probability",
                    "Baseline": format_pct(baseline_preds["heavy_probability"]),
                    "Scenario": format_pct(scenario_preds["heavy_probability"]),
                    "Change": comparison_delta(
                        scenario_preds["heavy_probability"] * 100,
                        baseline_preds["heavy_probability"] * 100,
                        " pp",
                    ),
                },
                {
                    "Output": "Final dashboard risk",
                    "Baseline": baseline_risk,
                    "Scenario": scenario_risk,
                    "Change": "Changed" if baseline_risk != scenario_risk else "No change",
                },
                {
                    "Output": "Heavy alert",
                    "Baseline": baseline_alert,
                    "Scenario": scenario_alert,
                    "Change": "Changed" if baseline_alert != scenario_alert else "No change",
                },
            ]
        )
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        cols_small = st.columns(4)
        cols_small[0].markdown(
            f"""
            <div class=\"metric-card\">
                <div class=\"metric-label\">Scenario rain probability</div>
                <div class=\"metric-value\">{format_pct(scenario_preds['rain_probability'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cols_small[1].markdown(
            f"""
            <div class=\"metric-card\">
                <div class=\"metric-label\">Scenario rainfall</div>
                <div class=\"metric-value\">{format_mm(scenario_preds['predicted_rainfall'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cols_small[2].markdown(
            f"""
            <div class=\"metric-card\">
                <div class=\"metric-label\">Scenario heavy probability</div>
                <div class=\"metric-value\">{format_pct(scenario_preds['heavy_probability'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cols_small[3].markdown(
            f"""
            <div class=\"metric-card\">
                <div class=\"metric-label\">Scenario risk</div>
                <div class=\"metric-value\">{scenario_risk}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        changed_features = []
        for col in feature_columns:
            if col in base_row.index and col in scenario_row.index:
                old = base_row[col]
                new = scenario_row[col]
                if pd.isna(old) and pd.isna(new):
                    continue
                if old != new:
                    changed_features.append(
                        {
                            "Feature": col,
                            "Baseline": old,
                            "Scenario": new,
                            "Delta": comparison_delta(new, old),
                        }
                    )
        if changed_features:
            with st.expander("Inspect changed input features"):
                st.dataframe(pd.DataFrame(changed_features), use_container_width=True, hide_index=True)
        else:
            st.info("No input features changed because the selected baseline row did not contain matching model features.")


# ============================================================
# V1 vs V2 Results tab
# ============================================================
with tabs[3]:
    st.markdown("### V1 vs V2 model results")
    st.markdown(
        """
        <div class="good-box">
        We built a Punjab-Haryana prototype of an AI-powered rainfall-risk digital twin. It uses IMD rainfall as the ground-truth target and NASA POWER meteorological variables as climate inputs. The system predicts next-day rain probability, estimated rainfall amount, rainfall-risk category, and heavy-rain early-warning probability.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### V1 vs V2 comparison")
    st.dataframe(metrics_v1_v2, use_container_width=True, hide_index=True)
    if "V1" in metrics_v1_v2.columns and "V2" in metrics_v1_v2.columns:
        plot_df = metrics_v1_v2.melt(id_vars=["Metric"], value_vars=["V1", "V2"], var_name="Version", value_name="Score")
        fig_compare = px.bar(
            plot_df,
            x="Metric",
            y="Score",
            color="Version",
            barmode="group",
            title="V1 vs V2 comparison",
            height=420,
        )
        st.plotly_chart(fig_compare, use_container_width=True)

    st.markdown("#### Heavy-rain detector metrics")
    st.dataframe(metrics_heavy, use_container_width=True, hide_index=True)

    st.markdown("#### Limitation note")
    st.markdown(
        """
        <div class="bad-box">
        Exact heavy-rainfall amount prediction is still difficult because heavy rainfall is rare and imbalanced. This prototype focuses on risk screening, early warning, and spatial dashboarding rather than exact mm-level forecasting.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### What improved in V2")
    st.markdown(
        """
        - MAE improved from **3.41 → 3.08**.
        - RMSE improved from **8.68 → 8.36**.
        - R² improved from **0.17 → 0.23**.
        - Risk Accuracy improved from **0.63 → 0.69**.
        - Weighted F1 improved from **0.68 → 0.74**.
        """
    )
