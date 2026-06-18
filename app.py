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
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #eef9ff 0%, #f7fff4 50%, #fffbea 100%);
            color: #102a43;
        }
        .hero-card, .info-card, .metric-card, .mini-card {
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(32, 117, 163, 0.12);
            box-shadow: 0 10px 28px rgba(30, 85, 110, 0.07);
            padding: 1.2rem;
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2.2rem;
            font-weight: 900;
            color: #0d3b66;
            margin-bottom: 0.35rem;
        }
        .hero-subtitle {
            font-size: 1rem;
            color: #315b72;
            line-height: 1.6;
        }
        .metric-card {
            min-height: 110px;
        }
        .metric-label {
            color: #4b6b82;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            color: #0d3b66;
            font-size: 1.6rem;
            font-weight: 800;
        }
        .small-muted {
            color: #4b6b82;
            font-size: 0.90rem;
        }
        .note-box, .good-box, .bad-box {
            border-radius: 20px;
            padding: 1rem 1.1rem;
            margin: 1rem 0;
        }
        .note-box {
            background: rgba(230, 245, 255, 0.95);
            border: 1px solid rgba(36, 116, 197, 0.16);
            color: #133d5c;
        }
        .good-box {
            background: rgba(225, 250, 235, 0.94);
            border: 1px solid rgba(67, 154, 100, 0.18);
            color: #164f2d;
        }
        .bad-box {
            background: rgba(255, 238, 238, 0.95);
            border: 1px solid rgba(190, 80, 80, 0.18);
            color: #6b2b2b;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
        }
        .stTabs [aria-selected="true"] {
            background: #bee6ff !important;
            color: #0d3b66 !important;
            font-weight: 800;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Paths
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


# ============================================================
# Main interface
# ============================================================
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">🌧️ AI Rainfall-Risk Digital Twin</div>
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


tabs = st.tabs(["🏡 Home / Overview", "🗺️ Prediction Dashboard", "🧪 What-if Simulator", "📊 V1 vs V2 Results"])


# ============================================================
# Home tab
# ============================================================
with tabs[0]:
    st.markdown("### Project overview")
    st.write(
        "This rain-risk digital twin prototype combines historical IMD rainfall targets with NASA POWER climate features to score tomorrow's rainfall risk in Punjab and Haryana. "
        "It is built for insight, storyboarding, and early-warning visualization rather than exact millimeter forecasting."
    )
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
with tabs[1]:
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
        selected_date = st.selectbox("Select forecast date", available_dates, index=len(available_dates) - 1)
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
