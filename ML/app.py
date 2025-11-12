# app.py — month→season auto, manual dummies, no negative preds, cleaned UI
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bike Rentals (Daily)", layout="centered")
st.title("Daily Bike Rentals: Linear Regression")

# === Load artifacts ===
MODEL_PATH = "model.joblib"
COLS_PATH  = "model_columns.json"
model = joblib.load(MODEL_PATH)
with open(COLS_PATH, "r", encoding="utf-8") as f:
    MODEL_COLS = json.load(f)   # 19 columns (exact training order)

# === Label ↔ code maps (match original dataset coding) ===
SEASONS = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}
MONTHS  = {
    "January":1, "February":2, "March":3, "April":4, "May":5, "June":6,
    "July":7, "August":8, "September":9, "October":10, "November":11, "December":12
}
WEATHER = {
    "Clear / Few clouds / Partly cloudy": 1,
    "Mist / Cloudy": 2,
    "Light rain or light snow": 3
}

# Expected dummy layout from training (drop_first=True)
CAT_COLS = {"season":[2,3,4], "weathersit":[2,3], "mnth": list(range(2,13))}
BASE_NUMS = ["temp","hum","windspeed"]

def month_to_season_label(m_code:int) -> str:
    if m_code in (12, 1, 2):   return "Winter"
    if m_code in (3, 4, 5):    return "Spring"
    if m_code in (6, 7, 8):    return "Summer"
    return "Fall"  # 9,10,11

def scale_to_dataset_units(temp_c: float, hum_pct: float, wind_kmh: float):
    # temp_norm = temp_C / 41 ; hum_norm = % / 100 ; wind_norm = km/h / 67
    t = max(0.0, min(1.0, temp_c / 41.0))
    h = max(0.0, min(1.0, hum_pct / 100.0))
    w = max(0.0, min(1.0, wind_kmh / 67.0))
    return t, h, w

def to_model_df(season_code:int, mnth_code:int, weather_code:int,
                temp_norm:float, hum_norm:float, wind_norm:float) -> pd.DataFrame:
    # Manual one-hot encoding to exactly match MODEL_COLS
    row = {col: 0 for col in MODEL_COLS}
    row["temp"] = float(temp_norm)
    row["hum"] = float(hum_norm)
    row["windspeed"] = float(wind_norm)

    if season_code in (2, 3, 4):
        col = f"season_{season_code}"
        if col in row: row[col] = 1
    if weather_code in (2, 3):
        col = f"weathersit_{weather_code}"
        if col in row: row[col] = 1
    if 2 <= mnth_code <= 12:
        col = f"mnth_{mnth_code}"
        if col in row: row[col] = 1

    return pd.DataFrame([row], columns=MODEL_COLS)

# ===== UI =====
with st.sidebar:
    st.header("Inputs")

    # Month first
    month_label  = st.selectbox("Month",  list(MONTHS.keys()), index=0)
    m_code = MONTHS[month_label]

    # Season auto from month (disabled display)
    auto_season_label = month_to_season_label(m_code)
    st.selectbox("Season (auto from month)", list(SEASONS.keys()),
                 index=list(SEASONS.keys()).index(auto_season_label),
                 disabled=True)

    # Weather (labels)
    weather_label = st.selectbox("Weather", list(WEATHER.keys()), index=0)

    # Real-world units (no extra caption)
    temp_c   = st.slider("Temperature (°C)", 0.0, 41.0, 20.0, 0.5)
    hum_pct  = st.slider("Humidity (%)",     0.0, 100.0, 60.0, 1.0)
    wind_kmh = st.slider("Wind speed (km/h)",0.0, 67.0, 12.0, 0.5)

    # Show scaled values one per line
    t_norm, h_norm, w_norm = scale_to_dataset_units(temp_c, hum_pct, wind_kmh)
    st.text(f"temp={t_norm:.2f}")
    st.text(f"hum={h_norm:.2f}")
    st.text(f"windspeed={w_norm:.2f}")

    go = st.button("Predict")

# ===== Predict =====
if go:
    s_code = SEASONS[auto_season_label]
    w_code = WEATHER[weather_label]

    X = to_model_df(s_code, m_code, w_code, t_norm, h_norm, w_norm)
    yhat = float(model.predict(X)[0])
    yhat = max(0.0, yhat)  # never negative

    st.success(f"Predicted daily rentals: {yhat:,.0f}")

    
