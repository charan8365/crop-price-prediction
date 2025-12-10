# app_streamlit_demo.py — Clean multi-crop Streamlit app (quiet messages + robust CSV fallback)
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Crop Price Predictor — Multi-crop Demo", layout="wide")
st.title("Crop Price Predictor — Multi-crop Demo")

# --- Crop dataset map (prefer merged files if you created them) ---
DATA_DIR = "data"
crop_files = {
    "Tomato": (f"{DATA_DIR}/tomato_merged.csv", f"{DATA_DIR}/tomato_365.csv"),
    "Onion": (f"{DATA_DIR}/onion_merged.csv", f"{DATA_DIR}/onion_365.csv"),
    "Potato": (f"{DATA_DIR}/potato_merged.csv", f"{DATA_DIR}/potato_365.csv"),
    "Paddy": (f"{DATA_DIR}/paddy_merged.csv", f"{DATA_DIR}/paddy_365.csv"),
}

st.sidebar.header("Select Data / Model")
selected_crop = st.sidebar.selectbox("Crop", list(crop_files.keys()), index=0)

# choose preferred file (merged if exists, otherwise fallback)
merged_path, raw_path = crop_files[selected_crop]
if os.path.exists(merged_path):
    DATA_PATH = merged_path
elif os.path.exists(raw_path):
    DATA_PATH = raw_path
else:
    st.error(f"Dataset not found for {selected_crop}. Expected {merged_path} or {raw_path}.")
    st.stop()

# load dataframe
try:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
except Exception as e:
    st.error(f"Failed to read dataset {DATA_PATH}: {e}")
    st.stop()
df = df.sort_values("date").reset_index(drop=True)
if "modal_price" not in df.columns and "y" in df.columns and "ds" in df.columns:
    df = df.rename(columns={"ds": "date", "y": "modal_price"})

# sidebar controls
st.sidebar.subheader("Filters & Settings")
market_options = ["All"] + (sorted(df["market"].dropna().unique().tolist()) if "market" in df.columns else [])
market = st.sidebar.selectbox("Market", market_options)
days = st.sidebar.slider("Forecast days", min_value=7, max_value=365, value=30, step=1)

# filter by market for display and for regressors
df_market = df[df["market"] == market].copy() if market != "All" and "market" in df.columns else df.copy()

# Historical sample & chart
st.header("Historical Prices (sample)")
if "arrivals" in df_market.columns:
    st.dataframe(df_market[["date", "commodity", "market", "modal_price", "arrivals"]].head(10))
else:
    # show commodity/market only if present
    cols = ["date", "commodity", "market", "modal_price"]
    cols = [c for c in cols if c in df_market.columns]
    st.dataframe(df_market[cols].head(10))

st.subheader("Historical Price Chart")
try:
    st.line_chart(df_market.set_index("date")["modal_price"])
except Exception:
    st.warning("Could not draw price chart — check that 'date' and 'modal_price' columns exist and are parsed correctly.")

# -------------------------
# MODEL + FORECAST (robust)
# -------------------------

# Attempt to load model for the selected crop; keep UI quiet and provide details in expanders
model = None
model_path = f"models/prophet_{selected_crop.lower()}_All.pkl"

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        st.success("Loaded model from disk.")
    except Exception as e:
        # log full traceback to server log
        logging.exception("Failed to load model %s", model_path)
        # show short notice + details inside an expander
        st.warning("Model failed to load on server — app will try precomputed CSV forecasts. (Click to view error)")
        with st.expander("Model load error details"):
            st.write(str(e))
else:
    st.info("No model file found on server; app will try precomputed CSV forecasts from results/ if available.")

# If model loaded, attempt to forecast. Otherwise load forecast CSV from results/
forecast = None
if model is not None:
    try:
        # identify regressors used by model (quiet)
        expected_regressors = list(model.extra_regressors.keys()) if hasattr(model, "extra_regressors") and model.extra_regressors else []
        future = model.make_future_dataframe(periods=days)

        if expected_regressors:
            # Put regressor info inside an expander so it doesn't clutter UI
            with st.expander("Regressor details (click to view)"):
                st.write("Model expects regressors:", expected_regressors)
                st.write("Regressors will be aligned to future dates and filled (ffill/bfill, then median fallback).")

            # build regressor source aligned to dates
            reg_source = df_market.rename(columns={"date": "ds"}).set_index("ds")
            # if duplicate dates, aggregate sensible stats
            if not reg_source.index.is_unique:
                agg_dict = {}
                for r in expected_regressors:
                    agg_dict[r] = "sum" if r == "arrivals" else "mean"
                for r in expected_regressors:
                    if r not in reg_source.columns:
                        reg_source[r] = np.nan
                reg_source = reg_source.groupby(level=0).agg(agg_dict)

            # ensure regressors exist as columns
            for r in expected_regressors:
                if r not in reg_source.columns:
                    reg_source[r] = np.nan

            # align to future dates, fill missing sensibly
            reg_frame = reg_source[expected_regressors].reindex(future["ds"]).reset_index()
            reg_frame[expected_regressors] = reg_frame[expected_regressors].ffill().bfill()
            for r in expected_regressors:
                if reg_frame[r].isna().any():
                    median_val = float(reg_source[r].median(skipna=True)) if not reg_source[r].dropna().empty else 0.0
                    reg_frame[r] = reg_frame[r].fillna(median_val)

            future = future.merge(reg_frame, on="ds", how="left")

        # perform forecast
        forecast = model.predict(future)
    except Exception as e:
        logging.exception("Model forecasting failed")
        st.warning("Model forecasting failed — falling back to precomputed CSV forecasts (click to view error).")
        with st.expander("Forecast error details"):
            st.write(str(e))
        forecast = None

# If forecast None (no model or model failed), try loading CSVs
if forecast is None:
    csv_candidates = [
        f"results/{selected_crop.lower()}_forecast_full.csv",
        f"results/{selected_crop.lower()}_forecast_2026.csv",
        f"results/{selected_crop.lower()}_forecast_90.csv",
        f"results/{selected_crop.lower()}_forecast.csv",
        f"results/{selected_crop.lower()}_forecast_90.csv",
        "results/prophet_forecast.csv"
    ]
    found = False
    for c in csv_candidates:
        if os.path.exists(c):
            try:
                forecast = pd.read_csv(c, parse_dates=["ds"])
                st.success(f"Loaded precomputed forecast: {os.path.basename(c)}")
                found = True
                break
            except Exception as e:
                logging.exception("Failed to load CSV %s", c)
                st.warning(f"Could not load {c}: {e}")
    if not found and forecast is None:
        st.info("No precomputed forecast CSV found in results/. Forecasting unavailable.")

# Display forecast if available
if forecast is not None:
    source = "model" if model is not None and "yhat" in forecast.columns else "CSV"
    st.subheader(f"Forecast for next {days} days (from {source})")

    # Pick a sensible price column
    price_col = None
    for candidate in ["yhat", "modal_price", "y", "pred", "price"]:
        if candidate in forecast.columns:
            price_col = candidate
            break

    if price_col is None:
        st.dataframe(forecast.head(20))
    else:
        # try to chart tail(days), fallback to whole series
        try:
            st.line_chart(forecast.set_index("ds")[price_col].tail(days))
        except Exception:
            try:
                st.line_chart(forecast.set_index("ds")[price_col])
            except Exception:
                st.write("Could not chart forecast timeseries.")
        display_cols = [c for c in ["ds", "yhat", "yhat_lower", "yhat_upper", "modal_price", "price"] if c in forecast.columns]
        st.dataframe(forecast[display_cols].tail(days))
else:
    st.info("No forecast available (no model and no CSVs in results/).")

# -------------------------
# RECOMMENDATIONS DOWNLOAD
# -------------------------
if st.sidebar.button("Show 2026 Crop Recommendations"):
    if os.path.exists("results/crop_recommendations_2026.csv"):
        try:
            rec = pd.read_csv("results/crop_recommendations_2026.csv")
            st.header("Best Crop to Sell Each Month — 2026")
            st.dataframe(rec)
        except Exception as e:
            st.error(f"Could not read results/crop_recommendations_2026.csv: {e}")

        if os.path.exists("results/crop_advice_2026.csv"):
            try:
                advice = pd.read_csv("results/crop_advice_2026.csv")
                st.subheader("Farmer Advice (2026)")
                for _, row in advice.iterrows():
                    sec_txt = "No data" if pd.isna(row.get("second_price")) else f"{str(row.get('second_crop')).capitalize()} (~₹{row.get('second_price'):.0f})"
                    best_txt = "No data" if pd.isna(row.get("best_price")) else f"{str(row.get('best_crop')).capitalize()} (~₹{row.get('best_price'):.0f})"
                    st.write(f"• **{row.get('month_name')}** — Best: **{best_txt}**; Second: {sec_txt}")
            except Exception as e:
                st.warning(f"Could not read results/crop_advice_2026.csv: {e}")
    else:
        st.warning("Recommendations file not found. Run the notebook to generate results/crop_recommendations_2026.csv")
