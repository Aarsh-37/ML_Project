import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

# --- Load the trained model ---
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# --- App title & description ---
st.set_page_config(page_title="Power Consumption Predictor", layout="wide")
st.title("💡 Power Consumption Prediction")

# --- Input layout  ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌦 Environmental Inputs")
    temperature = st.number_input("Temperature (°C)", value=25.0, format="%.2f")
    wind_speed = st.number_input("Wind Speed (m/s)", value=3.0, format="%.2f")
    gen_diffuse_flows = st.number_input("General Diffuse Flows", value=180.0, format="%.2f")
    diffuse_flows = st.number_input("Diffuse Flows", value=70.0, format="%.2f")

with col2:
    st.subheader("🕒 Date & Time Input")
    d = st.date_input("Select a Date", datetime.now())
    t = st.time_input("Select a Time", datetime.now().time())

# Combine date & time
user_datetime = pd.to_datetime(f"{d} {t}")
time_fraction = user_datetime.hour + user_datetime.minute / 60 + user_datetime.second / 3600

# --- Predict Button ---
if st.button("Predict Power Consumption"):
    # Prepare input
    input_data = pd.DataFrame([[
        temperature,
        wind_speed,
        gen_diffuse_flows,
        diffuse_flows,
        time_fraction
    ]], columns=[
        'Temperature',
        'WindSpeed',
        'GeneralDiffuseFlows',
        'DiffuseFlows',
        'time_fraction'
    ])

    # Make prediction
    prediction = model.predict(input_data)[0]
    st.success(f"### ⚡ Predicted Power Consumption: **{prediction:,.2f} units**")

    # --- AI-based insights ---
    # --- Prediction for next 12 hours ---
    st.subheader("📈 Predicted Power Trend (Next 12 Hours)")
    future_hours = pd.date_range(user_datetime, periods=12, freq='H')
    future_time_fraction = [t.hour + t.minute / 60 for t in future_hours]

    future_df = pd.DataFrame({
        'Temperature': temperature,
        'WindSpeed': wind_speed,
        'GeneralDiffuseFlows': gen_diffuse_flows,
        'DiffuseFlows': diffuse_flows,
        'time_fraction': future_time_fraction
    })

    future_preds = model.predict(future_df)

    # --- Two columns: left (plot) | right (insights) ---
    col1, col2 = st.columns([2, 1])

    with col1:
        plt.figure(figsize=(5, 3))
        plt.plot(range(1, 13), future_preds, marker='o', color='tab:blue')
        plt.title("Predicted Power (Next 12 Hours)")
        plt.xlabel("Next Hours")
        plt.ylabel("Power (units)")
        plt.xticks(range(1, 13))
        plt.grid(True)
        st.pyplot(plt)

    with col2:
        st.subheader("📊 Smart Insights")
        if temperature > 35:
            st.warning("🔥 High temperature — likely increased cooling load and higher consumption.")
        elif temperature < 10:
            st.info("❄️ Low temperature — power usage may rise due to heating requirements.")
        else:
            st.success("🌤 Moderate temperature — balanced consumption expected.")

        if wind_speed > 5:
            st.info("💨 Strong wind — could slightly reduce overall temperature-related power demand.")

                # Display average
        avg_pred = np.mean(future_preds)
        st.markdown(f"**📅 Average Predicted Consumption (Next 12 Hours): {avg_pred:,.2f} units**")

else:
    st.info("👈 Enter input values and click **Predict Power Consumption** to get predictions and insights.")
