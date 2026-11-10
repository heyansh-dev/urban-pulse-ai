import streamlit as st
import pandas as pd
import requests
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="UrbanPulse AI - Air Quality Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 UrbanPulse AI: Air Quality Forecasting Dashboard")
st.markdown("Production inference dashboard for real-time PM10 and PM2.5 multi-horizon forecasting.")

# Sidebar Configuration
st.sidebar.header("Configuration")
city = st.sidebar.selectbox("Select Target Node", ["Bengaluru", "Chennai", "Delhi"])
model_type = st.sidebar.selectbox("Model Architecture", ["xgboost", "lightgbm"])
api_url = st.sidebar.text_input("FastAPI Service URL", "http://127.0.0.1:8000")

# Main Interface Layout
col1, col2 = st.columns([1, 1])
sample_file = os.path.join("data", "raw_sample_data.csv")

with col1:
    st.subheader(f"📊 Recent Sample Historical Data ({city})")
    if os.path.exists(sample_file):
        df_raw = pd.read_csv(sample_file)
        if 'city' in df_raw.columns:
            df_city_view = df_raw[df_raw['city'].str.lower() == city.lower()]
        else:
            df_city_view = df_raw
        st.dataframe(df_city_view.head(10), use_container_width=True)
    else:
        st.warning("Sample dataset missing at data/raw_sample_data.csv")

with col2:
    st.subheader("⚡ Live Forecast Engine")
    generate_btn = st.button("Generate Air Quality Forecast")

if generate_btn:
    with st.spinner(f"Fetching {model_type.upper()} forecasts for {city}..."):
        try:
            # Load city-specific readings from CSV if available, or build city-varied payload
            if os.path.exists(sample_file):
                df_raw = pd.read_csv(sample_file)
                if 'parameter' in df_raw.columns:
                    df_city = df_raw[df_raw['city'].str.lower() == city.lower()].copy()
                    df_pivoted = df_city.pivot_table(
                        index=['timestamp', 'city'], 
                        columns='parameter', 
                        values='value'
                    ).reset_index()
                    
                    recent_readings = [
                        {
                            "timestamp": str(row["timestamp"]),
                            "city": city.lower(),
                            "pm10": float(row.get("pm10", 80.0) if pd.notna(row.get("pm10")) else 80.0),
                            "pm25": float(row.get("pm25", 45.0) if pd.notna(row.get("pm25")) else 45.0)
                        }
                        for _, row in df_pivoted.tail(48).iterrows()
                    ]
                else:
                    df_city = df_raw[df_raw['city'].str.lower() == city.lower()].tail(48)
                    recent_readings = [
                        {
                            "timestamp": str(row["timestamp"]),
                            "city": city.lower(),
                            "pm10": float(row["pm10"]),
                            "pm25": float(row["pm25"])
                        }
                        for _, row in df_city.iterrows()
                    ]
            else:
                city_offsets = {"bengaluru": 15.0, "chennai": 40.0, "delhi": 95.0}
                offset = city_offsets.get(city.lower(), 0.0)
                timestamps = pd.date_range(end=pd.Timestamp.now(), periods=48, freq='h')
                recent_readings = [
                    {
                        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "city": city.lower(),
                        "pm10": float(75.0 + offset + (i % 6) * 3.0),
                        "pm25": float(42.0 + (offset * 0.5) + (i % 4) * 2.0)
                    }
                    for i, ts in enumerate(timestamps)
                ]

            payload = {
                "city": city.lower(),
                "recent_readings": recent_readings
            }

            # Single backend request
            response = requests.post(f"{api_url}/predict?model_type={model_type}", json=payload, timeout=30)
            
            if response.status_code == 200:
                predictions = response.json()["predictions"]
                pred_dict = {p["horizon"]: p for p in predictions}

                st.write("---")
                st.success(f"Forecast results generated for **{city}** using **{model_type.upper()}**!")

                # Render direct metric cards
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("1-Hour PM10 Forecast", f"{pred_dict['1h']['pm10_prediction']} µg/m³")
                m2.metric("24-Hour PM10 Forecast", f"{pred_dict['24h']['pm10_prediction']} µg/m³")
                m3.metric("1-Hour PM2.5 Forecast", f"{pred_dict['1h']['pm25_prediction']} µg/m³")
                m4.metric("24-Hour PM2.5 Forecast", f"{pred_dict['24h']['pm25_prediction']} µg/m³")

                # Trajectory chart
                st.write("---")
                st.subheader(f"📈 Forecast Trajectory ({city})")

                c1, c2 = st.columns(2)
                horizons = ["Current", "1-Hour Ahead", "24-Hour Ahead"]
                curr_pm10 = recent_readings[-1]["pm10"]
                curr_pm25 = recent_readings[-1]["pm25"]

                pm10_vals = [curr_pm10, pred_dict['1h']['pm10_prediction'], pred_dict['24h']['pm10_prediction']]
                pm25_vals = [curr_pm25, pred_dict['1h']['pm25_prediction'], pred_dict['24h']['pm25_prediction']]

                fig1 = go.Figure(data=go.Scatter(x=horizons, y=pm10_vals, mode='lines+markers', line=dict(color='#FF4B4B', width=3)))
                fig1.update_layout(title="PM10 Trajectory (µg/m³)", xaxis_title="Horizon", yaxis_title="PM10 Level")

                fig2 = go.Figure(data=go.Scatter(x=horizons, y=pm25_vals, mode='lines+markers', line=dict(color='#0083B8', width=3)))
                fig2.update_layout(title="PM2.5 Trajectory (µg/m³)", xaxis_title="Horizon", yaxis_title="PM2.5 Level")

                with c1:
                    st.plotly_chart(fig1, use_container_width=True)
                with c2:
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.error("API call failed. Make sure FastAPI server is running.")

        except Exception as e:
            st.error(f"Error querying backend: {str(e)}")