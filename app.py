import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load and preprocess data (cached)
@st.cache_data
def load_data():
    df = pd.read_excel('data0.xlsx')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['demandMW_lag1'] = df['demandMW'].shift(1)
    df['demandMW_lag2'] = df['demandMW'].shift(2)
    df['demandMW_lag3'] = df['demandMW'].shift(3)
    df['demandMW_rollmean3'] = df['demandMW'].shift(1).rolling(window=3).mean()
    df = df.dropna().reset_index(drop=True)
    return df

df = load_data()

# Define training set
train_df = df[df['datetime'] < '2025-01-01'].copy()
features = ['demandMW_lag1', 'demandMW_lag2', 'demandMW_lag3', 'demandMW_rollmean3', 'hour_sin']
X_train = train_df[features]
y_train = train_df['demandMW']

# Train model (cached)
@st.cache_resource
def train_model():
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

model = train_model()

st.title('Hourly Demand Forecasting')

user_date = st.date_input("Select a date to predict hourly demand")
if user_date:
    forecast_date = pd.Timestamp(user_date)
    hours = pd.date_range(start=forecast_date, periods=24, freq='H')

    # Use the most recent values before the selected day to create initial state
    past_df = df[df['datetime'] < forecast_date].copy().reset_index(drop=True)

    if len(past_df) < 3:
        st.error("Not enough historical data to predict.")
    else:
        last_known = past_df.iloc[-3:].copy()
        preds = []
        current_history = last_known.copy()

        for h in range(24):
            hour = h
            hour_sin = np.sin(2 * np.pi * hour / 24)

            lag1 = current_history.iloc[-1]['demandMW']
            lag2 = current_history.iloc[-2]['demandMW']
            lag3 = current_history.iloc[-3]['demandMW']
            rollmean3 = current_history['demandMW'].iloc[-3:].mean()

            input_features = pd.DataFrame([{
                'demandMW_lag1': lag1,
                'demandMW_lag2': lag2,
                'demandMW_lag3': lag3,
                'demandMW_rollmean3': rollmean3,
                'hour_sin': hour_sin
            }])

            prediction = model.predict(input_features)[0]
            preds.append(prediction)

            # Append predicted value to history for next step
            next_row = pd.DataFrame([{
                'datetime': forecast_date + pd.Timedelta(hours=h),
                'hour': hour,
                'demandMW': prediction
            }])
            current_history = pd.concat([current_history, next_row], ignore_index=True)

        # Create result dataframe
        result_df = pd.DataFrame({
            'Datetime': hours,
            'Predicted Demand (MW)': preds
        })

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(result_df['Datetime'], result_df['Predicted Demand (MW)'], marker='o', linestyle='-')
        ax.set_title(f'Predicted Hourly Demand for {forecast_date.date()}')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Demand (MW)')
        ax.grid(True)
        st.pyplot(fig)

        # Table
        st.subheader('Hourly Forecast Table')
        st.dataframe(result_df.style.format({'Predicted Demand (MW)': '{:.2f}'}))
