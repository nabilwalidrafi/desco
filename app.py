import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load and preprocess data (cached)
@st.cache_data
def load_data():
    df = pd.read_excel('data0.xlsx')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['demandMW_lag1'] = df['demandMW'].shift(1)
    df['demandMW_lag2'] = df['demandMW'].shift(2)
    df['demandMW_lag3'] = df['demandMW'].shift(3)
    df['demandMW_rollmean3'] = df['demandMW'].shift(1).rolling(window=3).mean()
    df = df.dropna()
    return df

df = load_data()

# Define training and test sets
train_start = '2023-01-01'
train_end = '2024-12-31'
test_start = '2025-01-01'
test_end = '2025-03-10'

train_df = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)].copy()
test_df = df[(df['datetime'] >= test_start) & (df['datetime'] <= test_end)].copy()

features = ['demandMW_lag1', 'demandMW_lag2', 'demandMW_lag3', 'demandMW_rollmean3', 'hour_sin']
X_train = train_df[features]
y_train = train_df['demandMW']

# Train model once (cached)
@st.cache_resource
def train_model():
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

rf_final = train_model()

st.title('Demand Forecasting App')

user_date = st.date_input(
    'Select a date between January 1, 2025 and March 10, 2025',
    min_value=pd.Timestamp('2025-01-01'),
    max_value=pd.Timestamp('2025-03-10')
)

if user_date:
    user_date = pd.Timestamp(user_date)
    start_date = user_date - pd.Timedelta(days=6)  # 7-day window

    # Filter the test data for the 7-day window
    week_data = test_df[(test_df['datetime'] >= start_date) & (test_df['datetime'] <= user_date)].copy()

    if week_data.empty:
        st.warning("No data available for the selected week.")
    else:
        X_week = week_data[features]
        y_actual = week_data['demandMW']

        # Predict
        y_pred = rf_final.predict(X_week)

        # Calculate difference
        diff = y_actual.values - y_pred

        # Calculate R² on the week window, show as Accuracy
        accuracy = r2_score(y_actual, y_pred)

        # Plot actual vs predicted demand
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(week_data['datetime'], y_actual, label='Actual DemandMW', marker='o')
        ax.plot(week_data['datetime'], y_pred, label='Predicted DemandMW', marker='x')
        ax.set_title('Actual vs Predicted DemandMW (7-day window)')
        ax.set_xlabel('Date')
        ax.set_ylabel('DemandMW')
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)

        st.markdown(f"### Accuracy (R²) over selected week: {accuracy:.3f}")

        # Show predicted demand and difference table
        result_df = pd.DataFrame({
            'Date': week_data['datetime'],
            'Actual DemandMW': y_actual,
            'Predicted DemandMW': y_pred,
            'Difference (Actual - Predicted)': diff
        }).set_index('Date')

        st.subheader('Predicted Demand and Differences')
        st.dataframe(result_df.style.format({
            'Actual DemandMW': '{:.2f}',
            'Predicted DemandMW': '{:.2f}',
            'Difference (Actual - Predicted)': '{:.2f}'
        }))
