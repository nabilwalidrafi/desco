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
    df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['demandMW_lag1'] = df['demandMW'].shift(1)
    df['demandMW_lag2'] = df['demandMW'].shift(2)
    df['demandMW_lag3'] = df['demandMW'].shift(3)
    df['demandMW_rollmean3'] = df['demandMW'].shift(1).rolling(window=3).mean()
    df = df.dropna()
    return df

df = load_data()

# Define training set
train_start = '2023-01-01'
train_end = '2024-12-31'
train_df = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)].copy()

features = ['demandMW_lag1', 'demandMW_lag2', 'demandMW_lag3', 'demandMW_rollmean3', 'hour_sin']
X_train = train_df[features]
y_train = train_df['demandMW']

# Train model (cached)
@st.cache_resource
def train_model():
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

rf_final = train_model()

st.title('Demand Forecasting App')

# User input for any date
user_date = st.date_input('Select a date for prediction')

if user_date:
    user_date = pd.Timestamp(user_date)
    # Create a 24-hour datetime range for the selected date
    date_range = pd.date_range(start=user_date, end=user_date + pd.Timedelta(hours=23), freq='H')
    
    # Get the last known data point and predict up to the selected date
    last_known_date = df['datetime'].max()
    if last_known_date < user_date:
        # Generate datetime range from last known date + 1 hour to the start of user_date
        pred_range = pd.date_range(start=last_known_date + pd.Timedelta(hours=1), 
                                 end=user_date - pd.Timedelta(hours=1), freq='H')
        feature_data = df[df['datetime'] <= last_known_date].tail(3).copy()
        
        if len(feature_data) < 3:
            st.warning("Not enough historical data to make predictions.")
        else:
            # Predict sequentially up to user_date
            for dt in pred_range:
                hour_sin = np.sin(2 * np.pi * dt.hour / 24)
                demand_lag1 = feature_data['demandMW'].iloc[-1]
                demand_lag2 = feature_data['demandMW'].iloc[-2]
                demand_lag3 = feature_data['demandMW'].iloc[-3]
                roll_mean = feature_data['demandMW'].tail(3).mean()
                
                features_array = np.array([[demand_lag1, demand_lag2, demand_lag3, roll_mean, hour_sin]])
                pred = rf_final.predict(features_array)[0]
                
                # Update feature_data with new prediction
                new_row = pd.DataFrame({
                    'datetime': [dt],
                    'demandMW': [pred],
                    'hour_sin': [hour_sin],
                    'demandMW_lag1': [demand_lag1],
                    'demandMW_lag2': [demand_lag2],
                    'demandMW_lag3': [demand_lag3],
                    'demandMW_rollmean3': [roll_mean]
                })
                feature_data = pd.concat([feature_data, new_row], ignore_index=True)
                feature_data = feature_data.tail(3)  # Keep only the last 3 rows
    else:
        # If user_date is on or before last known date, use available data
        feature_data = df[df['datetime'] < user_date].tail(3).copy()
        if len(feature_data) < 3:
            st.warning("Not enough historical data to make predictions for the selected date.")
            feature_data = None
    
    if feature_data is not None:
        # Predict for the selected date (24 hours)
        predictions = []
        for dt in date_range:
            hour_sin = np.sin(2 * np.pi * dt.hour / 24)
            demand_lag1 = feature_data['demandMW'].iloc[-1]
            demand_lag2 = feature_data['demandMW'].iloc[-2]
            demand_lag3 = feature_data['demandMW'].iloc[-3]
            roll_mean = feature_data['demandMW'].tail(3).mean()
            
            features_array = np.array([[demand_lag1, demand_lag2, demand_lag3, roll_mean, hour_sin]])
            pred = rf_final.predict(features_array)[0]
            predictions.append(pred)
            
            # Update feature_data with the predicted value
            new_row = pd.DataFrame({
                'datetime': [dt],
                'demandMW': [pred],
                'hour_sin': [hour_sin],
                'demandMW_lag1': [demand_lag1],
                'demandMW_lag2': [demand_lag2],
                'demandMW_lag3': [demand_lag3],
                'demandMW_rollmean3': [roll_mean]
            })
            feature_data = pd.concat([feature_data, new_row], ignore_index=True)
            feature_data = feature_data.tail(3)  # Keep only the last 3 rows
        
        # Create results DataFrame for table
        result_df = pd.DataFrame({
            'Hour': date_range,
            'Predicted Demand (MW)': predictions
        }).set_index('Hour')
        
        # Display table
        st.subheader(f'Predicted Demand for {user_date.date()}')
        st.dataframe(result_df.style.format({
            'Predicted Demand (MW)': '{:.2f}'
        }))
        
        # Plot predictions
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(date_range, predictions, label='Predicted Demand (MW)', marker='o', color='#1f77b4')
	ax.set_title(f'Predicted Demand for {user_date.strftime("%I %p")}')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Demand (MW)')
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)