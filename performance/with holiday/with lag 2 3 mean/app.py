import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Define holiday periods (Eid weeks and other holidays) in global scope
holiday_periods = [
    # Eid ul Fitr weeks
    ('2023-04-20', '2023-04-26'), ('2024-04-08', '2024-04-14'), ('2025-03-29', '2025-04-04'),
    ('2026-03-18', '2026-03-24'), ('2027-03-07', '2027-03-13'), ('2028-02-24', '2028-03-01'),
    ('2029-02-13', '2029-02-19'), ('2030-02-02', '2030-02-08'),
    # Eid ul Adha weeks
    ('2023-06-28', '2023-07-04'), ('2024-06-16', '2024-06-22'), ('2025-06-05', '2025-06-11'),
    ('2026-05-25', '2026-05-31'), ('2027-05-14', '2027-05-20'), ('2028-05-02', '2028-05-08'),
    ('2029-04-21', '2029-04-27'), ('2030-04-10', '2030-04-16'),
]
annual_holidays = [
    ('02-21', 'Shaheed Dibash'), ('03-26', 'Independence Day'), ('04-14', 'Bengali New Year'),
    ('05-01', 'May Day'), ('08-15', 'National Mourning Day'), ('12-16', 'Victory Day'),
    ('12-25', 'Christmas Day'),
]
variable_holidays = [
    ('2023-05-05', 'Buddha Purnima'), ('2023-08-20', 'Ashura'), ('2023-10-24', 'Durga Puja'),
    ('2024-08-09', 'Ashura'), ('2025-05-11', 'Buddha Purnima'), ('2025-07-29', 'Ashura'),
    ('2025-09-04', 'Janmashtami'), ('2025-10-01', 'Eid-e-Miladunnabi'), ('2025-10-02', 'Eid-e-Miladunnabi'),
    ('2025-10-01', 'Durga Puja'), ('2026-05-01', 'Buddha Purnima'), ('2026-07-18', 'Ashura'),
    ('2026-09-20', 'Eid-e-Miladunnabi'), ('2026-09-20', 'Durga Puja'), ('2027-04-20', 'Buddha Purnima'),
    ('2027-07-07', 'Ashura'), ('2027-09-09', 'Eid-e-Miladunnabi'), ('2027-09-09', 'Durga Puja'),
    ('2028-05-08', 'Buddha Purnima'), ('2028-06-26', 'Ashura'), ('2028-08-31', 'Janmashtami'),
    ('2028-09-27', 'Durga Puja'), ('2029-04-27', 'Buddha Purnima'), ('2029-06-15', 'Ashura'),
    ('2029-08-20', 'Janmashtami'), ('2029-09-16', 'Durga Puja'), ('2030-04-16', 'Buddha Purnima'),
    ('2030-06-04', 'Ashura'), ('2030-08-09', 'Janmashtami'), ('2030-09-05', 'Durga Puja'),
]

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
    
    # Add holiday feature
    df['is_holiday'] = 0
    for start, end in holiday_periods:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        df.loc[(df['datetime'] >= start) & (df['datetime'] <= end), 'is_holiday'] = 1
    for date, _ in annual_holidays:
        for year in range(2023, 2031):
            holiday_date = pd.to_datetime(f'{year}-{date}')
            df.loc[df['datetime'].dt.date == holiday_date.date(), 'is_holiday'] = 1
    for date, _ in variable_holidays:
        holiday_date = pd.to_datetime(date)
        df.loc[df['datetime'].dt.date == holiday_date.date(), 'is_holiday'] = 1
    
    df = df.dropna()
    return df

df = load_data()

# Filter training set
train_start = '2023-01-01'
train_end = '2025-03-10'
train_df = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)].copy()

# Features
features = ['demandMW_lag1', 'demandMW_lag2', 'demandMW_lag3', 'demandMW_rollmean3', 'hour_sin', 'is_holiday']
X_train = train_df[features]
y_train = train_df['demandMW']

# Train model (cached)
@st.cache_resource
def train_model():
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf

rf_final = train_model()

st.title('DESCO Electricity Demand Predictor')

# User input for prediction date
user_date = st.date_input('Select a date for prediction (2025-03-11 to 2030-12-31)')

if user_date:
    user_date = pd.Timestamp(user_date)
    if user_date < pd.to_datetime('2025-03-11') or user_date > pd.to_datetime('2030-12-31'):
        st.error('Please select a date between March 11, 2025, and December 31, 2030.')
    else:
        # Create 24-hour datetime range for the selected date
        date_range = pd.date_range(start=user_date, end=user_date + pd.Timedelta(hours=23), freq='H')
        
        # Get the last known data point
        last_known_date = df['datetime'].max()
        feature_data = df[df['datetime'] <= last_known_date].tail(3).copy()
        
        if len(feature_data) < 3:
            st.warning("Not enough historical data to make predictions.")
        else:
            # Define holiday check function
            def is_holiday_date(dt, holiday_periods, annual_holidays, variable_holidays):
                # Check holiday periods (Eid weeks)
                for start, end in holiday_periods:
                    if pd.to_datetime(start) <= dt <= pd.to_datetime(end):
                        return 1
                # Check annual holidays
                dt_str = dt.strftime('%m-%d')
                if any(dt_str == h[0] for h in annual_holidays):
                    return 1
                # Check variable holidays
                if any(dt.date() == pd.to_datetime(h[0]).date() for h in variable_holidays):
                    return 1
                return 0

            # Predict sequentially up to user_date if needed
            if last_known_date < user_date:
                pred_range = pd.date_range(start=last_known_date + pd.Timedelta(hours=1), 
                                         end=user_date - pd.Timedelta(hours=1), freq='H')
                for dt in pred_range:
                    hour_sin = np.sin(2 * np.pi * dt.hour / 24)
                    demand_lag1 = feature_data['demandMW'].iloc[-1]
                    demand_lag2 = feature_data['demandMW'].iloc[-2]
                    demand_lag3 = feature_data['demandMW'].iloc[-3]
                    roll_mean = feature_data['demandMW'].tail(3).mean()
                    is_holiday = is_holiday_date(dt, holiday_periods, annual_holidays, variable_holidays)
                    
                    features_array = np.array([[demand_lag1, demand_lag2, demand_lag3, roll_mean, hour_sin, is_holiday]])
                    pred = rf_final.predict(features_array)[0]
                    
                    new_row = pd.DataFrame({
                        'datetime': [dt],
                        'demandMW': [pred],
                        'hour_sin': [hour_sin],
                        'demandMW_lag1': [demand_lag1],
                        'demandMW_lag2': [demand_lag2],
                        'demandMW_lag3': [demand_lag3],
                        'demandMW_rollmean3': [roll_mean],
                        'is_holiday': [is_holiday]
                    })
                    feature_data = pd.concat([feature_data, new_row], ignore_index=True)
                    feature_data = feature_data.tail(3)
            
            # Predict for the selected date
            predictions = []
            for dt in date_range:
                hour_sin = np.sin(2 * np.pi * dt.hour / 24)
                demand_lag1 = feature_data['demandMW'].iloc[-1]
                demand_lag2 = feature_data['demandMW'].iloc[-2]
                demand_lag3 = feature_data['demandMW'].iloc[-3]
                roll_mean = feature_data['demandMW'].tail(3).mean()
                is_holiday = is_holiday_date(dt, holiday_periods, annual_holidays, variable_holidays)
                
                features_array = np.array([[demand_lag1, demand_lag2, demand_lag3, roll_mean, hour_sin, is_holiday]])
                pred = rf_final.predict(features_array)[0]
                predictions.append(pred)
                
                new_row = pd.DataFrame({
                    'datetime': [dt],
                    'demandMW': [pred],
                    'hour_sin': [hour_sin],
                    'demandMW_lag1': [demand_lag1],
                    'demandMW_lag2': [demand_lag2],
                    'demandMW_lag3': [demand_lag3],
                    'demandMW_rollmean3': [roll_mean],
                    'is_holiday': [is_holiday]
                })
                feature_data = pd.concat([feature_data, new_row], ignore_index=True)
                feature_data = feature_data.tail(3)
            
            # Create results DataFrame
            result_df = pd.DataFrame({
                'Hour': date_range,
                'Predicted Demand (MW)': predictions
            }).set_index('Hour')
            
            # Display table
            st.subheader(f'Predicted Demand for {user_date.date()}')
            st.dataframe(result_df.style.format({'Predicted Demand (MW)': '{:.2f}'}))

            # Plot predictions
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(date_range, predictions, label='Predicted Demand (MW)', marker='o', color='#1f77b4')
            ax.set_title(f'Predicted Demand for {user_date.date()}')
            ax.set_xlabel('Hour')
            ax.set_ylabel('Demand (MW)')
            ax.legend()
            ax.grid(True)
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
            plt.xticks(rotation=45)
            st.pyplot(fig)

# Feature importance
perm_importance = permutation_importance(rf_final, X_train, y_train, n_repeats=10, random_state=42)
st.subheader('Feature Importance')
for i, feature in enumerate(features):
    st.write(f"{feature}: {perm_importance.importances_mean[i]:.4f}")

# Plot feature importance
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(features, perm_importance.importances_mean, color='#1f77b4')
ax.set_xlabel('Permutation Importance')
ax.set_title('Feature Importance on Training Set (2023–Mar 2025)')
plt.tight_layout()
st.pyplot(fig)