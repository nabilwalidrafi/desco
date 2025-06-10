import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Define Eid periods
holiday_periods = [
    ('2023-04-20', '2023-04-26'), ('2024-04-08', '2024-04-14'), ('2025-03-29', '2025-04-04'),
    ('2026-03-18', '2026-03-24'), ('2027-03-07', '2027-03-13'), ('2028-02-24', '2028-03-01'),
    ('2029-02-13', '2029-02-19'), ('2030-02-01', '2030-02-07'),
    ('2023-06-28', '2023-07-04'), ('2024-06-16', '2024-06-22'), ('2025-06-05', '2025-06-11'),
    ('2026-05-25', '2026-05-31'), ('2027-05-14', '2027-05-20'), ('2028-05-02', '2028-05-08'),
    ('2029-04-21', '2029-04-27'), ('2030-04-10', '2030-04-16'),
]

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_excel('data0.xlsx')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour_sin'] = np.sin(2 * np.pi * df['datetime'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['datetime'].dt.hour / 24)
    df['demandMW_lag1'] = df['demandMW'].shift(1)
    
    # Add Eid feature
    df['is_eid'] = 0
    for start, end in holiday_periods:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        df.loc[(df['datetime'] >= start) & (df['datetime'] <= end), 'is_eid'] = 1
    
    df = df.dropna()
    return df

df = load_data()

# Plot historical Eid demand
st.subheader('Historical Eid Demand (2023–2024)')
eid_df = df[df['is_eid'] == 1][['datetime', 'demandMW']]
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(eid_df['datetime'], eid_df['demandMW'], label='Demand (MW)', color='#1f77b4')
ax.set_title('Historical Demand During Eid Periods')
ax.set_xlabel('Date')
ax.set_ylabel('Demand (MW)')
ax.legend()
ax.grid(True)
plt.xticks(rotation=45)
st.pyplot(fig)
st.write(f"Eid demand summary: {eid_df['demandMW'].describe()}")

# Estimate average temperature by month and hour
temp_avg = df.groupby([df['datetime'].dt.month.rename('month'), 
                       df['datetime'].dt.hour.rename('hour')])['temp'].mean().reset_index()
temp_avg = temp_avg.set_index(['month', 'hour'])['temp'].to_dict()

# Filter training set
train_start = '2023-01-01'
train_end = '2025-03-10'
train_df = df[(df['datetime'] >= train_start) & (df['datetime'] <= train_end)].copy()

# Oversample Eid periods
eid_train_df = train_df[train_df['is_eid'] == 1]
train_df = pd.concat([train_df, eid_train_df, eid_train_df, eid_train_df, eid_train_df, eid_train_df, eid_train_df, eid_train_df], ignore_index=True)  # 8x Eid data

# Features
features = ['demandMW_lag1', 'hour_sin', 'hour_cos', 'is_eid', 'temp']

# Train RandomForest (non-Eid)
X_train_rf = train_df[features]
y_train_rf = train_df['demandMW']
sample_weights_rf = np.ones(len(train_df))  # Uniform weights for RF
@st.cache_resource
def train_rf():
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_rf, y_train_rf, sample_weight=sample_weights_rf)
    return rf
rf_model = train_rf()

# Train XGBoost (Eid-focused)
train_df_xgb = train_df.copy()
eid_mean_lag = train_df[train_df['is_eid'] == 1]['demandMW_lag1'].mean()  # Approx historical Eid lag
train_df_xgb['demandMW_lag1_eid_adjusted'] = train_df_xgb.apply(
    lambda row: row['demandMW_lag1'] * 0.5 if row['is_eid'] == 1 and row['demandMW_lag1'] > eid_mean_lag else row['demandMW_lag1'], axis=1
)
features_xgb = ['demandMW_lag1_eid_adjusted', 'hour_sin', 'hour_cos', 'is_eid', 'temp']
X_train_xgb = train_df_xgb[features_xgb]
y_train_xgb = train_df_xgb['demandMW']
sample_weights_xgb = np.where(train_df_xgb['is_eid'] == 1, 200, 1)
@st.cache_resource
def train_xgb():
    model = xgb.XGBRegressor(n_estimators=300, random_state=42, max_depth=5, learning_rate=0.03, subsample=0.8)
    model.fit(X_train_xgb, y_train_xgb, sample_weight=sample_weights_xgb)
    return model
xgb_model = train_xgb()

st.title('DESCO Electricity Demand Predictor (Optimized Hybrid Model)')

# User input for prediction date
user_date = st.date_input('Select a date for prediction (2025-03-11 to 2030-12-31)')

if user_date:
    user_date = pd.Timestamp(user_date)
    if user_date < pd.to_datetime('2025-03-11') or user_date > pd.to_datetime('2030-12-31'):
        st.error('Please select a date between March 11, 2025, and December 31, 2030.')
    else:
        # Create 24-hour datetime range
        date_range = pd.date_range(start=user_date, end=user_date + pd.Timedelta(hours=23), freq='H')
        
        # Get last known data point
        last_known_date = df['datetime'].max()
        feature_data = df[df['datetime'] <= last_known_date].tail(1).copy()
        
        if len(feature_data) < 1:
            st.warning("Not enough historical data to make predictions.")
        else:
            # Eid check function
            def is_eid_date(dt, holiday_periods):
                for start, end in holiday_periods:
                    if pd.to_datetime(start) <= dt <= pd.to_datetime(end):
                        return 1
                return 0

            # Predict sequentially up to user_date
            if last_known_date < user_date:
                pred_range = pd.date_range(start=last_known_date + pd.Timedelta(hours=1), 
                                         end=user_date - pd.Timedelta(hours=1), freq='H')
                for dt in pred_range:
                    hour_sin = np.sin(2 * np.pi * dt.hour / 24)
                    hour_cos = np.cos(2 * np.pi * dt.hour / 24)
                    demand_lag1 = feature_data['demandMW'].iloc[-1]
                    is_eid = is_eid_date(dt, holiday_periods)
                    temp = temp_avg.get((dt.month, dt.hour), 25.0)
                    
                    if is_eid:
                        # Smooth transition by weighting with historical Eid mean
                        demand_lag1_adjusted = (demand_lag1 * 0.3 + eid_mean_lag * 0.7) if demand_lag1 > eid_mean_lag else demand_lag1
                        features_array = np.array([[demand_lag1_adjusted, hour_sin, hour_cos, is_eid, temp]])
                        pred = xgb_model.predict(features_array)[0]
                    else:
                        features_array = np.array([[demand_lag1, hour_sin, hour_cos, is_eid, temp]])
                        pred = rf_model.predict(features_array)[0]
                    
                    new_row = pd.DataFrame({
                        'datetime': [dt],
                        'demandMW': [pred],
                        'hour_sin': [hour_sin],
                        'hour_cos': [hour_cos],
                        'demandMW_lag1': [demand_lag1],
                        'is_eid': [is_eid],
                        'temp': [temp]
                    })
                    feature_data = pd.concat([feature_data, new_row], ignore_index=True)
                    feature_data = feature_data.tail(1)
            
            # Predict for selected date
            predictions = []
            for dt in date_range:
                hour_sin = np.sin(2 * np.pi * dt.hour / 24)
                hour_cos = np.cos(2 * np.pi * dt.hour / 24)
                demand_lag1 = feature_data['demandMW'].iloc[-1]
                is_eid = is_eid_date(dt, holiday_periods)
                temp = temp_avg.get((dt.month, dt.hour), 25.0)
                
                if is_eid:
                    demand_lag1_adjusted = (demand_lag1 * 0.3 + eid_mean_lag * 0.7) if demand_lag1 > eid_mean_lag else demand_lag1
                    features_array = np.array([[demand_lag1_adjusted, hour_sin, hour_cos, is_eid, temp]])
                    pred = xgb_model.predict(features_array)[0]
                else:
                    features_array = np.array([[demand_lag1, hour_sin, hour_cos, is_eid, temp]])
                    pred = rf_model.predict(features_array)[0]
                
                predictions.append(pred)
                
                new_row = pd.DataFrame({
                    'datetime': [dt],
                    'demandMW': [pred],
                    'hour_sin': [hour_sin],
                    'hour_cos': [hour_cos],
                    'demandMW_lag1': [demand_lag1],
                    'is_eid': [is_eid],
                    'temp': [temp]
                })
                feature_data = pd.concat([feature_data, new_row], ignore_index=True)
                feature_data = feature_data.tail(1)
            
            # Results DataFrame
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

            # Validation plot for historical Eid (April 10, 2024) with XGBoost
            st.subheader('Validation: Predicted vs Historical Eid (April 10, 2024)')
            hist_date = pd.to_datetime('2024-04-10')
            hist_df = df[(df['datetime'].dt.date == hist_date.date())][['datetime', 'demandMW']]
            hist_feature_data = df[df['datetime'] < hist_date].tail(1).copy()
            
            hist_predictions = []
            hist_date_range = pd.date_range(start=hist_date, end=hist_date + pd.Timedelta(hours=23), freq='H')
            for dt in hist_date_range:
                hour_sin = np.sin(2 * np.pi * dt.hour / 24)
                hour_cos = np.cos(2 * np.pi * dt.hour / 24)
                demand_lag1 = hist_feature_data['demandMW'].iloc[-1]
                is_eid = is_eid_date(dt, holiday_periods)
                temp = temp_avg.get((dt.month, dt.hour), 25.0)
                
                demand_lag1_adjusted = (demand_lag1 * 0.3 + eid_mean_lag * 0.7) if demand_lag1 > eid_mean_lag else demand_lag1
                features_array = np.array([[demand_lag1_adjusted, hour_sin, hour_cos, is_eid, temp]])
                pred = xgb_model.predict(features_array)[0]
                
                hist_predictions.append(pred)
                
                new_row = pd.DataFrame({
                    'datetime': [dt],
                    'demandMW': [pred],
                    'hour_sin': [hour_sin],
                    'hour_cos': [hour_cos],
                    'demandMW_lag1': [demand_lag1],
                    'is_eid': [is_eid],
                    'temp': [temp]
                })
                hist_feature_data = pd.concat([hist_feature_data, new_row], ignore_index=True)
                hist_feature_data = hist_feature_data.tail(1)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(hist_df['datetime'], hist_df['demandMW'], label='Historical (2024)', color='green')
            ax.plot(hist_date_range, hist_predictions, label='Predicted', marker='o', color='#1f77b4')
            ax.set_title('Predicted vs Historical Eid Demand (April 10, 2024)')
            ax.set_xlabel('Hour')
            ax.set_ylabel('Demand (MW)')
            ax.legend()
            ax.grid(True)
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Validation plot for historical Eid (June 16, 2024) with XGBoost
            st.subheader('Validation: Predicted vs Historical Eid (June 16, 2024)')
            hist_date2 = pd.to_datetime('2024-06-16')
            hist_df2 = df[(df['datetime'].dt.date == hist_date2.date())][['datetime', 'demandMW']]
            hist_feature_data2 = df[df['datetime'] < hist_date2].tail(1).copy()
            
            hist_predictions2 = []
            hist_date_range2 = pd.date_range(start=hist_date2, end=hist_date2 + pd.Timedelta(hours=23), freq='H')
            for dt in hist_date_range2:
                hour_sin = np.sin(2 * np.pi * dt.hour / 24)
                hour_cos = np.cos(2 * np.pi * dt.hour / 24)
                demand_lag1 = hist_feature_data2['demandMW'].iloc[-1]
                is_eid = is_eid_date(dt, holiday_periods)
                temp = temp_avg.get((dt.month, dt.hour), 25.0)
                
                demand_lag1_adjusted = (demand_lag1 * 0.3 + eid_mean_lag * 0.7) if demand_lag1 > eid_mean_lag else demand_lag1
                features_array = np.array([[demand_lag1_adjusted, hour_sin, hour_cos, is_eid, temp]])
                pred = xgb_model.predict(features_array)[0]
                
                hist_predictions2.append(pred)
                
                new_row = pd.DataFrame({
                    'datetime': [dt],
                    'demandMW': [pred],
                    'hour_sin': [hour_sin],
                    'hour_cos': [hour_cos],
                    'demandMW_lag1': [demand_lag1],
                    'is_eid': [is_eid],
                    'temp': [temp]
                })
                hist_feature_data2 = pd.concat([hist_feature_data2, new_row], ignore_index=True)
                hist_feature_data2 = hist_feature_data2.tail(1)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(hist_df2['datetime'], hist_df2['demandMW'], label='Historical (2024)', color='green')
            ax.plot(hist_date_range2, hist_predictions2, label='Predicted', marker='o', color='#1f77b4')
            ax.set_title('Predicted vs Historical Eid Demand (June 16, 2024)')
            ax.set_xlabel('Hour')
            ax.set_ylabel('Demand (MW)')
            ax.legend()
            ax.grid(True)
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Feature importance for RandomForest
            st.subheader('RandomForest Feature Importance (Non-Eid)')
            perm_importance_rf = permutation_importance(rf_model, X_train_rf, y_train_rf, n_repeats=10, random_state=42)
            for i, feature in enumerate(features):
                st.write(f"{feature}: {perm_importance_rf.importances_mean[i]:.4f}")

            # Feature importance for XGBoost
            st.subheader('XGBoost Feature Importance (Eid)')
            perm_importance_xgb = permutation_importance(xgb_model, X_train_xgb, y_train_xgb, n_repeats=10, random_state=42)
            for i, feature in enumerate(features_xgb):
                st.write(f"{feature}: {perm_importance_xgb.importances_mean[i]:.4f}")

            # Plot feature importance
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].barh(features, perm_importance_rf.importances_mean, color='#1f77b4')
            ax[0].set_xlabel('Permutation Importance')
            ax[0].set_title('RandomForest Feature Importance (Non-Eid)')
            ax[1].barh(features_xgb, perm_importance_xgb.importances_mean, color='#2ca02c')
            ax[1].set_xlabel('Permutation Importance')
            ax[1].set_title('XGBoost Feature Importance (Eid)')
            plt.tight_layout()
            st.pyplot(fig)