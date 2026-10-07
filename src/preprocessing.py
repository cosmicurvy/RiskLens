import pandas as pd
import numpy as np

def convert_into_datetime_obj(data, feature):
    """Converts a feature into datetime object."""
    data[feature] = pd.to_datetime(data[feature])
    return data[feature]


def extract_features(data):
    """Extract features from the existing columns"""
   
    if all(col in data.columns for col in ['timestamp', 'user_id', 'amount']):
        # time-based features
        data['trans_hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.day_of_week
        data['is_night_trans'] = ((data['trans_hour'] >= 19) | (data['trans_hour'] < 6)).astype(int)
        data['account_age_days'] = (data['timestamp'].dt.normalize() - data['account_created_date'].dt.normalize()).dt.days
        # amount-based features
        data['user_avg_amt'] = data.groupby('user_id')['amount'].transform(lambda x: x.expanding().mean().shift(1)).fillna(0)
        data['amt_to_avg_ratio'] = np.where(data['user_avg_amt'] > 0, data['amount'] / data['user_avg_amt'], 1.0)
        data['is_high_amount'] = (data['amount'] > data['user_avg_amt'] * 2).astype(int)

        df_time = data.set_index('timestamp')

        data['user_trans_count_1h'] = df_time.groupby('user_id')['amount'].rolling('1h', closed='left').count().fillna(0).values
        data['user_trans_count_5min'] = df_time.groupby('user_id')['amount'].rolling('5min', closed='left').count().fillna(0).values
        data['user_trans_sum_1h'] = df_time.groupby('user_id')['amount'].rolling('1h', closed='left').sum().fillna(0).values

        data['time_since_last_min'] = (data.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 60.0).fillna(0)

        return data


def drop_columns(data):
    """Drop columns that are no longer needed for model training."""
    cols_to_drop = ['transaction_id','user_id', 'timestamp', 'account_created_date']
    data.drop(columns=cols_to_drop, inplace=True)
    return data
