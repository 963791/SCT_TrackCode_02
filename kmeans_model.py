import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def preprocess_data(df, selected_columns):
    df_processed = df[selected_columns].copy()

    # Convert Annual Income to Rupees for clustering
    if 'Annual Income (k$)' in df_processed.columns:
        df_processed['Annual Income (₹)'] = df_processed['Annual Income (k$)'] * 1000 * 83
        df_processed.drop(columns=['Annual Income (k$)'], inplace=True)

    # One-hot encode Gender if used
    if 'Gender' in df_processed.columns:
        df_processed = pd.get_dummies(df_processed, columns=['Gender'], drop_first=True)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_processed)
    return scaled_data, df_processed.columns
