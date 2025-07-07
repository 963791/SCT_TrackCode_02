import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from kmeans_model import preprocess_data, apply_kmeans

st.set_page_config(page_title="Mall Customer Segmentation", layout="centered")

st.title("🛍️ Mall Customer Segmentation using K-Means")
st.markdown("Cluster mall customers based on demographics and spending behavior.")

# Load dataset
df = pd.read_csv("Mall_Customers.csv")
df['Annual Income (₹)'] = df['Annual Income (k$)'] * 1000 * 83  # Convert to rupees
df['Annual Income (₹)'] = df['Annual Income (₹)'].apply(lambda x: f"₹{x:,.0f}")  # Format

st.write("### Sample Data (with Rupee conversion)", df.head())

# Feature selection for clustering (numerical features)
all_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender']
default_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

selected_columns = st.multiselect(
    "Select features to use for clustering",
    options=all_columns,
    default=default_features
)

if selected_columns:
    # Process data: conversion, encoding, scaling
    scaled_data, final_features = preprocess_data(df, selected_columns)

    # Cluster selection
    n_clusters = st.slider("Number of Clusters", 2, 10, value=3)

    # Apply K-means
    clusters, model = apply_kmeans(scaled_data, n_clusters)
    df['Cluster'] = clusters

    st.write("### Clustered Data (with Cluster Labels)", df[['CustomerID'] + selected_columns + ['Cluster']].head())

    # Visualization if exactly 2 features selected
    if len(final_features) == 2:
        st.write("### 2D Cluster Visualization")
        plt.figure(figsize=(8, 5))
        sns.scatterplot(
            x=final_features[0],
            y=final_features[1],
            hue=df['Cluster'],
            palette="Set2",
            s=100
        )
        plt.title("Customer Segments")
        plt.xlabel(final_features[0])
        plt.ylabel(final_features[1])
        st.pyplot(plt)
    else:
        st.info("Select exactly 2 numerical features to visualize clusters in 2D.")
