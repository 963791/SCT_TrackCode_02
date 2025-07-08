import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from kmeans_model import preprocess_data, apply_kmeans

st.set_page_config(page_title="Mall Customer Segmentation", layout="centered")

st.title("🛍️ K-Means Customer Segmentation App")
st.markdown("Use the built-in dataset or upload your own.")

# Load built-in dataset
default_df = pd.read_csv("Mall_Customers.csv")

# File uploader (optional)
uploaded_file = st.file_uploader("📤 Upload your own CSV (optional)", type=["csv"])

# Use uploaded file if available, else use built-in
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom file loaded.")
else:
    df = default_df.copy()
    st.info("ℹ️ Using built-in dataset: Mall_Customers.csv")

# Convert and format Annual Income to ₹
if 'Annual Income (k$)' in df.columns:
    df['Annual Income (₹)'] = df['Annual Income (k$)'] * 1000 * 83
    df['Annual Income (₹)'] = df['Annual Income (₹)'].apply(lambda x: f"₹{x:,.0f}")

# Show data sample
st.write("### Sample Data", df.head())

# Feature selection
all_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender']
default_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
selected_columns = st.multiselect("Select features for clustering", all_columns, default_features)

if selected_columns:
    # Preprocess
    scaled_data, final_features = preprocess_data(df, selected_columns)

    # Choose number of clusters
    n_clusters = st.slider("Select number of clusters", 2, 10, 3)

    # Run KMeans
    clusters, model = apply_kmeans(scaled_data, n_clusters)
    df['Cluster'] = clusters

    # Display results
    st.write("### Clustered Data", df[['CustomerID'] + selected_columns + ['Cluster']].head())

    # 2D Plot
    if len(final_features) == 2:
        st.write("### 2D Cluster Visualization")
       import numpy as np

st.write("### 2D Cluster Visualization")

        import numpy as np

st.write("### 2D Cluster Visualization")
    
    # Create a new DataFrame with scaled features and clusters
    plot_df = pd.DataFrame(scaled_data, columns=final_features)
    plot_df['Cluster'] = clusters
    
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=plot_df,
        x=final_features[0],
        y=final_features[1],
        hue='Cluster',
        palette="Set2",
        s=100
    )
    plt.title("Customer Segments")
    plt.xlabel(final_features[0])
    plt.ylabel(final_features[1])
    st.pyplot(plt)


    else:
        st.info("Select exactly 2 numerical features to show 2D cluster plot.")
else:
    st.warning("Please select at least one feature for clustering.")
