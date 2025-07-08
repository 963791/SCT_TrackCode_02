import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from kmeans_model import preprocess_data, apply_kmeans

st.set_page_config(page_title="Mall Customer Segmentation", layout="centered")

st.title("🛍️ Mall Customer Segmentation using K-Means")
st.markdown("Upload your data and perform customer clustering.")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Convert and format Rupees
    if 'Annual Income (k$)' in df.columns:
        df['Annual Income (₹)'] = df['Annual Income (k$)'] * 1000 * 83
        df['Annual Income (₹)'] = df['Annual Income (₹)'].apply(lambda x: f"₹{x:,.0f}")

    st.write("### Sample Data", df.head())

    # Feature selection
    all_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender']
    default_features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    selected_columns = st.multiselect("Select features for clustering", all_columns, default_features)

    if selected_columns:
        # Preprocess data
        scaled_data, final_features = preprocess_data(df, selected_columns)

        # Select number of clusters
        n_clusters = st.slider("Select number of clusters", 2, 10, 3)

        # Apply KMeans
        clusters, model = apply_kmeans(scaled_data, n_clusters)
        df['Cluster'] = clusters

        st.write("### Clustered Data", df[['CustomerID'] + selected_columns + ['Cluster']].head())

        # 2D Visualization
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
            st.info("Select exactly 2 numerical features to show 2D cluster plot.")
else:
    st.info("Please upload a CSV file to begin.")
