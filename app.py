import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from kmeans_model import preprocess_data, apply_kmeans

st.title("🛍️ Customer Segmentation using K-Means Clustering")

uploaded_file = st.file_uploader("Upload your customer dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(Mall_customers.csv)
    st.write("### Data Preview", df.head())

    selected_columns = st.multiselect("Select features for clustering", df.columns)

    if selected_columns:
        data = df[selected_columns]
        scaled_data = preprocess_data(data)

        n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)

        clusters, model = apply_kmeans(scaled_data, n_clusters)
        df['Cluster'] = clusters

        st.write("### Clustered Data", df.head())

        # Plotting
        if len(selected_columns) == 2:
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=selected_columns[0], y=selected_columns[1], hue="Cluster", data=df, palette="Set2")
            st.pyplot(plt)
        else:
            st.write("Select exactly 2 features to display 2D scatter plot.")
