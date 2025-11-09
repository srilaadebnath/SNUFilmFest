import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.title("SNU Film Fest: Student Preferences Clustering")

# File upload
uploaded_file = st.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    # Column cleaning
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    st.write("Cleaned Columns:", df.columns.tolist())

    categorical_cols = ['movie_genre_top1','series_genre_top1','ott_top1','content_lang_top1']
    categorical_cols = [col for col in categorical_cols if col in df.columns]
    df_selected = df[categorical_cols].fillna("Unknown")
    st.write("Used Columns: ", categorical_cols)

    # Encoding
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded = encoder.fit_transform(df_selected)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

    # Elbow Plot
    inertia = []
    K_range = range(2, 8)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(encoded_df)
        inertia.append(km.inertia_)
    
    fig, ax = plt.subplots()
    ax.plot(K_range, inertia, marker="o")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method for Optimal k")
    st.pyplot(fig)

    k = st.slider("Select k for KMeans", min_value=2, max_value=8, value=3)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(encoded_df)

    # Metrics
    silhouette = silhouette_score(encoded_df, df['Cluster'])
    db_index = davies_bouldin_score(encoded_df, df['Cluster'])
    st.write(f"**Silhouette Score:** {silhouette:.3f}")
    st.write(f"**Davies-Bouldin Index:** {db_index:.3f}")

    # PCA Plot
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(encoded_df)
    fig, ax = plt.subplots()
    sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df['Cluster'], palette="viridis", s=60, ax=ax)
    ax.set_title("Clusters of Student Preferences (PCA)")
    st.pyplot(fig)

    # Cluster count
    fig2, ax2 = plt.subplots()
    sns.countplot(x=df['Cluster'], palette="viridis", ax=ax2)
    ax2.set_title("Number of Students per Cluster")
    st.pyplot(fig2)

    # Cluster summary
    for cluster in sorted(df['Cluster'].unique()):
        st.subheader(f"Cluster {cluster} Insights")
        cluster_data = df[df['Cluster']==cluster]
        for col in categorical_cols:
            top_choice = cluster_data[col].mode()[0]
            st.write(f"**Most common {col}:** {top_choice}")

else:
    st.info("Upload a file to get started.")
