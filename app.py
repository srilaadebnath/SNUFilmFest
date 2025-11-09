import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.title("OTT Audience Map for SNU Film Fest")

st.markdown("""
The Cultural Committee is planning the annual Film Fest. 
They want to know the student audience segments.

📊 **Columns:** `movie_genre_top1`, `series_genre_top1`, `ott_top1`, `content_lang_top1`
🧩 **Task:** Cluster students into viewing preference groups.
🏆 **Impact:** Helps plan screenings and OTT tie-ups.
""")

# --- Choose input mode ---
input_mode = st.radio(
    "Choose how to enter your data:",
    ("Upload CSV or XLSX file", "Manual input (single user)")
)

# Main columns required as per the question
required_categorical_cols = ['movie_genre_top1', 'series_genre_top1', 'ott_top1', 'content_lang_top1']

if input_mode == "Upload CSV or XLSX file":
    uploaded_file = st.file_uploader("Upload CSV or Excel File", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        # Column cleaning
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        st.write("Detected Columns in dataset:", df.columns.tolist())

        # Select only the required columns
        available_cols = [col for col in required_categorical_cols if col in df.columns]
        missing_cols = [col for col in required_categorical_cols if col not in df.columns]
        df_selected = df[available_cols].fillna("Unknown")
        st.write(f"Used Columns for clustering: {available_cols}")

        if missing_cols:
            st.warning(f"Missing columns in your file: {missing_cols}. Clustering will use only available columns.")

        # Encoding
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(df_selected)
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(available_cols))

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

        k = st.slider("Select k for KMeans (audience segments)", min_value=2, max_value=8, value=3)
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
        ax.set_title("Student Audience Groups (PCA)")
        st.pyplot(fig)

        # Cluster count
        fig2, ax2 = plt.subplots()
        sns.countplot(x=df['Cluster'], palette="viridis", ax=ax2)
        ax2.set_title("Number of Students per Audience Group")
        st.pyplot(fig2)

        # Cluster summary
        st.subheader("Audience Group Insights")
        for cluster in sorted(df['Cluster'].unique()):
            st.markdown(f"**Group {cluster} — Viewing Preferences:**")
            cluster_data = df[df['Cluster']==cluster]
            for col in available_cols:
                top_choice = cluster_data[col].mode()[0]
                st.write(f"Most common {col.replace('_', ' ')}: `{top_choice}`")

        st.info("Use these insights to plan screenings and OTT tie-ups for different student segments!")

    else:
        st.info("Upload a file to get started.")

elif input_mode == "Manual input (single user)":
    st.subheader("Enter your preferences below:")
    movie_genre = st.selectbox(
        "Your top movie genre:",
        ["Action", "Thriller", "Comedy", "Romance", "Drama", "Horror", "Sci-Fi", "Other"]
    )
    series_genre = st.selectbox(
        "Your top series genre:",
        ["Crime", "Drama", "Comedy", "Thriller", "Fantasy", "Other"]
    )
    ott_platform = st.selectbox(
        "Your preferred OTT platform:",
        ["Netflix", "Amazon Prime", "Disney+", "Hotstar", "Other"]
    )
    content_lang = st.selectbox(
        "Preferred content language:",
        ["English", "Hindi", "Bengali", "Tamil", "Telugu", "Other"]
    )
    if st.button("Show my input as processed data"):
        # Create single-row dataframe
        df = pd.DataFrame([{
            "movie_genre_top1": movie_genre,
            "series_genre_top1": series_genre,
            "ott_top1": ott_platform,
            "content_lang_top1": content_lang
        }])
        st.write("Processed Data:", df)

        # Dummy clustering for one user
        st.warning("With single user input, clustering and metrics are not meaningful (need multiple students).")
        st.info("To see clustering and student audience groups, upload a dataset of multiple students.")

        # Show encoding as demonstration
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(df)
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(df.columns.tolist()))
        st.write("One-hot encoded data for your input:", encoded_df)
