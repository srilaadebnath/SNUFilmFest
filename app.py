import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.title("OTT Audience Map for SNU Film Fest")

st.markdown("""
The Cultural Committee is planning the annual Film Fest. 
They want to know the student audience segments.

📊 **Columns:** `movie_genre_top1`, `series_genre_top1`, `ott_top1`, `content_lang_top1`
🧩 **Task:** Cluster students into viewing preference groups.
🏆 **Impact:** Helps plan screenings and OTT tie-ups.
""")

# --- Embedded training dataset (not shown to user but used in clustering) ---
embedded_csv = """
movie_genre_top1,series_genre_top1,ott_top1,content_lang_top1
Action,Crime,Netflix,English
Comedy,Drama,Amazon Prime,Hindi
Romance,Fantasy,Disney+,English
Thriller,Drama,Hotstar,Bengali
Drama,Comedy,Amazon Prime,Hindi
Horror,Thriller,Netflix,English
Sci-Fi,Fantasy,Hotstar,Tamil
Comedy,Drama,Netflix,Hindi
Action,Crime,Disney+,Telugu
Romance,Drama,Amazon Prime,English
"""

required_categorical_cols = ['movie_genre_top1', 'series_genre_top1', 'ott_top1', 'content_lang_top1']

# Load embedded CSV as DataFrame (used only for training, not displayed)
from io import StringIO
df_train = pd.read_csv(StringIO(embedded_csv))
df_train.columns = df_train.columns.str.strip().str.lower().str.replace(" ", "_")
available_cols = [col for col in required_categorical_cols if col in df_train.columns]
df_selected = df_train[available_cols].fillna("Unknown")

# Fit encoder and kmeans only once, cache for performance
@st.cache_resource
def fit_cluster_model():
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_train = encoder.fit_transform(df_selected)
    # Number of clusters k (can be tuned, kept fixed for simplicity)
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(encoded_train)
    return encoder, kmeans, available_cols

encoder, kmeans, encoded_columns = fit_cluster_model()

# Get cluster labels for base profiles
df_train['audience_group'] = kmeans.predict(encoder.transform(df_selected))

# --- Manual user input section ---
st.subheader("Enter your preferences below to discover your audience group:")
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
if st.button("Find my audience group"):
    df_user = pd.DataFrame([{
        "movie_genre_top1": movie_genre,
        "series_genre_top1": series_genre,
        "ott_top1": ott_platform,
        "content_lang_top1": content_lang
    }])

    try:
        user_encoded = encoder.transform(df_user[encoded_columns])
        cluster = kmeans.predict(user_encoded)
        st.success(f"Your viewing preferences belong to **Audience Group {cluster[0]}**.")

        # 1. Show cluster distribution
        st.write("### Audience Groups Distribution")
        group_counts = df_train['audience_group'].value_counts().sort_index()
        fig1, ax1 = plt.subplots()
        ax1.bar(group_counts.index.astype(str), group_counts.values)
        ax1.set_xlabel("Audience Group")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)
        
        # 2. Show genre/OTT/lang breakdown for user's group
        st.write(f"### Preferences Within Your Audience Group {cluster[0]}")
        group_df = df_train[df_train['audience_group'] == cluster[0]]
        col1, col2, col3 = st.columns(3)

        with col1:
            genre_counts = group_df['movie_genre_top1'].value_counts()
            fig2, ax2 = plt.subplots()
            ax2.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%')
            ax2.set_title("Movie Genre Breakdown")
            st.pyplot(fig2)

        with col2:
            ott_counts = group_df['ott_top1'].value_counts()
            fig3, ax3 = plt.subplots()
            ax3.pie(ott_counts, labels=ott_counts.index, autopct='%1.1f%%')
            ax3.set_title("OTT Platform Breakdown")
            st.pyplot(fig3)

        with col3:
            lang_counts = group_df['content_lang_top1'].value_counts()
            fig4, ax4 = plt.subplots()
            ax4.pie(lang_counts, labels=lang_counts.index, autopct='%1.1f%%')
            ax4.set_title("Content Language Breakdown")
            st.pyplot(fig4)

        # Optional: Save previous user's selections for further analysis (not implemented here)
    except Exception as e:
        st.warning("Could not predict audience group for your input. Please check your selections and try again.")

st.info("Your preferences are mapped into a hidden audience segmentation model. Only your cluster/group is shown above—no dataset is displayed.")
