import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans

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
    except Exception as e:
        st.warning("Could not predict audience group for your input. Please check your selections and try again.")

st.info("Your preferences are mapped into a hidden audience segmentation model. Only your cluster/group is shown above—no dataset is displayed.")
