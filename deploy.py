import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the CSV file
df = pd.read_csv("data.csv")

# Handling missing values
missing_values = pd.isnull(df).sum()

# Drop columns with too many missing values or that are not needed for recommendations
df.drop(["key", "mode", "explicit"], axis=1, inplace=True)

# Calculate the duration in seconds and drop the original duration_ms column
df["duration"] = df["duration_ms"].apply(lambda x: round(x / 1000))
df.drop("duration_ms", inplace=True, axis=1)

# Feature selection
features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'loudness', 'speechiness', 'tempo', 'duration']
song_features = df[features]

# Normalize the features
song_features_normalized = (song_features - song_features.mean()) / (song_features.max() - song_features.min())

# Load the K-means clustering model (you need to train it or load a pre-trained one)
# kmeans = load_model("kmeans_model.h5")

# Load the Random Forest popularity prediction model (you need to train it or load a pre-trained one)
# rf_model = load_model("rf_model.h5")

# Define a function to get song recommendations from the same cluster
def get_cluster_recommendations(song_name, kmeans_model, data):
    song_index = data[data['name'] == song_name].index[0]
    cluster_label = kmeans_model.labels_[song_index]
    cluster_songs = data[kmeans_model.labels_ == cluster_label]['name']
    return cluster_songs

# Define a function to get song recommendations based on predicted popularity
def get_popularity_based_recommendations(song_name, model, data):
    song_features = data[data['name'] == song_name][features]
    predicted_popularity = model.predict(song_features)
    similar_songs = data[data['popularity'] >= predicted_popularity[0]].sort_values(by='popularity', ascending=False)
    return similar_songs['name']

# Define Streamlit app
def main():
    st.title("Music Recommendation System")

    # Add Streamlit widgets and user interface elements
    # For example, you can create text input fields or buttons to interact with your app
    user_id = st.text_input("Enter User ID:", value="123")
    song_name = st.text_input("Enter a Song Name:", value="Song Name")

    # Get cluster-based recommendations
    if st.button("Get Cluster Recommendations"):
        cluster_recommendations = get_cluster_recommendations(song_name, kmeans, df)
        st.header("Cluster-Based Recommendations:")
        st.table(cluster_recommendations)

    # Get popularity-based recommendations
    if st.button("Get Popularity Recommendations"):
        popularity_recommendations = get_popularity_based_recommendations(song_name, rf_model, df)
        st.header("Popularity-Based Recommendations:")
        st.table(popularity_recommendations)

if __name__ == "__main__":
    main()
