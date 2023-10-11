import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# Add other imports for data visualization or processing if necessary

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
from sklearn.cluster import KMeans

# If you have a pre-trained K-means model, load it here
# kmeans = load_model("kmeans_model.h5")

# Define a function to get song recommendations from the same cluster
def get_recommendations(song_name, kmeans_model=kmeans, data=df):
    song_index = df[df['name'] == song_name].index[0]
    cluster_label = kmeans_model.labels_[song_index]
    cluster_songs = df[kmeans_model.labels_ == cluster_label]['name']
    return cluster_songs

# Load the Random Forest popularity prediction model (you need to train it or load a pre-trained one)
from sklearn.ensemble import RandomForestRegressor

# If you have a pre-trained Random Forest model, load it here
# rf_model = load_model("rf_model.h5")

# Define a function to get song recommendations based on predicted popularity
def get_recommendations(song_name, model=rf_model, data=df):
    song_features = df[df['name'] == song_name][features]
    predicted_popularity = model.predict(song_features)
    similar_songs = df[df['popularity'] >= predicted_popularity[0]].sort_values(by='popularity', ascending=False)
    return similar_songs['name']

# Define Streamlit app
def main():
    st.title("Music Recommendation System")

    # Add Streamlit widgets and user interface elements
    # For example, you can create text input fields or buttons to interact with your app
    user_id = st.text_input("Enter User ID:", value="123")
    recommendations = get_song_recommendations(int(user_id), n=10)

    st.header("Recommended Songs for User:")
    st.table(recommendations)

if __name__ == "__main__":
    main()


