import streamlit as st
import pandas as pd

# --- Data Loading and Preprocessing ---
# Use st.cache_data to cache the data loading so it only runs once.
@st.cache_data
def load_data():
    # Define file paths. These paths assume the ml-100k folder is in the same directory as app.py
    u_data_path = 'ml-100k/u.data'
    u_item_path = 'ml-100k/u.item'

    # Define column names
    u_data_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings_df = pd.read_csv(u_data_path, sep='\t', names=u_data_cols, encoding='latin-1')

    u_item_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure',
                   'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                   'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    movies_df = pd.read_csv(u_item_path, sep='|', names=u_item_cols, encoding='latin-1')

    # Merge the dataframes
    movie_ratings = pd.merge(ratings_df, movies_df, on='movie_id')

    # Create the user-item matrix for calculations
    user_movie_matrix = movie_ratings.pivot_table(index='user_id', columns='title', values='rating')

    # Create a summary of ratings for filtering
    ratings_summary = movie_ratings.groupby('title')['rating'].agg(['count', 'mean'])

    return user_movie_matrix, ratings_summary, movies_df

# --- Recommender Function ---
def get_recommendations(movie_title, user_movie_matrix, ratings_summary, min_ratings=100):
    # Get the ratings for the input movie
    movie_ratings_vec = user_movie_matrix[movie_title]

    # Calculate correlation with all other movies
    similar_movies = user_movie_matrix.corrwith(movie_ratings_vec)

    # Create a DataFrame for the correlations
    correlation_df = pd.DataFrame(similar_movies, columns=['Correlation'])
    correlation_df.dropna(inplace=True)

    # Join with the ratings summary
    full_rec_df = correlation_df.join(ratings_summary['count'])

    # Filter out movies that don't meet the minimum rating count and sort
    recommendations = full_rec_df[full_rec_df['count'] > min_ratings].sort_values('Correlation', ascending=False)

    return recommendations


# --- Streamlit App Interface ---
st.title('🎬 Movie Recommender System')
st.write("Select a movie you like, and we'll recommend similar ones!")

# Load the data using the cached function
user_movie_matrix, ratings_summary, movies_df = load_data()

# Create a dropdown for movie selection
movie_list = movies_df['title'].sort_values().unique()
selected_movie = st.selectbox(
    "Choose a movie:",
    movie_list
)

# Create a button to get recommendations
if st.button('Get Recommendations'):
    if selected_movie:
        with st.spinner(f"Finding recommendations for '{selected_movie}'..."):
            recommendations = get_recommendations(selected_movie, user_movie_matrix, ratings_summary)

            st.subheader("Here are your top recommendations:")
            # Display top 6 (since the movie itself is #1)
            st.dataframe(recommendations.head(6))
    else:
        st.write("Please select a movie.")