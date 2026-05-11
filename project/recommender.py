import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import joblib

print("--- Step 1: Loading Data ---")
ratings = pd.read_csv('ml-32m/ratings.csv')
movies = pd.read_csv('ml-32m/movies.csv')

# Merge to get titles
ratings = ratings.merge(movies[['movieId', 'title']], on='movieId')

# Filter popular movies AND active users (smarter filtering)
movie_counts = ratings['movieId'].value_counts()
user_counts = ratings['userId'].value_counts()

popular_movies = movie_counts[movie_counts >= 50].index  # Movies with 50+ ratings
active_users = user_counts[user_counts >= 30].index      # Users with 30+ ratings

ratings_filtered = ratings[
    ratings['movieId'].isin(popular_movies) & 
    ratings['userId'].isin(active_users)
]

print(f"Filtered to {ratings_filtered['movieId'].nunique()} movies and {ratings_filtered['userId'].nunique()} users")

# Create sparse matrix (memory efficient!)
print("--- Step 2: Creating Sparse Matrix ---")
from sklearn.preprocessing import LabelEncoder

movie_encoder = LabelEncoder()
user_encoder = LabelEncoder()

movie_idx = movie_encoder.fit_transform(ratings_filtered['movieId'])
user_idx = user_encoder.fit_transform(ratings_filtered['userId'])

# Sparse matrix: ratings_range normalized to 0-1
ratings_norm = ratings_filtered['rating'].values / 5.0

sparse_matrix = csr_matrix(
    (ratings_norm, (movie_idx, user_idx)),
    shape=(len(movie_encoder.classes_), len(user_encoder.classes_))
)

print(f"Sparse matrix shape: {sparse_matrix.shape}")

# Apply SVD
print("--- Step 3: Training SVD ---")
svd = TruncatedSVD(n_components=100, random_state=42)
item_factors = svd.fit_transform(sparse_matrix)

# Compute item-item similarity (cosine is better than correlation for sparse data)
print("--- Step 4: Computing Similarity ---")
item_similarity = cosine_similarity(item_factors)

# Get movie names in correct order
movie_names = movies.set_index('movieId').loc[movie_encoder.classes_]['title'].tolist()

# Save
joblib.dump(item_similarity, 'corr_matrix.pkl')
joblib.dump(movie_names, 'movie_names.pkl')

print("Done! Model saved with proper item-item similarity.")

# Test
def get_similar_movies(title, n=5):
    """Find similar movies using proper cosine similarity"""
    title_lower = title.lower()
    
    # Find movie
    matches = [i for i, name in enumerate(movie_names) if title_lower in name.lower()]
    
    if not matches:
        print(f"Movie '{title}' not found")
        return []
    
    idx = matches[0]  # Take first match
    similarities = item_similarity[idx]
    top_indices = np.argsort(similarities)[-n-1:-1][::-1]  # Exclude self
    
    results = [(movie_names[i], similarities[i]) for i in top_indices]
    return results

# Demo
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Testing Recommendations:")
    print("="*50)
    
    for test_movie in ["Toy Story (1995)", "The Dark Knight (2008)", "Inception (2010)"]:
        print(f"\n🎬 Similar to '{test_movie}':")
        recs = get_similar_movies(test_movie, 5)
        for title, score in recs:
            print(f"  • {title} ({score:.3f})")