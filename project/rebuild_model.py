# rebuild_model.py - MEMORY EFFICIENT VERSION
import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import gc

print("--- Step 1: Loading Data ---")
ratings = pd.read_csv('ml-32m/ratings.csv')
movies = pd.read_csv('ml-32m/movies.csv')

print(f"Total ratings: {len(ratings):,}")
print(f"Total movies: {len(movies):,}")

# Less aggressive filtering
print("\n--- Step 2: Filtering ---")
movie_counts = ratings['movieId'].value_counts()
user_counts = ratings['userId'].value_counts()

# Keep movies with at least 10 ratings (reduced from 50)
popular_movies = movie_counts[movie_counts >= 10].index
# Keep users who rated at least 5 movies
active_users = user_counts[user_counts >= 5].index

ratings_filtered = ratings[
    ratings['movieId'].isin(popular_movies) & 
    ratings['userId'].isin(active_users)
]

print(f"Filtered to {ratings_filtered['movieId'].nunique():,} movies")
print(f"Filtered to {ratings_filtered['userId'].nunique():,} users")
print(f"Filtered ratings: {len(ratings_filtered):,}")

# Take a sample if still too large (for speed)
if len(ratings_filtered) > 1000000:
    print("\n--- Sampling 1M ratings for speed ---")
    ratings_filtered = ratings_filtered.sample(n=1000000, random_state=42)

# Free memory
del ratings
gc.collect()

print("\n--- Step 3: Encoding IDs ---")
# Encode movie and user IDs to sequential integers
movie_encoder = LabelEncoder()
user_encoder = LabelEncoder()

movie_indices = movie_encoder.fit_transform(ratings_filtered['movieId'])
user_indices = user_encoder.fit_transform(ratings_filtered['userId'])

n_movies = len(movie_encoder.classes_)
n_users = len(user_encoder.classes_)

print(f"Encoded matrix dimensions: {n_movies} movies x {n_users} users")

print("\n--- Step 4: Creating Sparse Matrix ---")
# Create sparse matrix (memory efficient!)
ratings_values = ratings_filtered['rating'].values.astype(np.float32)

sparse_matrix = csr_matrix(
    (ratings_values, (movie_indices, user_indices)),
    shape=(n_movies, n_users)
)

print(f"Sparse matrix shape: {sparse_matrix.shape}")
print(f"Matrix density: {sparse_matrix.nnz / (n_movies * n_users) * 100:.2f}%")

# Free memory
del ratings_filtered, movie_indices, user_indices, ratings_values
gc.collect()

print("\n--- Step 5: Normalizing (Centering) ---")
# Convert to float32 for memory efficiency
sparse_matrix = sparse_matrix.astype(np.float32)

# Calculate mean rating per movie (only for rated items)
movie_means = np.array(sparse_matrix.mean(axis=1)).flatten()

# Center the data (subtract mean) - this is better for SVD
# We'll do this after converting to dense for SVD

print("\n--- Step 6: Running SVD ---")
# For SVD, we need a dense matrix but can use a smaller one
# Instead, use a memory-efficient approach with incremental/fewer components

# Sample users to reduce dimensions if needed
if n_users > 50000:
    print(f"Too many users ({n_users}), sampling 50,000 for SVD...")
    user_sample = np.random.choice(n_users, size=50000, replace=False)
    matrix_for_svd = sparse_matrix[:, user_sample].toarray()
    matrix_for_svd = matrix_for_svd.astype(np.float32)
else:
    matrix_for_svd = sparse_matrix.toarray().astype(np.float32)

# Center the matrix (subtract movie means)
matrix_for_svd = matrix_for_svd - movie_means[:, np.newaxis]
# Replace NaN with 0
matrix_for_svd = np.nan_to_num(matrix_for_svd, nan=0.0)

print(f"SVD matrix shape: {matrix_for_svd.shape}")

# Run SVD with moderate components
n_components = min(100, min(matrix_for_svd.shape) - 1)
print(f"Using {n_components} components")

svd = TruncatedSVD(n_components=n_components, random_state=42)
item_factors = svd.fit_transform(matrix_for_svd)

print(f"Explained variance: {svd.explained_variance_ratio_.sum():.2%}")
print(f"Item factors shape: {item_factors.shape}")

# Free memory
del sparse_matrix, matrix_for_svd
gc.collect()

print("\n--- Step 7: Computing Movie Similarity ---")
# Compute cosine similarity between movies
similarity_matrix = cosine_similarity(item_factors.astype(np.float32))
print(f"Similarity matrix shape: {similarity_matrix.shape}")

print("\n--- Step 8: Building Movie Name List ---")
# Get movie names in the correct order
movie_ids = movie_encoder.classes_
movie_names_df = movies.set_index('movieId').loc[movie_ids]
movie_names = movie_names_df['title'].tolist()

print(f"Total movies in model: {len(movie_names)}")

# Verify some famous movies are included
test_movies = [
    "godfather", "500 days of summer", "fight club", 
    "insomnia", "toy story", "dark knight", "inception",
    "pulp fiction", "shawshank", "forrest gump"
]

print("\n--- Step 9: Verifying Movies ---")
for search in test_movies:
    found = [name for name in movie_names if search.lower() in name.lower()]
    if found:
        print(f"✅ '{search}' → {found[0]}")
    else:
        print(f"❌ '{search}' not found")

print("\n--- Step 10: Saving Model ---")
# Save the model components
joblib.dump(similarity_matrix, 'corr_matrix.pkl')
joblib.dump(movie_names, 'movie_names.pkl')

# Also save metadata
joblib.dump({
    'movie_means': movie_means,
    'movie_ids': movie_ids,
    'n_components': n_components,
    'explained_variance': float(svd.explained_variance_ratio_.sum())
}, 'model_metadata.pkl')

print("\n✅ Model saved successfully!")
print(f"📁 corr_matrix.pkl: {similarity_matrix.shape}")
print(f"📁 movie_names.pkl: {len(movie_names)} movies")
print(f"📁 model_metadata.pkl: Metadata saved")

# Quick test
print("\n" + "="*50)
print("QUICK TEST:")
print("="*50)

test_movie = "Toy Story (1995)"
if test_movie in movie_names:
    idx = movie_names.index(test_movie)
    similar_idx = np.argsort(similarity_matrix[idx])[-6:-1][::-1]
    
    print(f"\nSimilar to '{test_movie}':")
    for i in similar_idx:
        score = similarity_matrix[idx][i]
        print(f"  • {movie_names[i]} ({score:.3f})")
else:
    print(f"'{test_movie}' not found in final model")