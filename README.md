# CineMatch - AI Movie Recommendation System

A machine learning-powered movie recommendation system that suggests similar movies based on user preferences. Built with Singular Value Decomposition (SVD) and trained on the MovieLens 32M dataset containing 32 million ratings across 200,000 users and 87,000 movies.

## Features

- AI-powered recommendations using SVD matrix factorization
- Real-time results from pre-computed similarity matrices
- Smart fuzzy search supporting partial movie names
- Modern glassmorphism user interface with smooth animations
- 23,031 movies indexed from 32 million ratings
- FastAPI backend for high-performance request handling
- Responsive design compatible with all devices

## How It Works

The system processes user rating patterns through several stages:

1. Loads and filters the MovieLens 32M dataset to retain active users and popular movies
2. Constructs a sparse user-movie rating matrix for memory efficiency
3. Applies TruncatedSVD to reduce dimensionality to 50 latent features
4. Computes cosine similarity between all movie feature vectors
5. For any queried movie, returns the five most similar titles based on vector proximity

## Screenshots

<img width="2256" height="1504" alt="image" src="https://github.com/user-attachments/assets/eaba1024-b4b3-43e0-a43d-a7fe4313a6b9" />


## Technology Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI |
| Frontend | HTML5, CSS3, Jinja2 |
| ML Model | Scikit-learn (TruncatedSVD) |
| Data Processing | Pandas, NumPy, SciPy |
| Serialization | Joblib |
| Styling | Bootstrap 5.3, Custom CSS |

## Installation

### Prerequisites

- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- 2GB free disk space

### Setup

1. Clone the repository
```bash
git clone https://github.com/BSSE23063/CineMatch.git
cd CineMatch
Install dependencies

bash
pip install -r requirements.txt
Download the MovieLens 32M dataset from GroupLens

bash
wget https://files.grouplens.org/datasets/movielens/ml-32m.zip
unzip ml-32m.zip
Train the model

bash
python rebuild_model.py
Download Bootstrap assets

bash
python download_bootstrap.py
Launch the application

bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
Open in browser at http://127.0.0.1:8000

Project Structure
text
CineMatch/
├── main.py                    # FastAPI web application
├── recommender.py             # Recommendation engine
├── rebuild_model.py           # Model training script
├── download_bootstrap.py      # Bootstrap downloader
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
├── templates/
│   └── index.html            # Web interface template
└── static/
    ├── css/
    │   ├── style.css         # Custom styling
    │   └── bootstrap.min.css # Bootstrap framework
    └── js/
        └── bootstrap.bundle.min.js
Model Performance
Training Data: 32 million ratings

Movies Indexed: 23,031

Latent Features: 50

Variance Explained: 32.5%

Recommendation Speed: Less than 100ms per query

Example Recommendations
Input Movie	Top Recommendations
The Dark Knight	Batman Begins, Inception, The Prestige, Interstellar, Memento
Toy Story	Toy Story 2, A Bug's Life, Shrek, Monsters Inc., Finding Nemo
The Godfather	The Godfather Part II, Goodfellas, Scarface, Casino, Heat
Pulp Fiction	Reservoir Dogs, Jackie Brown, Kill Bill, Fargo, The Big Lebowski
Challenges and Solutions
Memory Management: The initial implementation attempted to allocate 17.4 GB for a dense pivot table. This was resolved by implementing sparse matrices using SciPy CSR format, reducing memory usage by over 99%.

Missing Popular Movies: Random sampling of 200,000 ratings caused well-known movies to fall below the filtering threshold. This was addressed by implementing a guaranteed inclusion list and using active user-based sampling instead of random sampling.

Data Sparsity: Using fillna(0) corrupted rating patterns by treating unwatched movies as zero-rated. The solution involved proper handling with mean imputation and sparse matrix representations that preserve the distinction between missing and actual ratings.

Search Accuracy: Exact name matching failed for partial queries like "batman" or "god father". A multi-strategy fuzzy search algorithm was implemented with exact matching, substring matching, and word-by-word scoring.

Large File Storage: The trained model files exceeded GitHub's 100 MB limit. These files are excluded from the repository and must be generated locally using the training script.
