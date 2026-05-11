# app.py - FINAL WORKING VERSION
from fastapi import FastAPI, Request, Query
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import joblib
import numpy as np
from typing import Optional
import re

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load model
print("Loading model components...")
try:
    corr_matrix = joblib.load('corr_matrix.pkl')
    movie_names = joblib.load('movie_names.pkl')
    print(f"✅ Loaded {len(movie_names)} movies successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    movie_names = ["The Dark Knight (2008)", "Inception (2010)", "Toy Story (1995)"]
    corr_matrix = np.eye(len(movie_names))

def smart_search(query: str, movie_list: list, max_results: int = 5) -> list:
    """Intelligent movie search with multiple strategies"""
    query = query.lower().strip()
    
    # Strategy 1: Exact match
    exact_matches = [name for name in movie_list if query == name.lower()]
    if exact_matches:
        return exact_matches[:max_results]
    
    # Strategy 2: Query is substring of movie name
    substring_matches = [name for name in movie_list if query in name.lower()]
    if substring_matches:
        return substring_matches[:max_results]
    
    # Strategy 3: Word-by-word matching
    query_words = query.split()
    scored = []
    
    for name in movie_list:
        name_lower = name.lower()
        score = sum(1 for word in query_words if word in name_lower)
        if score > 0:
            scored.append((name, score))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    return [match[0] for match in scored[:max_results]]

def get_recommendations(user_input: str, top_n: int = 5) -> dict:
    """Get recommendations with proper error handling"""
    
    # Validate input
    if not user_input or len(user_input.strip()) < 2:
        return {
            "found": False,
            "search_query": user_input,
            "matched_movie": None,
            "alternatives": [],
            "recommendations": [],
            "error": "Please enter at least 2 characters to search"
        }
    
    # Search for the movie
    matches = smart_search(user_input, list(movie_names), max_results=10)
    
    if not matches:
        return {
            "found": False,
            "search_query": user_input,
            "matched_movie": None,
            "alternatives": [],
            "recommendations": [],
            "error": f"No movies found matching '{user_input}'. Try a different search term."
        }
    
    found_movie = matches[0]
    alternatives = matches[1:4] if len(matches) > 1 else []
    
    try:
        idx = list(movie_names).index(found_movie)
        movie_correlations = corr_matrix[idx]
        
        # Get top N similar movies
        similar_indices = np.argsort(movie_correlations)[-top_n-1:-1][::-1]
        
        recs = []
        for i in similar_indices:
            score = float(movie_correlations[i])
            # Normalize score to 0-100
            normalized_score = max(0, min(100, int(score * 100)))
            recs.append({
                "title": str(movie_names[i]),
                "score": normalized_score
            })
        
        return {
            "found": True,
            "search_query": user_input,
            "matched_movie": found_movie,
            "alternatives": alternatives,
            "recommendations": recs,
            "error": None
        }
    
    except Exception as e:
        return {
            "found": False,
            "search_query": user_input,
            "matched_movie": None,
            "alternatives": [],
            "recommendations": [],
            "error": f"Error generating recommendations. Please try again."
        }

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "search_name": None,
        "recommendations": [],
        "error": None,
        "alternatives": []
    })

@app.get("/recommend", response_class=HTMLResponse)
def recommend(
    request: Request, 
    movie: str = Query(default="", min_length=1, max_length=200)
):
    result = get_recommendations(movie)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "search_name": result.get("matched_movie") if result["found"] else movie,
        "recommendations": result.get("recommendations", []),
        "error": result.get("error"),
        "alternatives": result.get("alternatives", [])
    })

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "movies_loaded": len(movie_names),
        "model_ready": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)