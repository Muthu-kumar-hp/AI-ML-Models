from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import re

app = Flask(__name__)

# Global variables to store data and models
movies_df = None
tfidf_matrix = None
cosine_sim = None

def safe_literal_eval(val):
    """Safely evaluate string representations of lists/dicts"""
    if pd.isna(val) or val == '':
        return []
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []

def extract_names(data, key='name', limit=3):
    """Extract names from parsed JSON-like data"""
    if isinstance(data, list):
        return [item.get(key, '') for item in data[:limit] if isinstance(item, dict)]
    return []

def preprocess_data():
    """Load and preprocess the movie data"""
    global movies_df, tfidf_matrix, cosine_sim
    
    try:
        # Load the dataset
        movies_df = pd.read_csv('tmdb_5000_movies.csv')
        
        # Handle missing values
        movies_df = movies_df.fillna('')
        
        # Parse JSON-like columns
        for col in ['genres', 'keywords', 'production_companies', 'production_countries', 'spoken_languages']:
            if col in movies_df.columns:
                movies_df[col] = movies_df[col].apply(safe_literal_eval)
        
        # Extract features for recommendation
        movies_df['genre_names'] = movies_df['genres'].apply(lambda x: extract_names(x, 'name', 5))
        movies_df['keyword_names'] = movies_df['keywords'].apply(lambda x: extract_names(x, 'name', 10))
        movies_df['company_names'] = movies_df['production_companies'].apply(lambda x: extract_names(x, 'name', 3))
        
        # Create combined features for similarity calculation
        movies_df['combined_features'] = (
            movies_df['genre_names'].apply(lambda x: ' '.join(x) if x else '') + ' ' +
            movies_df['keyword_names'].apply(lambda x: ' '.join(x) if x else '') + ' ' +
            movies_df['overview'].fillna('') + ' ' +
            movies_df['company_names'].apply(lambda x: ' '.join(x) if x else '')
        )
        
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(movies_df['combined_features'])
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        print(f"Loaded {len(movies_df)} movies successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

def get_recommendations(title, num_recommendations=10):
    """Get movie recommendations based on title"""
    try:
        # Find the movie index
        idx = movies_df[movies_df['title'].str.lower() == title.lower()].index
        
        if len(idx) == 0:
            # Try partial matching
            partial_matches = movies_df[movies_df['title'].str.lower().str.contains(title.lower(), na=False)]
            if len(partial_matches) == 0:
                return None, "Movie not found in database"
            idx = partial_matches.index[0]
        else:
            idx = idx[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top recommendations (excluding the movie itself)
        movie_indices = [i[0] for i in sim_scores[1:num_recommendations+1]]
        
        recommendations = []
        for i in movie_indices:
            movie = movies_df.iloc[i]
            recommendations.append({
                'title': movie['title'],
                'overview': movie['overview'][:200] + '...' if len(movie['overview']) > 200 else movie['overview'],
                'genres': ', '.join(extract_names(movie['genres'], 'name', 3)),
                'release_date': movie.get('release_date', 'N/A'),
                'vote_average': movie.get('vote_average', 'N/A'),
                'popularity': round(movie.get('popularity', 0), 1)
            })
        
        return recommendations, None
        
    except Exception as e:
        return None, f"Error generating recommendations: {str(e)}"

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/search')
def search():
    """Search for movies (AJAX endpoint)"""
    query = request.args.get('q', '').strip()
    
    if len(query) < 2:
        return jsonify([])
    
    # Search for movies matching the query
    matches = movies_df[movies_df['title'].str.lower().str.contains(query.lower(), na=False)]
    results = matches['title'].head(10).tolist()
    
    return jsonify(results)

@app.route('/recommend', methods=['POST'])
def recommend():
    """Get recommendations"""
    movie_title = request.form.get('movie_title', '').strip()
    
    if not movie_title:
        return render_template('recommend.html', 
                             error="Please enter a movie title")
    
    recommendations, error = get_recommendations(movie_title)
    
    if error:
        return render_template('recommend.html', 
                             error=error, 
                             movie_title=movie_title)
    
    return render_template('recommend.html', 
                         recommendations=recommendations, 
                         movie_title=movie_title)

@app.route('/api/recommend/<movie_title>')
def api_recommend(movie_title):
    """API endpoint for recommendations"""
    recommendations, error = get_recommendations(movie_title)
    
    if error:
        return jsonify({'error': error}), 404
    
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    print("Loading movie data...")
    if preprocess_data():
        print("Starting Flask app...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load movie data. Please check your CSV file.")
