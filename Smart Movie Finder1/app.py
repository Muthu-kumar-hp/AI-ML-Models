import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="🎬 Smart Movie Finder", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for futuristic neon theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    /* Root variables */
    :root {
        --primary-neon: #00ffff;
        --secondary-neon: #ff00ff;
        --accent-neon: #00ff41;
        --bg-dark: #0a0a0a;
        --bg-card: rgba(20, 20, 30, 0.8);
        --text-primary: #ffffff;
        --text-secondary: #b0b0b0;
        --glow-intensity: 0 0 20px;
    }
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        background-attachment: fixed;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Animated background particles */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(2px 2px at 20px 30px, var(--primary-neon), transparent),
            radial-gradient(2px 2px at 40px 70px, var(--secondary-neon), transparent),
            radial-gradient(1px 1px at 90px 40px, var(--accent-neon), transparent),
            radial-gradient(1px 1px at 130px 80px, var(--primary-neon), transparent),
            radial-gradient(2px 2px at 160px 30px, var(--secondary-neon), transparent);
        background-repeat: repeat;
        background-size: 200px 100px;
        animation: particleFloat 20s linear infinite;
        opacity: 0.3;
        z-index: -1;
    }
    
    @keyframes particleFloat {
        0% { transform: translateY(0px) translateX(0px); }
        33% { transform: translateY(-10px) translateX(10px); }
        66% { transform: translateY(5px) translateX(-5px); }
        100% { transform: translateY(0px) translateX(0px); }
    }
    
    /* Title styling */
    h1 {
        font-family: 'Orbitron', monospace !important;
        font-weight: 900 !important;
        text-align: center !important;
        color: var(--text-primary) !important;
        text-shadow: var(--glow-intensity) var(--primary-neon) !important;
        font-size: 3.5rem !important;
        margin-bottom: 0.5rem !important;
        animation: titleGlow 3s ease-in-out infinite alternate;
    }
    
    @keyframes titleGlow {
        from { text-shadow: 0 0 20px var(--primary-neon), 0 0 30px var(--primary-neon), 0 0 40px var(--primary-neon); }
        to { text-shadow: 0 0 30px var(--secondary-neon), 0 0 40px var(--secondary-neon), 0 0 50px var(--secondary-neon); }
    }
    
    /* Subtitle styling */
    .stMarkdown p {
        color: var(--text-secondary) !important;
        text-align: center !important;
        font-size: 1.2rem !important;
        margin-bottom: 2rem !important;
        animation: fadeInUp 1s ease-out;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Container styling */
    .main .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 800px !important;
        background: var(--bg-card);
        border-radius: 20px;
        border: 1px solid rgba(0, 255, 255, 0.3);
        box-shadow: 
            0 8px 32px rgba(0, 255, 255, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        animation: containerFloat 6s ease-in-out infinite;
    }
    
    @keyframes containerFloat {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: rgba(0, 0, 0, 0.6) !important;
        border: 2px solid var(--primary-neon) !important;
        border-radius: 15px !important;
        color: var(--text-primary) !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: var(--glow-intensity) var(--primary-neon);
    }
    
    .stSelectbox > div > div:hover {
        border-color: var(--secondary-neon) !important;
        box-shadow: var(--glow-intensity) var(--secondary-neon);
        transform: translateY(-2px);
    }
    
    .stSelectbox label {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 0 0 10px var(--accent-neon);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, var(--primary-neon), var(--secondary-neon)) !important;
        color: var(--bg-dark) !important;
        border: none !important;
        border-radius: 25px !important;
        font-family: 'Orbitron', monospace !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        padding: 0.75rem 2rem !important;
        width: 100% !important;
        margin-top: 1rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 2px !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 10px 25px rgba(0, 255, 255, 0.4) !important;
        background: linear-gradient(45deg, var(--secondary-neon), var(--accent-neon)) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(1.02) !important;
    }
    
    /* Button ripple effect */
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.5);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:active::before {
        width: 300px;
        height: 300px;
    }
    
    /* Subheader styling */
    h3 {
        color: var(--text-primary) !important;
        font-family: 'Orbitron', monospace !important;
        text-align: center !important;
        margin: 2rem 0 1rem 0 !important;
        text-shadow: 0 0 15px var(--accent-neon);
        animation: slideInFromLeft 0.8s ease-out;
    }
    
    @keyframes slideInFromLeft {
        from { opacity: 0; transform: translateX(-50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Movie list styling */
    .stMarkdown div[data-testid="stMarkdownContainer"] p {
        background: rgba(0, 255, 255, 0.1) !important;
        border-left: 4px solid var(--primary-neon) !important;
        padding: 1rem 1.5rem !important;
        margin: 0.5rem 0 !important;
        border-radius: 0 15px 15px 0 !important;
        color: var(--text-primary) !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        overflow: hidden !important;
        animation: slideInFromRight 0.6s ease-out forwards;
        opacity: 0;
        transform: translateX(50px);
    }
    
    .stMarkdown div[data-testid="stMarkdownContainer"] p:nth-child(1) { animation-delay: 0.1s; }
    .stMarkdown div[data-testid="stMarkdownContainer"] p:nth-child(2) { animation-delay: 0.2s; }
    .stMarkdown div[data-testid="stMarkdownContainer"] p:nth-child(3) { animation-delay: 0.3s; }
    .stMarkdown div[data-testid="stMarkdownContainer"] p:nth-child(4) { animation-delay: 0.4s; }
    .stMarkdown div[data-testid="stMarkdownContainer"] p:nth-child(5) { animation-delay: 0.5s; }
    
    @keyframes slideInFromRight {
        to { opacity: 1; transform: translateX(0); }
    }
    
    .stMarkdown div[data-testid="stMarkdownContainer"] p:hover {
        background: rgba(255, 0, 255, 0.2) !important;
        border-left-color: var(--secondary-neon) !important;
        transform: translateX(10px) !important;
        box-shadow: 0 5px 15px rgba(255, 0, 255, 0.3) !important;
    }
    
    .stMarkdown div[data-testid="stMarkdownContainer"] p::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .stMarkdown div[data-testid="stMarkdownContainer"] p:hover::before {
        left: 100%;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: var(--primary-neon) var(--secondary-neon) var(--accent-neon) transparent !important;
        animation: spinGlow 1s linear infinite !important;
    }
    
    @keyframes spinGlow {
        0% { 
            transform: rotate(0deg);
            box-shadow: 0 0 20px var(--primary-neon);
        }
        33% { 
            box-shadow: 0 0 20px var(--secondary-neon);
        }
        66% { 
            box-shadow: 0 0 20px var(--accent-neon);
        }
        100% { 
            transform: rotate(360deg);
            box-shadow: 0 0 20px var(--primary-neon);
        }
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-dark);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, var(--primary-neon), var(--secondary-neon));
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, var(--secondary-neon), var(--accent-neon));
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Responsive design */
    @media (max-width: 768px) {
        h1 { font-size: 2.5rem !important; }
        .main .block-container { padding: 1rem !important; }
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🎬 Smart Movie Finder")
st.markdown("Get personalized movie recommendations based on plot overviews!")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")
    df['overview'] = df['overview'].fillna('')
    return df

@st.cache_data
def compute_similarity(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(df.index, index=df['original_title']).drop_duplicates()
    return cosine_sim, indices

# Recommendation function
def recommend(title, cosine_sim, indices, df, num=5):
    title = title.strip()
    if title not in indices:
        return ["Movie not found."]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['original_title'].iloc[movie_indices].tolist()

# Load and prepare data
df = load_data()
cosine_sim, indices = compute_similarity(df)
movie_list = sorted(df['original_title'].unique())

# UI: Select a movie
selected_movie = st.selectbox("Choose a movie you like:", movie_list)

# On button click, show recommendations
if st.button("🚀 Find Similar Movies"):
    with st.spinner("Scanning the digital movie matrix..."):
        results = recommend(selected_movie, cosine_sim, indices, df)
        st.subheader(f"Movies similar to **{selected_movie}**:")
        for i, title in enumerate(results, 1):
            st.write(f"{i}. {title}")