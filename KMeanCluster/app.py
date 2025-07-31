import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Configure page
st.set_page_config(
    page_title="Customer Segment Predictor",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling and animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-header {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        animation: fadeInDown 1s ease-out;
    }
    
    .subtitle {
        text-align: center;
        color: #000000;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        animation: fadeInUp 1s ease-out 0.3s both;
        font-weight: 600;
        text-shadow: 0 1px 2px rgba(255, 255, 255, 0.8);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
        animation: slideInLeft 0.8s ease-out;
        color: #000000;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        margin: 2rem 0;
        animation: pulse 2s infinite;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.3);
    }
    
    .cluster-info {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        animation: fadeIn 1s ease-out;
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-50px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(79, 70, 229, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.4);
    }
    
    .sidebar .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        color: #000000;
    }
    
    /* Additional color improvements */
    .stMarkdown h3 {
        color: #000000 !important;
        font-weight: 600;
    }
    
    .stSlider > div > div > div > div {
        color: #000000;
    }
    
    .stSlider label {
        color: #000000 !important;
        font-weight: 500;
    }
    
    /* Sidebar text colors */
    .css-1d391kg, .css-1v0mbdj {
        color: #000000 !important;
    }
    
    /* Main content text */
    .stApp .main .block-container {
        color: #000000;
    }
    
    /* Section headers */
    .stMarkdown h3, .stMarkdown h2, .stMarkdown h1 {
        color: #000000 !important;
        font-weight: 700;
        text-shadow: 0 1px 2px rgba(255, 255, 255, 0.8);
    }
    
    /* Ensure metric card headers are black */
    .metric-card h3, .metric-card h4 {
        color: #000000 !important;
        text-shadow: none !important;
        font-weight: 600;
    }
    
    /* Sidebar background for better contrast */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Toggle and slider components */
    .stCheckbox label, .stSelectSlider label {
        color: #000000 !important;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# Sample data with more realistic values
@st.cache_data
def load_data():
    data = {
        "CustomerID": list(range(1, 21)),
        "AnnualIncome": [15, 16, 17, 28, 30, 45, 55, 60, 65, 70, 25, 35, 42, 58, 72, 38, 48, 62, 75, 52],
        "SpendingScore": [39, 81, 6, 77, 40, 76, 6, 94, 3, 72, 45, 82, 15, 88, 25, 65, 55, 90, 20, 75],
        "Age": [19, 21, 20, 23, 24, 30, 32, 33, 35, 36, 25, 28, 31, 29, 34, 27, 26, 37, 38, 33],
    }
    return pd.DataFrame(data)

# Train model
@st.cache_resource
def train_model(df):
    X = df[['AnnualIncome', 'SpendingScore']]
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X)
    return kmeans, X

# Load data and train model
df = load_data()
kmeans, X = train_model(df)

# Header
st.markdown('<h1 class="main-header">🎯 AI Customer Segment Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Harness the power of machine learning to identify customer segments and drive business growth</p>', unsafe_allow_html=True)

# Sidebar for advanced options
with st.sidebar:
    st.markdown("### 🎛️ Advanced Settings")
    
    show_visualization = st.toggle("Show Data Visualization", value=True)
    show_cluster_analysis = st.toggle("Show Cluster Analysis", value=True)
    animation_speed = st.select_slider(
        "Animation Speed",
        options=["Slow", "Medium", "Fast"],
        value="Medium"
    )

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("### 💰 Customer Profile Input")
    
    # Input controls with enhanced styling
    income = st.slider(
        "Annual Income (in thousands)",
        min_value=10,
        max_value=100,
        value=50,
        step=1,
        help="Customer's annual income in thousands of dollars"
    )
    
    score = st.slider(
        "Spending Score (1-100)",
        min_value=1,
        max_value=100,
        value=50,
        step=1,
        help="Spending behavior score based on customer data and purchase patterns"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction button
    if st.button("🔮 Predict Customer Segment", use_container_width=True):
        with st.spinner("Analyzing customer profile..."):
            time.sleep(0.5)  # Simulate processing time
            
            user_input = np.array([[income, score]])
            cluster = kmeans.predict(user_input)[0]
            probabilities = kmeans.transform(user_input)[0]
            confidence = 1 / (1 + min(probabilities))
            
            st.session_state.prediction_made = True
            st.session_state.cluster = cluster
            st.session_state.confidence = confidence
            st.session_state.user_input = user_input

with col2:
    if st.session_state.prediction_made:
        cluster = st.session_state.cluster
        confidence = st.session_state.confidence
        
        # Prediction result card
        st.markdown(f'''
        <div class="prediction-card">
            <h2 style="margin-bottom: 1rem;">🎯 Prediction Result</h2>
            <h1 style="font-size: 3rem; margin: 1rem 0;">Cluster {cluster}</h1>
            <p style="font-size: 1.2rem; opacity: 0.9;">Confidence: {confidence:.1%}</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Cluster interpretation
        cluster_descriptions = {
            0: {
                "name": "💼 Conservative Spenders",
                "description": "Customers with moderate income and careful spending habits",
                "strategy": "Focus on value propositions and practical benefits"
            },
            1: {
                "name": "💎 Premium Customers",
                "description": "High-income customers with high spending scores",
                "strategy": "Target with premium products and personalized experiences"
            },
            2: {
                "name": "🎯 Potential Growth",
                "description": "Customers with growth potential in spending behavior",
                "strategy": "Engage with targeted promotions and loyalty programs"
            }
        }
        
        if cluster in cluster_descriptions:
            desc = cluster_descriptions[cluster]
            st.markdown(f'''
            <div class="cluster-info">
                <h3>{desc["name"]}</h3>
                <p><strong>Profile:</strong> {desc["description"]}</p>
                <p><strong>Strategy:</strong> {desc["strategy"]}</p>
            </div>
            ''', unsafe_allow_html=True)

# Visualization section
if show_visualization and st.session_state.prediction_made:
    st.markdown("---")
    st.markdown("### 📊 Interactive Data Visualization")
    
    # Prepare data for visualization
    df['Cluster'] = kmeans.predict(X)
    
    # Create interactive scatter plot
    fig = px.scatter(
        df, 
        x='AnnualIncome', 
        y='SpendingScore',
        color='Cluster',
        size='Age',
        hover_data=['CustomerID', 'Age'],
        title="Customer Segments Visualization",
        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
    )
    
    # Add user input point
    if 'user_input' in st.session_state:
        user_point = st.session_state.user_input[0]
        fig.add_trace(
            go.Scatter(
                x=[user_point[0]],
                y=[user_point[1]],
                mode='markers',
                marker=dict(
                    size=20,
                    color='gold',
                    symbol='star',
                    line=dict(width=2, color='darkgoldenrod')
                ),
                name='Your Input',
                hovertemplate='<b>Your Customer</b><br>Income: %{x}<br>Score: %{y}<extra></extra>'
            )
        )
    
    fig.update_layout(
        template='plotly_white',
        height=500,
        font=dict(family="Inter", size=12)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Cluster analysis
if show_cluster_analysis:
    st.markdown("---")
    st.markdown("### 📈 Cluster Analysis Dashboard")
    
    df['Cluster'] = kmeans.predict(X)
    cluster_stats = df.groupby('Cluster').agg({
        'AnnualIncome': ['mean', 'std', 'count'],
        'SpendingScore': ['mean', 'std'],
        'Age': ['mean', 'std']
    }).round(2)
    
    # Display metrics in columns
    cols = st.columns(3)
    
    for i, cluster in enumerate(df['Cluster'].unique()):
        with cols[i]:
            cluster_data = df[df['Cluster'] == cluster]
            
            st.markdown(f'''
            <div class="metric-card">
                <h4 style="color: #4f46e5; margin-bottom: 1rem;">Cluster {cluster}</h4>
                <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                    <div><strong>Size:</strong> {len(cluster_data)} customers</div>
                    <div><strong>Avg Income:</strong> ${cluster_data['AnnualIncome'].mean():.0f}k</div>
                    <div><strong>Avg Score:</strong> {cluster_data['SpendingScore'].mean():.0f}</div>
                    <div><strong>Avg Age:</strong> {cluster_data['Age'].mean():.0f} years</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #000000; padding: 2rem;">
    <p style="font-weight: 600;">🚀 Powered by Machine Learning | Built with Streamlit & Scikit-learn</p>
    <p style="font-size: 0.9rem; font-weight: 500;">Enhance your customer understanding with AI-driven insights</p>
</div>
""", unsafe_allow_html=True)