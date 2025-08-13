import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Theme Configuration ---
THEMES = {
    "Vibrant Gradient": {
        "bg_main": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        "bg_container": "rgba(255, 255, 255, 0.95)",
        "color_title": "linear-gradient(135deg, #667eea, #764ba2)",
        "font_family": "'Inter', sans-serif",
        "color_subtitle": "#6b7280",
        "color_prediction_card": "linear-gradient(135deg, #667eea, #764ba2)",
        "color_button": "linear-gradient(135deg, #667eea, #764ba2)",
        "color_metrics": "linear-gradient(135deg, #f8fafc, #e2e8f0)",
        "color_metrics_value": "#667eea",
    },
    "Dark & Sleek": {
        "bg_main": "#1e293b",
        "bg_container": "rgba(30, 41, 59, 0.95)",
        "color_title": "linear-gradient(135deg, #a78bfa, #818cf8)",
        "font_family": "'Roboto Mono', monospace",
        "color_subtitle": "#94a3b8",
        "color_prediction_card": "linear-gradient(135deg, #374151, #1f2937)",
        "color_button": "linear-gradient(135deg, #818cf8, #a78bfa)",
        "color_metrics": "linear-gradient(135deg, #374151, #1f2937)",
        "color_metrics_value": "#818cf8",
    },
    "Light & Minimal": {
        "bg_main": "#f4f7f9",
        "bg_container": "rgba(255, 255, 255, 0.95)",
        "color_title": "linear-gradient(135deg, #1f2937, #4b5563)",
        "font_family": "'Lato', sans-serif",
        "color_subtitle": "#6b7280",
        "color_prediction_card": "linear-gradient(135deg, #d1d5db, #9ca3af)",
        "color_button": "linear-gradient(135deg, #4b5563, #1f2937)",
        "color_metrics": "linear-gradient(135deg, #f9fafb, #f3f4f6)",
        "color_metrics_value": "#1f2937",
    },
    "Ocean Blue": {
        "bg_main": "linear-gradient(135deg, #0072ff, #00c6ff)",
        "bg_container": "rgba(255, 255, 255, 0.95)",
        "color_title": "linear-gradient(135deg, #003366, #0055a4)",
        "font_family": "'Inter', sans-serif",
        "color_subtitle": "#334155",
        "color_prediction_card": "linear-gradient(135deg, #0056b3, #003d7a)",
        "color_button": "linear-gradient(135deg, #00c6ff, #0072ff)",
        "color_metrics": "linear-gradient(135deg, #e0f2ff, #cceeff)",
        "color_metrics_value": "#0056b3",
    }
}

# --- Sidebar for Theme Selection ---
st.sidebar.title("🎨 Theme Settings")
selected_theme = st.sidebar.selectbox("Choose a Theme", list(THEMES.keys()))
theme = THEMES[selected_theme]

# --- Advanced CSS Styling based on selected theme ---
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Roboto+Mono:wght@400;700&family=Lato:wght@400;700&display=swap');
    
    :root {{
        --bg-main: {theme['bg_main']};
        --bg-container: {theme['bg_container']};
        --color-title: {theme['color_title']};
        --font-family: {theme['font_family']};
        --color-subtitle: {theme['color_subtitle']};
        --color-prediction-card: {theme['color_prediction_card']};
        --color-button: {theme['color_button']};
        --color-metrics: {theme['color_metrics']};
        --color-metrics-value: {theme['color_metrics_value']};
    }}

    .stApp {{
        background: var(--bg-main);
        font-family: var(--font-family);
    }}
    
    .main-container {{
        background: var(--bg-container);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }}
    
    .main-title {{
        background: var(--color-title);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }}
    
    .subtitle {{
        text-align: center;
        color: var(--color-subtitle);
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }}
    
    .metric-card {{
        background: var(--color-metrics);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(226, 232, 240, 0.8);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
    }}
    
    .prediction-card {{
        background: var(--color-prediction-card);
        color: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }}
    
    .prediction-digit {{
        font-size: 4rem;
        font-weight: 700;
        margin: 1rem 0;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }}
    
    .confidence-score {{
        font-size: 1.5rem;
        opacity: 0.9;
        margin-bottom: 1rem;
    }}
    
    .stButton > button {{
        background: var(--color-button);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
    }}
    
    .stFileUploader > div > div {{
        background: var(--color-metrics);
        border: 2px dashed #cbd5e0;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }}
    
    .stFileUploader > div > div:hover {{
        border-color: #667eea;
        background: linear-gradient(135deg, #f0f4ff, #e6f0ff);
    }}
    
    .stProgress .st-bo {{
        background: var(--color-button);
        border-radius: 10px;
    }}
    
    .stInfo {{
        background: linear-gradient(135deg, #dbeafe, #bfdbfe);
        border: 1px solid #93c5fd;
        border-radius: 10px;
    }}
    
    .stSuccess {{
        background: linear-gradient(135deg, #dcfce7, #bbf7d0);
        border: 1px solid #86efac;
        border-radius: 10px;
    }}
    
    .css-1d391kg {{
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
    }}
    
    .metric-container {{
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
    }}
    
    .metric-item {{
        text-align: center;
        padding: 1rem;
        background: var(--color-metrics);
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        min-width: 120px;
    }}
    
    .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        color: var(--color-metrics_value);
        display: block;
    }}
    
    .metric-label {{
        font-size: 0.9rem;
        color: var(--color-subtitle);
        margin-top: 0.5rem;
    }}
    
    @keyframes fadeInUp {{
        from {{
            opacity: 0;
            transform: translateY(20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    .fade-in {{
        animation: fadeInUp 0.6s ease-out;
    }}
    
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: var(--color-button);
        border-radius: 10px;
    }}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "advanced_mnist_cnn_model.h5"
CLASSES = [str(i) for i in range(10)]

def create_advanced_model():
    """Create an advanced CNN model with a residual block"""
    
    # Define a residual block
    def residual_block(x, filters):
        # The shortcut connection
        shortcut = x
        
        # Main path
        x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Add the shortcut to the main path
        x = layers.add([x, shortcut])
        x = layers.Activation('relu')(x)
        return x
        
    inputs = layers.Input(shape=(28, 28, 1))
    
    # Initial Conv Block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.25)(x)
    
    # First Residual Block
    x = residual_block(x, 32)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.SpatialDropout2D(0.25)(x)
    
    # Second Residual Block
    x = residual_block(x, 32)
    x = layers.GlobalAveragePooling2D()(x)
    
    # Dense layers with regularization
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Advanced optimizer with learning rate scheduling
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_k_categorical_accuracy')]
    )
    
    return model

@st.cache_resource
def load_or_train_model():
    """Load existing model or train a new advanced model"""
    if os.path.exists(MODEL_PATH):
        st.info("🔄 Loading advanced MNIST model...")
        progress_bar = st.progress(0)
        for i in range(100):
            progress_bar.progress(i + 1)
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Load test data
        (_, _), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        test_images = test_images.astype('float32') / 255.0
        test_images = test_images.reshape(-1, 28, 28, 1)
        test_labels = to_categorical(test_labels, num_classes=10)
        
        st.success("✅ Model loaded successfully!")
        
    else:
        st.info("🚀 Training new advanced MNIST model...")
        st.warning("This may take a few minutes on first run...")
        
        # Load and preprocess data
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        
        # Normalize images
        train_images = train_images.astype('float32') / 255.0
        test_images = test_images.astype('float32') / 255.0
        
        # Reshape for CNN
        train_images = train_images.reshape(-1, 28, 28, 1)
        test_images = test_images.reshape(-1, 28, 28, 1)
        
        # Convert labels to categorical
        train_labels = to_categorical(train_labels, num_classes=10)
        test_labels = to_categorical(test_labels, num_classes=10)
        
        # Data augmentation
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            fill_mode='nearest'
        )
        datagen.fit(train_images)
        
        # Create advanced model
        model = create_advanced_model()
        
        # Advanced callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy', 
                patience=5, 
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=0.0001,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                MODEL_PATH,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train with data augmentation
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        class StreamlitCallback(callbacks.Callback):
            def __init__(self, progress_bar, status_text):
                self.progress_bar = progress_bar
                self.status_text = status_text
                
            def on_epoch_end(self, epoch, logs=None):
                progress = min((epoch + 1) / 30, 1.0)
                self.progress_bar.progress(progress)
                self.status_text.text(f"Epoch {epoch + 1}/30 - Accuracy: {logs['accuracy']:.4f} - Val Accuracy: {logs['val_accuracy']:.4f}")
        
        callbacks_list.append(StreamlitCallback(progress_bar, status_text))
        
        history = model.fit(
            datagen.flow(train_images, train_labels, batch_size=128),
            epochs=30,
            validation_data=(test_images, test_labels),
            callbacks=callbacks_list,
            verbose=0,
            steps_per_epoch=len(train_images) // 128
        )
        
        st.success("🎉 Advanced model trained and saved successfully!")
        
    return model, test_images, test_labels

def preprocess_uploaded_image(image):
    """Advanced preprocessing for uploaded images"""
    # Convert to grayscale and resize
    if image.mode != 'L':
        image = image.convert('L')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    
    # Resize with high-quality resampling
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to array and normalize
    img_array = np.array(image).astype('float32') / 255.0
    
    # Invert if background is white (MNIST digits are white on black)
    if np.mean(img_array) > 0.5:
        img_array = 1 - img_array
    
    return img_array.reshape(1, 28, 28, 1)

def create_prediction_visualization(predictions, image_array, true_label=None):
    """Create advanced visualization using Plotly"""
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Input Image", "Prediction Confidence"),
        specs=[[{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # Add image
    fig.add_trace(
        go.Heatmap(
            z=image_array.reshape(28, 28),
            colorscale='gray',
            showscale=False,
            hovertemplate='<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Add prediction bars
    colors = ['#667eea' if i == np.argmax(predictions) else '#cbd5e0' for i in range(10)]
    
    fig.add_trace(
        go.Bar(
            x=list(range(10)),
            y=predictions,
            marker_color=colors,
            text=[f'{p:.1%}' for p in predictions],
            textposition='auto',
            hovertemplate='Digit: %{x}<br>Confidence: %{y:.1%}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="", showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text="", showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text="Digit", row=1, col=2)
    fig.update_yaxes(title_text="Confidence", range=[0, 1], row=1, col=2)
    
    fig.update_layout(
        height=400,
        showlegend=False,
        template="plotly_white",
        title_text=f"True Label: {true_label}" if true_label is not None else "Prediction Analysis"
    )
    
    return fig

# Main App Interface
st.markdown('<div class="main-container fade-in">', unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-title">🧠 Advanced MNIST Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">State-of-the-art CNN with attention mechanisms for handwritten digit recognition</p>', unsafe_allow_html=True)

# Load model
with st.spinner("Initializing advanced neural network..."):
    model, test_images, test_labels = load_or_train_model()

# Display model performance
test_loss, test_acc, test_top5_acc = model.evaluate(test_images, test_labels, verbose=0)

# Metrics display
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-item">
        <span class="metric-value">{test_acc:.1%}</span>
        <div class="metric-label">Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-item">
        <span class="metric-value">{test_loss:.3f}</span>
        <div class="metric-label">Loss</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-item">
        <span class="metric-value">{test_top5_acc:.1%}</span>
        <div class="metric-label">Top-5 Accuracy</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Main functionality
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🎲 Test with Random Sample")
    if st.button("🔄 Pick Random Test Image", use_container_width=True):
        idx = random.randint(0, len(test_images) - 1)
        sample_img = test_images[idx]
        sample_label = np.argmax(test_labels[idx])
        
        with st.spinner("Analyzing image..."):
            predictions = model.predict(sample_img.reshape(1, 28, 28, 1), verbose=0)[0]
            predicted_digit = np.argmax(predictions)
            confidence = predictions[predicted_digit]
        
        # Display prediction
        st.markdown(f"""
        <div class="prediction-card fade-in">
            <div class="prediction-digit">{predicted_digit}</div>
            <div class="confidence-score">Confidence: {confidence:.1%}</div>
            <div>True Label: {sample_label}</div>
            <div style="margin-top: 1rem; font-size: 0.9rem;">
                {'✅ Correct!' if predicted_digit == sample_label else '❌ Incorrect'}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show visualization
        fig = create_prediction_visualization(predictions, sample_img, sample_label)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show top 3 predictions
        top_indices = predictions.argsort()[-3:][::-1]
        st.markdown("**Top 3 Predictions:**")
        for i, idx in enumerate(top_indices):
            confidence = predictions[idx]
            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            st.write(f"{emoji} **{CLASSES[idx]}** — {confidence:.1%}")

with col2:
    st.markdown("### 📤 Upload Your Own Image")
    uploaded_file = st.file_uploader(
        "Choose an image file (28x28 pixels recommended)", 
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a clear image of a handwritten digit (0-9)"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Preprocess image
        with st.spinner("Processing image..."):
            processed_img = preprocess_uploaded_image(image)
            predictions = model.predict(processed_img, verbose=0)[0]
            predicted_digit = np.argmax(predictions)
            confidence = predictions[predicted_digit]
        
        # Display prediction
        st.markdown(f"""
        <div class="prediction-card fade-in">
            <div class="prediction-digit">{predicted_digit}</div>
            <div class="confidence-score">Confidence: {confidence:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Show visualization
        fig = create_prediction_visualization(predictions, processed_img)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show all predictions
        st.markdown("**All Predictions:**")
        for i, prob in enumerate(predictions):
            bar_width = int(prob * 100)
            st.write(f"**{i}**: {'█' * (bar_width // 5)} {prob:.1%}")

# Model Architecture Info
with st.expander("🏗️ Model Architecture Details"):
    st.markdown("""
    ### Advanced CNN Architecture Features:
    
    - **Residual-like Connections**: Enhanced gradient flow
    - **Batch Normalization**: Faster training and better generalization
    - **Spatial Dropout**: Prevents overfitting in convolutional layers
    - **Global Average Pooling**: Reduces overfitting compared to flatten
    - **Data Augmentation**: Rotation, zoom, shift, and shear transformations
    - **Advanced Optimization**: Adam optimizer with learning rate scheduling
    - **Early Stopping**: Prevents overfitting with patience mechanism
    - **Model Checkpointing**: Saves best performing model
    
    ### Training Features:
    - **Learning Rate Reduction**: Adaptive learning rate based on validation loss
    - **Enhanced Preprocessing**: Contrast enhancement and intelligent inversion
    - **Robust Data Pipeline**: Efficient batch processing with augmentation
    """)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--color-subtitle); font-size: 0.9rem; margin-top: 2rem;">
    🚀 Powered by TensorFlow & Streamlit | Advanced Deep Learning Architecture
</div>
""", unsafe_allow_html=True)