import streamlit as st
import joblib

# Load trained model + vectorizer
model = joblib.load("model.pkl")
cv = joblib.load("vectorizer.pkl")

# Streamlit Page Config
st.set_page_config(page_title="🌍 Language Detector", page_icon="🌐", layout="centered")

# Header Section
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 40px;
        color: #4CAF50;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #666;
    }
    </style>
    <div class="title">🌍 Language Detection App</div>
    <div class="subtitle">Type text in any language & let AI detect it instantly!</div>
    <br>
    """,
    unsafe_allow_html=True,
)

# Input Section
with st.container():
    st.markdown("### ✍️ Enter your text below:")
    user_input = st.text_area("", height=150, placeholder="Type something...")

# Button + Prediction
if st.button("🔍 Detect Language"):
    if user_input.strip() != "":
        user_input_vectorized = cv.transform([user_input])
        prediction = model.predict(user_input_vectorized)[0]

        st.success(f"✅ Detected Language: **{prediction}**")
    else:
        st.warning("⚠️ Please enter some text to detect.")

# Extra Info Section
st.markdown("---")
st.markdown("✨ *This app uses Machine Learning (CountVectorizer + Naive Bayes) to detect languages.*")

# Footer
st.markdown(
    """
    <style>
    .footer {
        text-align: center;
        color: #999;
        font-size: 14px;
        margin-top: 40px;
    }
    </style>
    <div class="footer">Made with ❤️ using Streamlit</div>
    """,
    unsafe_allow_html=True,
)
