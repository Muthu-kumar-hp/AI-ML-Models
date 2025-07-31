import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import io

# Neon-glow background CSS animation
st.markdown("""
<style>
body {
    background: linear-gradient(315deg, #0f0c29, #302b63, #24243e);
    background-size: 600% 600%;
    animation: gradientBG 15s ease infinite;
    color: #00fff7;
}
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.stButton > button {
    background-color: #00fff7;
    color: #000;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px 20px;
    border: none;
    transition: all 0.3s ease;
    box-shadow: 0 0 10px #00fff7, 0 0 20px #00fff7;
}
.stButton > button:hover {
    background-color: #000;
    color: #00fff7;
    border: 1px solid #00fff7;
    box-shadow: 0 0 30px #00fff7;
}
h1, h2, h3 {
    color: #00fff7;
    text-align: center;
}
.sidebar .sidebar-content {
    background-color: rgba(0, 0, 0, 0.4);
}
</style>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1>☕  Coffee Purchase Predictor</h1>", unsafe_allow_html=True)

# --- DATA PREPARATION ---
data = {
    "Weather": ['Sunny', 'Rainy', 'Overcast', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Rainy', 'Sunny', 'Rainy'],
    "TimeOfDay": ['Morning', 'Morning', 'Afternoon', 'Afternoon', 'Evening', 'Morning', 'Morning', 'Afternoon', 'Evening', 'Morning'],
    "SleepQuality": ['Poor', 'Good', 'Poor', 'Good', 'Poor', 'Good', 'Poor', 'Good', 'Good', 'Poor'],
    "Mood": ['Tired', 'Fresh', 'Tired', 'Energetic', 'Tired', 'Fresh', 'Tired', 'Tired', 'Energetic', 'Tired'],
    "BuyCoffee": ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)

# Label Encoding
encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

X = df.drop("BuyCoffee", axis=1)
y = df["BuyCoffee"]

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

# --- SIDEBAR INPUTS ---
st.sidebar.header("👤 Customer Preferences")
weather = st.sidebar.selectbox("🌦️ Weather", ['Sunny', 'Rainy', 'Overcast'])
time = st.sidebar.selectbox("🕒 Time of Day", ['Morning', 'Afternoon', 'Evening'])
sleep = st.sidebar.selectbox("💤 Sleep Quality", ['Good', 'Poor'])
mood = st.sidebar.selectbox("🧠 Mood", ['Fresh', 'Tired', 'Energetic'])

if st.sidebar.button("🔮 Predict"):
    input_df = pd.DataFrame([[weather, time, sleep, mood]], columns=X.columns)
    for col in input_df.columns:
        input_df[col] = encoders[col].transform(input_df[col])
    prediction = clf.predict(input_df)
    result = encoders['BuyCoffee'].inverse_transform(prediction)[0]

    st.markdown(f"<h2>🔍 Prediction: <span style='color:#00fff7'>{result}</span></h2>", unsafe_allow_html=True)

    # Explanation
    st.markdown("### 🧠 AI Insight")
    st.write("Customers who are **tired** and had **poor sleep** are more likely to buy coffee, regardless of weather or time.")

# --- FUTURISTIC DECISION TREE PLOT ---
st.markdown("### 🌐  Decision Tree")

buf = io.BytesIO()
plt.figure(figsize=(12, 6), facecolor='#111')
ax = plt.gca()
ax.set_facecolor('#111')
plot_tree(clf, feature_names=X.columns,
          class_names=encoders['BuyCoffee'].classes_,
          filled=True, rounded=True,
          node_ids=True,
          impurity=True,
          fontsize=10)

# Add neon border to boxes by tweaking edgecolors
for t in ax.get_children():
    if hasattr(t, 'get_bbox'):
        bbox = t.get_bbox()
        if bbox:
            t.set_edgecolor("#00fff7")

plt.savefig(buf, format="png", bbox_inches='tight', dpi=200)
st.image(buf)

st.caption("⚙️ Trained using Decision Tree Classifier (ID3 entropy)")
# --- FOOTER ---
st.markdown("""
<footer style='text-align: center; padding: 20px;'>
    <p style='color: #00fff7;'>© 2025  Coffee Purchase Predictor. All rights reserved MUTHUKUMAR S.</p>
</footer>
""", unsafe_allow_html=True)