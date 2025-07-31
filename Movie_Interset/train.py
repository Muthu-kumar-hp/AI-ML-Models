import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Sample data
# Columns: Age, Gender (0 = Male, 1 = Female), Target (e.g., 1 = Likes Movie, 0 = Doesn't Like)
# Genre prediction example
data = {
    'age': [18, 25, 30, 22, 35, 40, 28, 50, 45, 60],
    'gender': [0, 1, 0, 1, 0, 1, 1, 0, 1, 0],
    'target': ['Animation', 'Romance', 'Action', 'Animation', 'Action', 
               'Romance', 'Romance', 'Action', 'Action', 'Action']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['age', 'gender']]
y = df['target']

# Split the data (optional, but good for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'movie_model.pkl')

print("Model trained and saved as 'movie_model.pkl'")
