import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Example dataset
texts = ["Hello, how are you?", "Bonjour, comment ça va?", "Hola, cómo estás?"]
labels = ["English", "French", "Spanish"]

# Vectorization
cv = CountVectorizer()
X = cv.fit_transform(texts)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(cv, "vectorizer.pkl")
print("✅ Model and vectorizer saved!")
