from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    message = ''
    if request.method == 'POST':
        message = request.form['message'].strip().lower()
        vector = vectorizer.transform([message])
        pred = model.predict(vector)[0]
        prediction = 'Spam' if pred == 1 else 'Not Spam'
    return render_template('index.html', prediction=prediction, message=message)

if __name__ == '__main__':
    app.run(debug=True)
