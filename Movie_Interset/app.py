from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load your trained model (ensure this file exists in the same folder)
model = joblib.load('movie_model.pkl')

@app.route('/')
def home():
    return render_template('muthu.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    age = int(request.form['age'])
    gender = int(request.form['gender'])  # Expecting 0 for Male, 1 for Female

    # Run prediction
    prediction = model.predict([[age, gender]])

    # Convert numeric gender to text for display
    gender_text = 'Male' if gender == 0 else 'Female'

    # Render the result page
    return render_template('result.html', age=age, gender=gender_text, prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
