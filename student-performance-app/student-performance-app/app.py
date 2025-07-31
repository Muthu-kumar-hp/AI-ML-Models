from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model
model_performance = pickle.load(open('model_performance.pkl', 'rb'))

# Label decoding (optional, depends on how your model was trained)
label_map = {0: 'Average', 1: 'Poor', 2: 'Good'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        study_hours = float(request.form['study_hours'])
        attendance = float(request.form['attendance_percentage'])
        previous_grade = float(request.form['previous_grade_numeric'])

        # Input to model
        input_data = [[study_hours, attendance, previous_grade]]
        prediction = model_performance.predict(input_data)
        output = label_map.get(prediction[0], "Unknown")

        return render_template('result.html', prediction=output)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)