from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained models and scaler
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")  # Assuming you saved the scaler used in training
label_encoders = joblib.load("label_encoders.pkl")  # Assuming label encoders are saved

# Define routes
@app.route('/')
def home():
    return render_template('home.html')  # Render home page

@app.route('/predict')
def index():
    return render_template('index.html')  # Render the input form page

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect data from form
        user_input = {
            'Genetics': request.form['genetics'],
            'Hormonal Changes': request.form['hormonal_changes'],
            'Medications & Treatments': request.form['medications'],
            'Nutritional Deficiencies ': request.form['nutritional_deficiency'],
            'Stress': request.form['stress'],
            'Poor Hair Care Habits ': request.form['hair_care'],
            'Environmental Factors': request.form['environmental_factors'],
            'Smoking': request.form['smoking'],
            'Weight Loss ': request.form['weight_loss'],
            'Age': float(request.form['age'])
        }

        # Preprocess input
        processed_input = preprocess_input(user_input)

        # Scale input
        scaled_input = scaler.transform(processed_input)

        # Make prediction with Random Forest
        rf_prediction = rf_model.predict(scaled_input)[0]

        return render_template('result.html', prediction=rf_prediction)  # Display prediction on results page

# Preprocess input (same as before)
def preprocess_input(user_input):
    processed_input = []
    for column, value in user_input.items():
        if column in label_encoders:
            processed_value = label_encoders[column].transform([value])[0]
        else:
            processed_value = value
        processed_input.append(processed_value)

    return np.array(processed_input).reshape(1, -1)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
