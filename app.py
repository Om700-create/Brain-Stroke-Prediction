from flask import Flask, request, jsonify, render_template
import pandas as pd
from src.model_inference import make_predictions
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the POST request
        data = request.get_json()

        # Convert data to DataFrame for prediction (ensure the keys match your input features)
        input_data = pd.DataFrame(data)

        # Make predictions
        predictions = make_predictions(input_data)

        # Return predictions as a JSON response
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run the app on port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)
