# scripts/predict.py

import os
import pandas as pd
from src.model_inference import model  # Ensure this is the trained model from your model_inference.py
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.feature_engineering import preprocess_data  # Assuming this is the correct preprocessing function from feature_engineering.py

# Define paths
INPUT_DATA_PATH = "data/raw/new_data.csv"  # Path to the new dataset
OUTPUT_RESULTS_PATH = "data/processed/predictions.csv"  # Path to save the predictions

def load_new_data():
    # Load the new data
    if os.path.exists(INPUT_DATA_PATH):
        print("Loading new data...")
        new_data = pd.read_csv(INPUT_DATA_PATH)
        return new_data
    else:
        print(f"Error: The file {INPUT_DATA_PATH} does not exist.")
        return None

def preprocess_new_data(data):
    # Preprocess the data for prediction (same preprocessing steps as training)
    preprocessed_data = preprocess_data(data)
    return preprocessed_data

def make_predictions(model, data):
    print("Making predictions...")
    predictions = model.predict(data)
    return predictions

if __name__ == "__main__":
    # Load new data
    new_data = load_new_data()

    if new_data is not None:
        # Preprocess the data
        processed_data = preprocess_new_data(new_data)

        # Make predictions
        predictions = make_predictions(model, processed_data)

        # Save predictions
        os.makedirs(os.path.dirname(OUTPUT_RESULTS_PATH), exist_ok=True)
        pd.DataFrame(predictions, columns=["prediction"]).to_csv(OUTPUT_RESULTS_PATH, index=False)
        print(f"Predictions saved to {OUTPUT_RESULTS_PATH}")

    print("Prediction complete!")
