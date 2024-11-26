import pandas as pd
from src.model_training import load_model  # Ensure you have this function in model_training.py
import os

# Function to load the model (if saved previously)
def load_trained_model(model_path="models/logistic_regression_model.pkl"):
    """
    Load the trained model from the specified path.
    :param model_path: Path to the saved model.
    :return: The loaded model.
    """
    return load_model(model_path)

# Function to make predictions using the trained model
def make_predictions(new_data: pd.DataFrame):
    """
    Make predictions on the new data.
    :param new_data: Processed new data (must be preprocessed as per the training data).
    :return: Predictions as numpy array.
    """
    # Load the trained model
    model = load_trained_model()

    # Make predictions
    predictions = model.predict(new_data)
    
    return predictions

