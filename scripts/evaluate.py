# scripts/evaluate.py
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from src.data_preprocessing import load_data

# Path to the saved best model
MODEL_PATH = 'models/best_model.pkl'
TEST_PATH = "data/processed/test_processed.csv"  # Adjust the path to your test dataset

if __name__ == "__main__":
    # Step 1: Load the best model and preprocessing pipeline
    print("Loading best model and pipeline...")
    best_model = joblib.load(MODEL_PATH)

    # Step 2: Load the test dataset
    print("Loading test dataset...")
    test_df = load_data(TEST_PATH)
    X_test = test_df.drop('stroke', axis=1)  # Features
    y_test = test_df['stroke']  # Target labels

    # Step 3: Preprocess the test data (using the saved pipeline)
    print("Preprocessing test data...")
    X_test_transformed = best_model.named_steps['preprocessor'].transform(X_test)

    # Step 4: Make predictions with the loaded model
    print("Making predictions on the test data...")
    y_pred = best_model.named_steps['model'].predict(X_test_transformed)

    # Step 5: Evaluate the model
    print("Evaluating model performance...")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Optionally, save the evaluation results to a file
    with open('evaluation_results.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Classification Report:\n{classification_report(y_test, y_pred)}\n")
        f.write(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

    print("Evaluation complete! Results saved to 'evaluation_results.txt'.")
