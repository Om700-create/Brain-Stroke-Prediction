import pandas as pd
from src.model_training import train_models
from src.data_preprocessing import load_data

# Define paths
PROCESSED_TRAIN_PATH = "data/processed/train_processed.csv"
PROCESSED_TEST_PATH = "data/processed/test_processed.csv"

if __name__ == "__main__":
    # Load the processed data
    print("Loading processed datasets...")
    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    test_df = pd.read_csv(PROCESSED_TEST_PATH)

    # Split features and target
    X_train = train_df.drop('stroke', axis=1)  # Assuming 'stroke' is the target
    y_train = train_df['stroke']

    # Train models and select the best one
    train_models(X_train, y_train)





