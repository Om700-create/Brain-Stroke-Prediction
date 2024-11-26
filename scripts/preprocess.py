import os
from src.data_preprocessing import load_data, clean_data, save_data

# Define paths
RAW_TRAIN_PATH = "data/raw/brain_stroke_train.csv"
RAW_TEST_PATH = "data/raw/brain_stroke_test.csv"
PROCESSED_TRAIN_PATH = "data/processed/train_processed.csv"
PROCESSED_TEST_PATH = "data/processed/test_processed.csv"

if __name__ == "__main__":
    # Step 1: Load the raw data
    print("Loading raw datasets...")
    train_df, test_df = load_data(RAW_TRAIN_PATH, RAW_TEST_PATH)
    
    # Step 2: Clean the datasets
    print("Cleaning training dataset...")
    train_df = clean_data(train_df)
    print("Cleaning testing dataset...")
    test_df = clean_data(test_df)

    # Step 3: Save the cleaned datasets
    print("Saving processed datasets...")
    os.makedirs(os.path.dirname(PROCESSED_TRAIN_PATH), exist_ok=True)
    save_data(train_df, PROCESSED_TRAIN_PATH)
    save_data(test_df, PROCESSED_TEST_PATH)

    print("Preprocessing complete! Cleaned datasets saved.")


