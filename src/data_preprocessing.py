import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(train_path, test_path):
    """
    Load the raw train and test datasets.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def clean_data(df):
    """
    Clean the dataset: handle missing values, remove duplicates, etc.
    """
    # Drop duplicates
    df = df.drop_duplicates()

    # Handle missing values
    df = df.fillna(df.mean())  # For numeric columns, replace NaN with mean
    df = df.dropna(axis=1, how='all')  # Drop columns that are entirely NaN

    # Convert categorical columns to numerical (if any)
    df = pd.get_dummies(df, drop_first=True)  # One-hot encoding for categorical features

    return df

def save_data(df, file_path):
    """
    Save the cleaned data to a specified path.
    """
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")



