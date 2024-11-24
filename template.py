import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# List of directories and files required for the Brain Stroke project
list_of_files = [
    "data/brain_stroke.csv",  # Raw dataset placeholder
    "data/processed/__init__.py",  # Processed data placeholder
    "notebooks/01_data_exploration.ipynb",  # Exploratory Data Analysis
    "notebooks/02_feature_engineering.ipynb",  # Feature engineering process
    "notebooks/03_model_training.ipynb",  # Model training and evaluation
    "notebooks/04_model_deployment.ipynb",  # Deployment considerations
    "notebooks/05_results_analysis.ipynb",  # Results interpretation and visualization
    "src/__init__.py",  # Package initialization
    "src/data_processing.py",  # Data cleaning and preprocessing functions
    "src/feature_engineering.py",  # Feature engineering functions
    "src/model.py",  # Model definition and training
    "src/utils.py",  # Utility functions (e.g., for logging and visualization)
    "src/app.py",  # Main application file (e.g., API or dashboard)
    "tests/__init__.py",  # Test initialization
    "tests/test_data_processing.py",  # Unit tests for data processing
    "tests/test_feature_engineering.py",  # Unit tests for feature engineering
    "tests/test_model.py",  # Unit tests for model training
    "README.md",  # Project documentation
    "requirements.txt",  # Python dependencies
    "setup.py",  # Setup script for packaging
    "LICENSE",  # License file (e.g., MIT)
    ".gitignore",  # Exclude unnecessary files from version control
]

# Creating directories and files as per the project structure
for filepath in list_of_files:
    filedir, filename = os.path.split(filepath)

    # Create directories as needed
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir} for the file {filename}")

    # Create the file if it does not exist or is empty
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, "w") as f:
            pass
        logging.info(f"Created empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
