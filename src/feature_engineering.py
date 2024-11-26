import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def create_feature_pipeline():
    """
    Create a feature pipeline to handle missing data, scaling, encoding, and polynomial feature generation.
    """
    # Identify categorical and numeric columns
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']  # Update with your columns
    numeric_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']  # Update with your columns
    
    # Define transformations for numeric and categorical columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with the mean
        ('scaler', StandardScaler())  # Standardize the numeric features
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing categorical values
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))  # One-Hot Encoding
    ])
    
    # Combine the transformers into one ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    return preprocessor

def add_polynomial_features(X, degree=2):
    """
    Add polynomial features to the dataset.
    """
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Create a DataFrame with the generated polynomial features
    X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(), index=X.index)
    return X_poly_df






