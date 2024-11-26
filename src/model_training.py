import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline  # Add this import
from src.feature_engineering import create_feature_pipeline, add_polynomial_features

def train_models(X, y):
    """
    Train multiple models and perform hyperparameter tuning.
    """
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 1: Create a preprocessing pipeline
    preprocessor = create_feature_pipeline()

    # Step 2: Models to train
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC()
    }

    # Hyperparameters for tuning
    param_grid = {
        'Logistic Regression': {'model__C': [0.1, 1, 10]},  # Use model__C instead of just C
        'Random Forest': {'model__n_estimators': [100, 200], 'model__max_depth': [10, 20]},
        'SVM': {'model__C': [0.1, 1, 10], 'model__kernel': ['linear', 'rbf']}
    }

    best_model = None
    best_score = 0

    # Step 3: Train each model and evaluate
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

        # Use GridSearchCV with the correct parameter grid
        grid_search = GridSearchCV(pipeline, param_grid[model_name], cv=5)
        grid_search.fit(X_train, y_train)
        
        # Evaluate the model on validation data
        y_pred = grid_search.predict(X_val)
        score = accuracy_score(y_val, y_pred)
        
        print(f"{model_name} Accuracy: {score}")
        print(classification_report(y_val, y_pred))
        
        # Save the best model and pipeline
        if score > best_score:
            best_score = score
            best_model = grid_search.best_estimator_

    # Step 4: Save the best model and the pipeline
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(preprocessor, 'models/pipeline.pkl')
    print(f"Best model and pipeline saved!")

def load_model(model_path):
    """
    Load the trained model.
    """
    return joblib.load(model_path)

def load_pipeline(pipeline_path):
    """
    Load the preprocessing pipeline.
    """
    return joblib.load(pipeline_path)










