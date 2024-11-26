import os

def create_project_structure(base_dir="."):
    # Define the folder and file structure
    structure = {
        "data": {
            "raw": ["brain_stroke_train.csv", "brain_stroke_test.csv"],
            "processed": [],
        },
        "notebooks": ["eda.ipynb"],
        "src": [
            "__init__.py",
            "data_preprocessing.py",
            "feature_engineering.py",
            "model_training.py",
            "model_inference.py",
            "utils.py",
        ],
        "models": [],
        "logs": [],
        "scripts": ["preprocess.py", "train.py", "predict.py"],
        "": ["app.py", "requirements.txt", "README.md", ".gitignore"],
    }

    # Create directories and files
    for folder, contents in structure.items():
        folder_path = os.path.join(base_dir, folder)
        if folder:  # Create the folder if it's not the root level
            os.makedirs(folder_path, exist_ok=True)
        for item in contents:
            # Create files inside the specified folder
            file_path = os.path.join(folder_path, item) if folder else os.path.join(base_dir, item)
            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    pass  # Create an empty file
    print(f"Project structure created at: {os.path.abspath(base_dir)}")

if __name__ == "__main__":
    create_project_structure()
