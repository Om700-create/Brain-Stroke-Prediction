# Brain Stroke Prediction using Machine Learning

This project leverages machine learning techniques to predict the likelihood of a person having a stroke based on health-related attributes. Using data such as age, gender, BMI, glucose level, and more, the model is trained to provide predictions on stroke risk. This repository contains the full pipeline from data preprocessing and feature engineering to model training and deployment using Flask.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies](#technologies)
- [Setup Instructions](#setup-instructions)
  - [Clone the Repository](#clone-the-repository)
  - [Install Dependencies](#install-dependencies)
  - [Running the Application](#running-the-application)
- [Usage](#usage)
- [Model Details](#model-details)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
  - [Model Evaluation](#model-evaluation)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This project aims to predict whether an individual is at risk of having a stroke based on their health profile. Using a variety of machine learning models, the application can classify users as at-risk or not based on inputs like age, BMI, glucose level, etc. The end result is a Flask-based web application where users can input their data and receive real-time predictions.

---

## Features

- **Real-time Prediction**: Users can enter their health data and instantly receive a prediction on their stroke risk.
- **Comprehensive Data Pipeline**: Includes data cleaning, feature engineering, model training, and evaluation.
- **Multiple Model Support**: Trained models include Logistic Regression, Random Forest, Gradient Boosting, and more.
- **Web Interface**: A user-friendly Flask web app that serves as the front-end interface for users to interact with the model.
- **Model Evaluation**: Multiple metrics such as accuracy, precision, recall, and F1-score are used to evaluate model performance.

---

## Technologies

- **Python**: Primary programming language for the project.
- **scikit-learn**: For machine learning model building, training, and evaluation.
- **pandas**: For data manipulation and preprocessing.
- **Flask**: Web framework for creating the application and API.
- **HTML, CSS, JavaScript**: For building the web interface.
- **Docker**: For containerizing the application for easier deployment.
- **joblib**: For model serialization and saving/loading.

---

## Setup Instructions

### Clone the Repository

Start by cloning the repository to your local machine:

```bash
git clone https://github.com/your-username/brain-stroke-prediction.git
cd brain-stroke-prediction
Install Dependencies
Create a virtual environment and activate it:

bash
Copy code
python -m venv brain
source brain/bin/activate  # For macOS/Linux
brain\Scripts\activate     # For Windows
Then, install all the required dependencies:

bash
Copy code
pip install -r requirements.txt
This will install the necessary libraries like Flask, scikit-learn, pandas, and others.

Running the Application
To run the Flask web app, execute:

bash
Copy code
python app.py
This will start a local server. Open your browser and navigate to http://127.0.0.1:5000 to use the app.

Usage
Open the web application at http://127.0.0.1:5000.
Enter your health data into the form fields (e.g., age, gender, BMI, hypertension, glucose levels).
Click the "Predict Stroke Risk" button.
The model will predict whether you're at risk of a stroke or not and display the result.
Model Details
Data Preprocessing
Data preprocessing is a critical step that involves cleaning and transforming raw data to ensure that it is suitable for modeling. The steps include:

Handling missing data using SimpleImputer.
Encoding categorical features using OneHotEncoder.
Scaling numerical features using StandardScaler.
Model Training
Various machine learning models are used to train the data, including:

Logistic Regression
Random Forest
Gradient Boosting
Each model is trained using the cleaned and preprocessed data, and hyperparameters are tuned using GridSearchCV to optimize the model's performance.

Model Evaluation
The models are evaluated using:

Accuracy: The proportion of correct predictions.
Precision: The proportion of true positive predictions among all positive predictions.
Recall: The proportion of true positive predictions among all actual positives.
F1-Score: The harmonic mean of precision and recall.
The final model is chosen based on these metrics.

Contributing
We welcome contributions to this project! If you'd like to contribute, follow these steps:

Fork the repository.
Create a new branch for your feature or bug fix.
Commit your changes and push them to your fork.
Submit a pull request describing your changes.
Ensure that your code adheres to the project's style guidelines and passes all tests.

License
This project is licensed under the MIT License - see the LICENSE file for details.

