# Credit Card Fraud Detection MLOps Pipeline

## Project Overview

This project implements an end-to-end MLOps pipeline for credit card fraud
detection using XGBoost and the UCI Credit Card Fraud Detection dataset.

The pipeline includes data preprocessing, model training, hyperparameter
tuning, evaluation, FastAPI model serving, automated testing, CI/CD,
Docker containerization, Docker Compose orchestration, and a model
retraining strategy.

## Dataset

Dataset: UCI Credit Card Fraud Detection dataset distributed through Kaggle.

The dataset contains credit card transactions with 30 input features and
a binary target named `Class`.

* Class 0: Legitimate transaction
* Class 1: Fraudulent transaction

The dataset is highly imbalanced, with fraud representing less than 1%
of all transactions.

## Data Preprocessing

The preprocessing pipeline includes:

* Missing-value analysis
* Duplicate-row removal
* Exploratory data analysis
* Standardization of Time and Amount
* Stratified train/validation/test splitting
* Class weighting using XGBoost `scale\_pos\_weight`

## Model

Algorithm: XGBoost Classifier

Hyperparameter tuning was performed using RandomizedSearchCV.

Best hyperparameters:

* n\_estimators: 200
* max\_depth: 5
* learning\_rate: 0.1
* subsample: 1.0
* colsample\_bytree: 0.8

Best cross-validation F1 score: 0.8559

## Final Test Performance

Fraud-class results:

* Precision: 0.90
* Recall: 0.81
* F1-Score: 0.85

Confusion Matrix:

* True Negatives: 28,322
* False Positives: 4
* False Negatives: 9
* True Positives: 38

## API

The trained model is served through FastAPI.

Available endpoints:

* `GET /` - API status
* `GET /health` - model health check
* `POST /predict` - fraud prediction

FastAPI automatically provides interactive API documentation at:

`/docs`

## Run Locally

Install dependencies:

&#x20;   pip install -r requirements.txt


Start the API:

&#x20;   uvicorn app:app --host 0.0.0.0 --port 8000


The API will run on port 8000.

## Docker

Build the Docker image:

&#x20;   docker build -t fraud-detection-api .


Run the container:

&#x20;   docker run -p 8000:8000 fraud-detection-api


## Docker Compose

Start the service:

&#x20;   docker compose up --build


Stop the service:

&#x20;   docker compose down


## Testing

Run automated tests using:

&#x20;   pytest test\_app.py -v


The test suite verifies the root endpoint, health endpoint, and prediction
endpoint.

## CI/CD

GitHub Actions is used for pipeline automation.

### Continuous Integration

The CI workflow:

1. Checks out the repository.
2. Configures Python.
3. Installs dependencies.
4. Runs automated API tests.

### Continuous Deployment

After successful CI completion, the staging workflow:

1. Builds the Docker image.
2. Starts a staging container.
3. Performs an API health check.
4. Cleans up the staging environment.

## Model Maintenance

A retraining script is included in `retrain.py`.

The retraining strategy supports periodic retraining when new approved
transaction data becomes available. Model performance should also be
monitored for degradation in precision, recall, F1-score, and ROC-AUC.

A scheduled GitHub Actions workflow is configured for monthly retraining
pipeline checks. In a production environment, new validated data would be
retrieved from secure persistent storage before candidate-model training
and evaluation.

A new model should be promoted only when it passes validation and meets
the required performance thresholds.

## Project Structure

&#x20;   fraud\_mlops\_project/
    ├── .github/
    │   └── workflows/
    │       ├── ci.yml
    │       ├── cd.yml
    │       └── retrain.yml
    ├── app.py
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements.txt
    ├── retrain.py
    ├── scaler.pkl
    ├── test\_app.py
    ├── xgboost\_fraud\_model.pkl
    └── README.md


## Technologies Used

* Python
* Pandas
* Scikit-learn
* XGBoost
* FastAPI
* Pytest
* Docker
* Docker Compose
* Git
* GitHub
* GitHub Actions

## Conclusion

This project demonstrates an end-to-end MLOps workflow for operationalizing
a machine learning fraud-detection model. It combines model development,
automated testing, containerization, API serving, CI/CD, and a retraining
strategy to improve reproducibility, reliability, and maintainability.

