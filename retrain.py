
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.getenv(
    "FRAUD_DATA_PATH",
    "/content/creditcard_data/creditcard.csv"
)

MODEL_PATH = os.path.join(BASE_DIR, "xgboost_fraud_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")


def retrain():
    print("Loading training data...")
    df = pd.read_csv(DATA_PATH)

    # Clean duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Split before fitting preprocessing to prevent leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    # Fit scaler ONLY on training data
    scaler = StandardScaler()

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[["Time", "Amount"]] = scaler.fit_transform(
        X_train[["Time", "Amount"]]
    )

    X_test[["Time", "Amount"]] = scaler.transform(
        X_test[["Time", "Amount"]]
    )

    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()
    scale_pos_weight = negative / positive

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    print("Training updated XGBoost model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print("\n=== RETRAINED MODEL RESULTS ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    # Save updated artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("\nUpdated model saved:", MODEL_PATH)
    print("Updated scaler saved:", SCALER_PATH)


if __name__ == "__main__":
    retrain()
