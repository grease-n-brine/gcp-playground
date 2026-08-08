import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Experiment name
mlflow.set_experiment("MALD_GCP")

with mlflow.start_run():
    # Load Dataset
    data_path = "data/raw/spam.csv"
    df = pd.read_csv(data_path, sep="\t", header=None, names=["label", "text"])
    df["target"] = df["label"].map({"ham": 0, "spam": 1})

    # 3. Train / Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["target"], test_size=0.2, random_state=42
    )

    # 4. Define Hyperparameters & Log to MLflow
    max_features = 3000
    c_param = 1.0

    mlflow.log_param("max_features", max_features)
    mlflow.log_param("C", c_param)

    # 5. Feature Extraction (TF-IDF)
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 6. Train Logistic Regression Model
    model = LogisticRegression(C=c_param)
    model.fit(X_train_vec, y_train)

    # 7. Evaluate Performance
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    print(f"Training Complete! Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

    # 8. Save Model & Vectorizer Local Artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    joblib.dump(model, "models/model.pkl")

    # 9. Register Model to MLflow
    mlflow.sklearn.log_model(model, "model")