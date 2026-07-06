import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def train_random_forest(pca, encoded_train, encoded_test, y_train, y_test):

    print("\n==============================")
    print("Training Random Forest")
    print("==============================")

    X_train = pca.transform(encoded_train)
    X_test = pca.transform(encoded_test)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        criterion="gini"
    )

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    precision = precision_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/random_forest.pkl")

    print("Random Forest Model Saved")

    return model, metrics