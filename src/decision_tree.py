import os
import joblib
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def train_decision_tree(encoded_train, encoded_test, y_train, y_test):
    """
    Train Decision Tree using PCA-reduced AutoEncoder features.
    """

    print("\n==============================")
    print("Decision Tree + PCA")
    print("==============================")

    # Apply PCA
    pca = PCA(n_components=min(5, encoded_train.shape[1]))

    X_train_pca = pca.fit_transform(encoded_train)
    X_test_pca = pca.transform(encoded_test)

    print(f"PCA Reduced Features : {X_train_pca.shape[1]}")

    # Train Decision Tree
    model = DecisionTreeClassifier(
        criterion="gini",
        random_state=42
    )

    model.fit(X_train_pca, y_train)

    predictions = model.predict(X_test_pca)

    # Calculate Metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print("\nDecision Tree Results")
    print("----------------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # Save model
    # Save model
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/decision_tree.pkl")
    joblib.dump(pca, "models/pca.pkl")

    print("\nModels saved successfully:")
    print("Decision Tree -> models/decision_tree.pkl")
    print("PCA -> models/pca.pkl")

    return model, pca, {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }