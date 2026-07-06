from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def calculate_metrics(y_true, y_pred):

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred) * 100,
        "Precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ) * 100,
        "Recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ) * 100,
        "F1 Score": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ) * 100
    }

    return metrics


def print_metrics(name, metrics):

    print("\n==============================")
    print(name)
    print("==============================")

    for key, value in metrics.items():
        print(f"{key:12}: {value:.2f}%")