import matplotlib.pyplot as plt
import numpy as np


def show_comparison_graph(dt_metrics, rf_metrics, dnn_metrics):

    algorithms = [
        "Decision Tree",
        "Random Forest",
        "DNN"
    ]

    accuracy = [
        dt_metrics["accuracy"],
        rf_metrics["accuracy"],
        dnn_metrics["accuracy"]
    ]

    precision = [
        dt_metrics["precision"],
        rf_metrics["precision"],
        dnn_metrics["precision"]
    ]

    recall = [
        dt_metrics["recall"],
        rf_metrics["recall"],
        dnn_metrics["recall"]
    ]

    f1 = [
        dt_metrics["f1_score"],
        rf_metrics["f1_score"],
        dnn_metrics["f1_score"]
    ]

    x = np.arange(len(algorithms))

    width = 0.20

    plt.figure(figsize=(12,6))

    plt.bar(
        x-0.3,
        accuracy,
        width,
        label="Accuracy"
    )

    plt.bar(
        x-0.1,
        precision,
        width,
        label="Precision"
    )

    plt.bar(
        x+0.1,
        recall,
        width,
        label="Recall"
    )

    plt.bar(
        x+0.3,
        f1,
        width,
        label="F1 Score"
    )

    plt.xticks(
        x,
        algorithms
    )

    plt.ylim(0,1.05)

    plt.ylabel("Performance")

    plt.xlabel("Algorithms")

    plt.title("Performance Comparison of Machine Learning Algorithms")

    plt.grid(axis="y")

    plt.legend()

    plt.tight_layout()

    plt.show()