import os
import pandas as pd
import matplotlib.pyplot as plt

from tkinter import filedialog, messagebox


def save_results(dt, rf, dnn):

    folder = filedialog.askdirectory(
        title="Select Folder to Save Results"
    )

    if folder == "":
        return

    data = {
        "Algorithm": [
            "Decision Tree",
            "Random Forest",
            "DNN"
        ],
        "Accuracy": [
            dt["accuracy"],
            rf["accuracy"],
            dnn["accuracy"]
        ],
        "Precision": [
            dt["precision"],
            rf["precision"],
            dnn["precision"]
        ],
        "Recall": [
            dt["recall"],
            rf["recall"],
            dnn["recall"]
        ],
        "F1 Score": [
            dt["f1_score"],
            rf["f1_score"],
            dnn["f1_score"]
        ]
    }

    df = pd.DataFrame(data)

    csv_path = os.path.join(folder, "comparison_results.csv")

    df.to_csv(csv_path, index=False)

    # Save Report

    report = os.path.join(folder, "classification_report.txt")

    with open(report, "w") as f:

        f.write("CYBER ATTACK DETECTION RESULTS\n")
        f.write("=" * 50 + "\n\n")

        for _, row in df.iterrows():

            f.write(f"Algorithm : {row['Algorithm']}\n")
            f.write(f"Accuracy  : {row['Accuracy']:.4f}\n")
            f.write(f"Precision : {row['Precision']:.4f}\n")
            f.write(f"Recall    : {row['Recall']:.4f}\n")
            f.write(f"F1 Score  : {row['F1 Score']:.4f}\n")
            f.write("-" * 40 + "\n")

    # Save Graph

    algorithms = df["Algorithm"]

    accuracy = df["Accuracy"]

    plt.figure(figsize=(8,5))

    plt.bar(algorithms, accuracy)

    plt.title("Algorithm Accuracy Comparison")

    plt.ylabel("Accuracy")

    plt.grid(axis="y")

    plt.tight_layout()

    graph = os.path.join(folder, "comparison_graph.png")

    plt.savefig(graph)

    plt.close()

    messagebox.showinfo(
        "Success",
        f"Results saved successfully.\n\n{folder}"
    )