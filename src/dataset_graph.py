import matplotlib.pyplot as plt


def show_dataset_graph(dataset):

    # Check if label column exists
    if "Label" not in dataset.columns:
        print("Label column not found.")
        return

    attack_counts = dataset["Label"].value_counts()

    plt.figure(figsize=(10,6))

    attack_counts.plot(
        kind="bar",
        color="steelblue"
    )

    plt.title("Various Cyber-Attacks Found in Dataset")

    plt.xlabel("Attack Name")

    plt.ylabel("Count")

    plt.xticks(rotation=90)

    plt.tight_layout()

    plt.show()