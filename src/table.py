from tkinter import Toplevel
from tkinter import ttk


def show_table(root, dt, rf, dnn):

    window = Toplevel(root)

    window.title("Performance Comparison Table")

    window.geometry("850x250")

    tree = ttk.Treeview(
        window,
        columns=(
            "Algorithm",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ),
        show="headings"
    )

    # Headings
    tree.heading("Algorithm", text="Algorithm")
    tree.heading("Accuracy", text="Accuracy")
    tree.heading("Precision", text="Precision")
    tree.heading("Recall", text="Recall")
    tree.heading("F1 Score", text="F1 Score")

    # Column Widths
    tree.column("Algorithm", width=180, anchor="center")
    tree.column("Accuracy", width=120, anchor="center")
    tree.column("Precision", width=120, anchor="center")
    tree.column("Recall", width=120, anchor="center")
    tree.column("F1 Score", width=120, anchor="center")

    # Decision Tree
    tree.insert(
        "",
        "end",
        values=(
            "Decision Tree",
            f"{dt['accuracy']:.4f}",
            f"{dt['precision']:.4f}",
            f"{dt['recall']:.4f}",
            f"{dt['f1_score']:.4f}"
        )
    )

    # Random Forest
    tree.insert(
        "",
        "end",
        values=(
            "Random Forest",
            f"{rf['accuracy']:.4f}",
            f"{rf['precision']:.4f}",
            f"{rf['recall']:.4f}",
            f"{rf['f1_score']:.4f}"
        )
    )

    # DNN
    tree.insert(
        "",
        "end",
        values=(
            "DNN",
            f"{dnn['accuracy']:.4f}",
            f"{dnn['precision']:.4f}",
            f"{dnn['recall']:.4f}",
            f"{dnn['f1_score']:.4f}"
        )
    )

    tree.pack(fill="both", expand=True, padx=15, pady=15)