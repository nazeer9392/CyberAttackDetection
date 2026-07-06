

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

# Import project modules
from src.preprocessing import load_dataset, preprocess_dataset
from src.autoencoder_model import train_autoencoder
from src.decision_tree import train_decision_tree
from src.dnn_model import train_dnn
from src.predict import predict_attack
from src.graphs import show_comparison_graph
from src.table import show_table
from src.graphs import show_comparison_graph
from src.random_forest import train_random_forest
from src.dataset_graph import show_dataset_graph
from src.save_results import save_results

class CyberAttackGUI:

    def __init__(self):

        self.root = tk.Tk()

        self.root.title(
            "Toward Detection and Attribution of Cyber-Attacks in IoT-enabled Cyber-physical Systems"
        )

        self.root.geometry("1200x700")

        self.root.configure(bg="#F5F5F5")

        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.encoder = None
        self.pca = None
        self.dt_model = None
        self.dnn_model = None

        self.create_widgets()

    def create_widgets(self):

        title = tk.Label(
            self.root,
            text="Cyber Attack Detection using AutoEncoder + Decision Tree + DNN",
            font=("Arial", 18, "bold"),
            bg="#F5F5F5",
            fg="navy"
        )

        title.pack(pady=10)

        frame = tk.Frame(self.root, bg="#F5F5F5")
        frame.pack()

        # Buttons

        tk.Button(
            frame,
            text="Upload Dataset",
            width=25,
            command=self.upload_dataset
        ).grid(row=0, column=0, padx=5, pady=5)

        tk.Button(
            frame,
            text="Preprocess Dataset",
            width=25,
            command=self.preprocess
        ).grid(row=0, column=1, padx=5, pady=5)

        tk.Button(
            frame,
            text="Run AutoEncoder",
            width=25,
            command=self.run_autoencoder
        ).grid(row=1, column=0, padx=5, pady=5)

        tk.Button(
            frame,
            text="Run Decision Tree + PCA",
            width=25,
            command=self.run_decision_tree
        ).grid(row=1, column=1, padx=5, pady=5)

        tk.Button(
            frame,
            text="Run Random Forest",
            width=25,
            command=self.run_random_forest
        ).grid(row=2, column=0, padx=5, pady=5)

        tk.Button(
            frame,
            text="Run DNN",
            width=25,
            command=self.run_dnn
        ).grid(row=2, column=1, padx=5, pady=5)

        tk.Button(
            frame,
            text="Detect Attack",
            width=25,
            command=self.detect_attack
        ).grid(row=3, column=0, padx=5, pady=5)

        tk.Button(
            frame,
            text="Comparison Graph",
            width=25,
            command=self.graph
        ).grid(row=3, column=1, padx=5, pady=5)

        tk.Button(
            frame,
            text="Comparison Table",
            width=25,
            command=self.table
        ).grid(row=4, column=0, padx=5, pady=5)
        tk.Button(
            frame,
            text="Save Results",
            width=25,
            command=self.save_results
        ).grid(row=4, column=1, padx=5, pady=5)

        self.output = scrolledtext.ScrolledText(
            self.root,
            width=140,
            height=25
        )

        self.output.pack(pady=20)

    def upload_dataset(self):

        filename = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=[("CSV Files", "*.csv")]
        )

        if filename:

            self.dataset = load_dataset(filename)
            show_dataset_graph(self.dataset)

            self.output.insert(
                tk.END,
                f"Dataset Loaded Successfully\n{filename}\n\n"
            )

            self.output.insert(
                tk.END,
                str(self.dataset.head()) + "\n\n"
            )

    def preprocess(self):

        if self.dataset is None:

            messagebox.showerror(
                "Error",
                "Upload Dataset First"
            )

            return

        self.X_train, self.X_test, self.y_train, self.y_test = preprocess_dataset(
            self.dataset
        )

        self.output.insert(
            tk.END,
            "Dataset Preprocessed Successfully\n\n"
        )

    def run_autoencoder(self):
        if self.X_train is None:
            messagebox.showerror("Error", "Please preprocess the dataset first.")
            return

        self.output.insert(tk.END, "\nTraining AutoEncoder...\n")

        self.encoded_train, self.encoder = train_autoencoder(self.X_train)
        self.encoded_test = self.encoder.predict(self.X_test)

        self.output.insert(
            tk.END,
            f"AutoEncoder completed.\nEncoded Train Shape: {self.encoded_train.shape}\n\n"
        )

        # Code will be connected in next step

    def run_decision_tree(self):
        if not hasattr(self, "encoded_train"):
            messagebox.showerror("Error", "Run AutoEncoder first.")
            return
        self.dt_model, self.pca, metrics = train_decision_tree(
            self.encoded_train,
            self.encoded_test,
            self.y_train,
            self.y_test
        )

        self.dt_metrics = metrics

        self.output.insert(
            tk.END,
            f"\nDecision Tree Accuracy : {metrics['accuracy']:.4f}\n"
            )
    def run_random_forest(self):
        if self.pca is None:
            messagebox.showerror(
                "Error",
                "Run Decision Tree first."
                )
            return

        self.rf_model, metrics = train_random_forest(
            self.pca,
            self.encoded_train,
            self.encoded_test,
            self.y_train,
            self.y_test
            )

        self.rf_metrics = metrics

        self.output.insert(
            tk.END,
            "\nRandom Forest Completed Successfully\n"
            )

        self.output.insert(
            tk.END,
            f"Accuracy : {metrics['accuracy']:.4f}\n"
            )

        self.output.insert(
            tk.END,
            f"Precision : {metrics['precision']:.4f}\n"
            )

        self.output.insert(
            tk.END,
            f"Recall : {metrics['recall']:.4f}\n"
            )

        self.output.insert(
            tk.END,
            f"F1 Score : {metrics['f1_score']:.4f}\n\n"
            )

    def run_dnn(self):
        if self.pca is None:
            messagebox.showerror("Error", "Run Decision Tree first.")
            return
        self.dnn_model, metrics = train_dnn(
            self.pca,
            self.encoded_train,
            self.encoded_test,
            self.y_train,
            self.y_test
        )

        self.dnn_metrics = metrics

        self.output.insert(
            tk.END,
            f"\nDNN Accuracy : {metrics['accuracy']:.4f}\n"
        )

    def detect_attack(self):
        if self.dnn_model is None:
            messagebox.showerror(
            "Error",
            "Please train DNN first."
            )

            return

        filename = filedialog.askopenfilename(
            title="Select Test Data",
            filetypes=[("CSV Files", "*.csv")]
            )

        if filename == "":
            return

        import joblib

        scaler = joblib.load("models/minmax.pkl")

        results = predict_attack(
            filename,
            scaler,
            self.encoder,
            self.pca,
            self.dnn_model
        )

        self.output.insert(
            tk.END,
            "\nAttack Prediction\n\n"
        )

        for r in results:
            self.output.insert(
                tk.END,
                r + "\n"
            )
    def graph(self):
        if (
            not hasattr(self, "dt_metrics")
            or not hasattr(self, "rf_metrics")
            or not hasattr(self, "dnn_metrics")
            ):
            messagebox.showerror(
                "Error",
                "Run Decision Tree, Random Forest and DNN first."
                )
            return

        show_comparison_graph(
            self.dt_metrics,
            self.rf_metrics,
            self.dnn_metrics
            )

    def table(self):
        if (
            not hasattr(self, "dt_metrics")
            or not hasattr(self, "rf_metrics")
            or not hasattr(self, "dnn_metrics")
            ):
            messagebox.showerror(
                "Error",
                "Run Decision Tree, Random Forest and DNN first."
                )
            return

        show_table(
            self.root,
            self.dt_metrics,
            self.rf_metrics,
            self.dnn_metrics
            )
        
    def save_results(self):
        if (
            not hasattr(self, "dt_metrics")
            or not hasattr(self, "rf_metrics")
            or not hasattr(self, "dnn_metrics")
            ):
            messagebox.showerror(
                "Error",
                "Please run all algorithms first."
                )
            return

        save_results(
            self.dt_metrics,
            self.rf_metrics,
            self.dnn_metrics
            )

    def run(self):

        self.root.mainloop()