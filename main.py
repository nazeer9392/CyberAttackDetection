# from src.preprocessing import load_dataset, preprocess_dataset

# dataset_path = "dataset/swat.csv"

# df = load_dataset(dataset_path)

# X_train, X_test, y_train, y_test = preprocess_dataset(df)

# print("Everything is working!")


# from src.preprocessing import load_dataset, preprocess_dataset
# from src.autoencoder_model import train_autoencoder

# dataset = load_dataset("dataset/swat.csv")

# X_train, X_test, y_train, y_test = preprocess_dataset(dataset)

# encoded_features, encoder = train_autoencoder(X_train)

# print("\nEncoded Feature Shape:", encoded_features.shape)

import os

os.environ["TCL_LIBRARY"] = r"C:\Users\nazeer\AppData\Local\Programs\Python\Python313\tcl\tcl8.6"
os.environ["TK_LIBRARY"] = r"C:\Users\nazeer\AppData\Local\Programs\Python\Python313\tcl\tk8.6"

from gui import CyberAttackGUI

if __name__ == "__main__":
    app = CyberAttackGUI()
    app.run()