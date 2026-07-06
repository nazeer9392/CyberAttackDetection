import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_dataset(file_path):
    """
    Load SWAT dataset
    """

    df = pd.read_csv(file_path)

    print("Dataset Loaded Successfully")
    print(df.head())

    return df


def preprocess_dataset(df):
    """
    Preprocess dataset
    """

    # Replace missing values
    df.fillna(0, inplace=True)

    # Remove timestamp if present
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    # Target column
    y = df["result"]

    # Feature columns
    X = df.drop(columns=["result"])

    # Normalize
    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)

    # Save scaler
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        scaler,
        "models/minmax.pkl"
    )

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nPreprocessing Completed")
    print("Training Samples :", len(X_train))
    print("Testing Samples  :", len(X_test))

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )