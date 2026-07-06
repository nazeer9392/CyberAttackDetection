import pandas as pd
import numpy as np

ATTACK_LABELS = {
    0: "Normal",
    1: "NMRI",
    2: "CMRI",
    3: "MSCI",
    4: "MPCI",
    5: "MFCI",
    6: "DoS"
}


def predict_attack(test_file, scaler, encoder, pca, dnn_model):

    # Load test dataset
    df = pd.read_csv(test_file)

    # Remove columns not used during training
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    if "result" in df.columns:
        df = df.drop(columns=["result"])

    # Scale data
    X = scaler.transform(df)

    # AutoEncoder
    encoded = encoder.predict(X, verbose=0)

    # PCA
    encoded = pca.transform(encoded)

    # DNN Prediction
    prediction = dnn_model.predict(encoded, verbose=0)

    prediction = np.argmax(prediction, axis=1)

    results = []

    for i, p in enumerate(prediction):
        attack = ATTACK_LABELS.get(int(p), "Unknown")
        results.append(f"Record {i+1} ---> {attack}")

    return results