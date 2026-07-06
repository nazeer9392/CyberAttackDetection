import os
import joblib
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def train_dnn(pca, encoded_train, encoded_test, y_train, y_test):

    print("\n==============================")
    print("Training Deep Neural Network")
    print("==============================")

    # Apply PCA
    X_train = pca.transform(encoded_train)
    X_test = pca.transform(encoded_test)

    # Convert labels to one-hot encoding
    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    num_classes = y_train_cat.shape[1]

    model = Sequential()

    model.add(Dense(64, activation="relu", input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.3))

    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X_train,
        y_train_cat,
        epochs=30,
        batch_size=16,
        verbose=1
    )

    # Prediction
    prediction = model.predict(X_test)

    prediction = np.argmax(prediction, axis=1)

    accuracy = accuracy_score(y_test, prediction)

    precision = precision_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        prediction,
        average="weighted",
        zero_division=0
    )

    print("\n========== DNN Results ==========")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    os.makedirs("models", exist_ok=True)

    model.save("models/dnn_model.keras")

    print("\nDNN Model Saved Successfully")

    return model, {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }