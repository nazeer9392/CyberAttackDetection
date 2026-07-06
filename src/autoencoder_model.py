import os
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json


def train_autoencoder(X_train):
    """
    Train AutoEncoder and extract encoded features.
    """

    input_dim = X_train.shape[1]

    encoding_dim = 6

    input_layer = Input(shape=(input_dim,))

    encoder = Dense(encoding_dim, activation="relu")(input_layer)

    decoder = Dense(input_dim, activation="sigmoid")(encoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)

    encoder_model = Model(inputs=input_layer, outputs=encoder)

    autoencoder.compile(
        optimizer="adam",
        loss="mse"
    )

    autoencoder.fit(
        X_train,
        X_train,
        epochs=30,
        batch_size=32,
        shuffle=True,
        verbose=1
    )

    # Create models folder if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Save model architecture
    with open("models/encoder_model.json", "w") as f:
        f.write(autoencoder.to_json())

    # Save weights
    autoencoder.save_weights("models/encoder_weights.weights.h5")

    print("\nAutoEncoder Model Saved")

    encoded_features = encoder_model.predict(X_train)

    return encoded_features, encoder_model


def load_autoencoder():
    """
    Load saved AutoEncoder.
    """

    with open("models/encoder_model.json", "r") as f:
        model_json = f.read()

    autoencoder = model_from_json(model_json)

    autoencoder.load_weights("models/encoder_weights.weights.h5")

    print("AutoEncoder Loaded Successfully")

    return autoencoder