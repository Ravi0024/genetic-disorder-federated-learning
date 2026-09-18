# ============================================================
# model.py
# Genetic Disorder Risk Prediction
# Neural Network Model for Federated Learning
# ============================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.optimizers import Adam


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

INPUT_DIM = 56
NUM_CLASSES = 16

LEARNING_RATE = 0.001


# ============================================================
# CREATE MODEL
# ============================================================

def create_model(
    input_dim=INPUT_DIM,
    num_classes=NUM_CLASSES
):
    """
    Creates the neural network used by both FL clients.

    Input:
        56 processed features

    Output:
        16 genetic disorder classes
    """

    model = Sequential([
        Input(shape=(input_dim,)),

        # First hidden layer
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.30),

        # Second hidden layer
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.20),

        # Third hidden layer
        Dense(32, activation="relu"),

        # Output layer
        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# GET MODEL WEIGHTS
# ============================================================

def get_model_weights(model):
    """
    Returns model weights.

    Used by Federated Learning server
    for FedAvg aggregation.
    """

    return model.get_weights()


# ============================================================
# SET MODEL WEIGHTS
# ============================================================

def set_model_weights(model, weights):
    """
    Updates model with weights received
    from the federated server.
    """

    model.set_weights(weights)


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model, path):
    """
    Saves trained Keras model.
    """

    model.save(path)


# ============================================================
# LOAD MODEL
# ============================================================

def load_trained_model(path):
    """
    Loads a previously saved Keras model.
    """

    return tf.keras.models.load_model(path)


# ============================================================
# MODEL SUMMARY
# ============================================================

def print_model_summary(model):
    """
    Displays model architecture.
    """

    model.summary()


# ============================================================
# TEST MODEL
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("GENETIC DISORDER PREDICTION MODEL")
    print("=" * 60)

    print(f"Input features : {INPUT_DIM}")
    print(f"Output classes : {NUM_CLASSES}")
    print()

    # Create model
    model = create_model()

    # Display architecture
    model.summary()

    # Test with dummy data
    dummy_data = np.random.random((5, INPUT_DIM))

    predictions = model.predict(dummy_data, verbose=0)

    print()
    print("Model test successful!")
    print("Dummy input shape :", dummy_data.shape)
    print("Prediction shape  :", predictions.shape)
    print("Expected output   :", (5, NUM_CLASSES))

    # Check probability sum
    print()
    print("Probability sums:")

    for i, prediction in enumerate(predictions):
        print(
            f"Sample {i + 1}: "
            f"{np.sum(prediction):.4f}"
        )

    print()
    print("=" * 60)
    print("model.py is working correctly.")
    print("=" * 60)