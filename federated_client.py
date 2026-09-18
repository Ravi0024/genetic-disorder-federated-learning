# ============================================================
# federated_client.py
# Genetic Disorder Risk Prediction
# Federated Learning Client
# ============================================================

import os
import sys
import numpy as np
import tensorflow as tf
import flwr as fl

from model import (
    create_model,
    get_model_weights,
    set_model_weights
)


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLIENTS = 2

INPUT_DIM = 56
NUM_CLASSES = 16

BATCH_SIZE = 32
LOCAL_EPOCHS = 5


# ============================================================
# SUPPRESS EXCESSIVE TENSORFLOW LOGS
# ============================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ============================================================
# CLIENT DATA LOADING
# ============================================================

def load_client_data(client_id):
    """
    Loads preprocessed training and testing data
    for the specified federated client.
    """

    base_path = "data/processed"

    X_train_path = os.path.join(
        base_path,
        f"client_{client_id}_X_train.npy"
    )

    X_test_path = os.path.join(
        base_path,
        f"client_{client_id}_X_test.npy"
    )

    y_train_path = os.path.join(
        base_path,
        f"client_{client_id}_y_train.npy"
    )

    y_test_path = os.path.join(
        base_path,
        f"client_{client_id}_y_test.npy"
    )

    # Check files
    required_files = [
        X_train_path,
        X_test_path,
        y_train_path,
        y_test_path
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    # Load arrays
    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)

    y_train = np.load(y_train_path)
    y_test = np.load(y_test_path)

    return X_train, X_test, y_train, y_test


# ============================================================
# FEDERATED CLIENT CLASS
# ============================================================

class GeneticDisorderClient(fl.client.NumPyClient):

    def __init__(self, client_id):

        self.client_id = client_id

        print()
        print("=" * 60)
        print(f"INITIALIZING FEDERATED CLIENT {client_id}")
        print("=" * 60)

        # ----------------------------------------------------
        # Load client data
        # ----------------------------------------------------

        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        ) = load_client_data(client_id)

        print(
            f"Client {client_id} training data: "
            f"{self.X_train.shape}"
        )

        print(
            f"Client {client_id} testing data: "
            f"{self.X_test.shape}"
        )

        print(
            f"Client {client_id} training labels: "
            f"{self.y_train.shape}"
        )

        print(
            f"Client {client_id} testing labels: "
            f"{self.y_test.shape}"
        )

        # ----------------------------------------------------
        # Create local model
        # ----------------------------------------------------

        self.model = create_model(
            input_dim=INPUT_DIM,
            num_classes=NUM_CLASSES
        )

        print()
        print(f"Client {client_id} local model created.")
        print("=" * 60)


    # ========================================================
    # GET PARAMETERS
    # ========================================================

    def get_parameters(self, config):

        return get_model_weights(self.model)


    # ========================================================
    # LOCAL TRAINING
    # ========================================================

    def fit(self, parameters, config):

        # Receive global model weights
        set_model_weights(
            self.model,
            parameters
        )

        print()
        print("-" * 60)
        print(f"CLIENT {self.client_id} - LOCAL TRAINING")
        print("-" * 60)

        print(
            f"Training samples : {len(self.X_train)}"
        )

        print(
            f"Features         : {self.X_train.shape[1]}"
        )

        print(
            f"Local epochs     : {LOCAL_EPOCHS}"
        )

        # Train local model
        history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.1,
            verbose=1
        )

        # Get final training accuracy
        final_accuracy = history.history["accuracy"][-1]

        print()
        print(
            f"Client {self.client_id} "
            f"training accuracy: "
            f"{final_accuracy:.4f}"
        )

        # Return updated weights
        return (
            get_model_weights(self.model),
            len(self.X_train),
            {
                "accuracy": float(final_accuracy)
            }
        )


    # ========================================================
    # LOCAL EVALUATION
    # ========================================================

    def evaluate(self, parameters, config):

        # Receive global model weights
        set_model_weights(
            self.model,
            parameters
        )

        print()
        print("-" * 60)
        print(f"CLIENT {self.client_id} - LOCAL EVALUATION")
        print("-" * 60)

        loss, accuracy = self.model.evaluate(
            self.X_test,
            self.y_test,
            verbose=0
        )

        print(
            f"Client {self.client_id} "
            f"test loss: {loss:.4f}"
        )

        print(
            f"Client {self.client_id} "
            f"test accuracy: {accuracy:.4f}"
        )

        return (
            float(loss),
            len(self.X_test),
            {
                "accuracy": float(accuracy)
            }
        )


# ============================================================
# START CLIENT
# ============================================================

def start_client(client_id):

    if client_id not in [1, 2]:
        raise ValueError(
            "client_id must be either 1 or 2."
        )

    client = GeneticDisorderClient(client_id)

    print()
    print("=" * 60)
    print(f"STARTING FEDERATED CLIENT {client_id}")
    print("=" * 60)

    # Connect to Flower server
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client()
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("GENETIC DISORDER FEDERATED LEARNING CLIENT")
    print("=" * 60)

    # --------------------------------------------------------
    # Client ID must be supplied
    # --------------------------------------------------------

    if len(sys.argv) != 2:

        print()
        print("Usage:")
        print("python federated_client.py 1")
        print("python federated_client.py 2")
        print()

        sys.exit(1)

    try:
        client_id = int(sys.argv[1])
    except ValueError:

        print("ERROR: Client ID must be 1 or 2.")
        sys.exit(1)

    start_client(client_id)