# ============================================================
# federated_server.py
# Genetic Disorder Risk Prediction
# Federated Learning Server using FedAvg
# Flower 1.36.0
# ============================================================

import os
import flwr as fl
from model import create_model


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLIENTS = 2
NUM_ROUNDS = 5

INPUT_DIM = 56
NUM_CLASSES = 16

MODEL_DIR = "models"
GLOBAL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "global_model.keras"
)


# ============================================================
# METRIC AGGREGATION
# ============================================================

def weighted_average(metrics):

    if not metrics:
        return {}

    total_examples = sum(
        num_examples
        for num_examples, _ in metrics
    )

    if total_examples == 0:
        return {}

    weighted_accuracy = sum(
        num_examples * metric["accuracy"]
        for num_examples, metric in metrics
    ) / total_examples

    return {
        "accuracy": weighted_accuracy
    }


# ============================================================
# CUSTOM FEDAVG STRATEGY
# ============================================================

class SaveModelFedAvg(fl.server.strategy.FedAvg):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.final_parameters = None

    # --------------------------------------------------------
    # Capture aggregated parameters after each round
    # --------------------------------------------------------

    def aggregate_fit(
        self,
        server_round,
        results,
        failures
    ):

        aggregated_parameters, metrics = super().aggregate_fit(
            server_round,
            results,
            failures
        )

        if aggregated_parameters is not None:

            self.final_parameters = aggregated_parameters

            print()
            print(
                f"FedAvg aggregation completed "
                f"for round {server_round}."
            )

        return aggregated_parameters, metrics


# ============================================================
# CREATE GLOBAL MODEL
# ============================================================

def create_global_model():

    return create_model(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES
    )


# ============================================================
# START SERVER
# ============================================================

def start_server():

    print()
    print("=" * 70)
    print("GENETIC DISORDER FEDERATED LEARNING SERVER")
    print("=" * 70)

    print(f"Number of clients : {NUM_CLIENTS}")
    print(f"Federated rounds  : {NUM_ROUNDS}")
    print(f"Input features    : {INPUT_DIM}")
    print(f"Output classes    : {NUM_CLASSES}")
    print("Aggregation       : FedAvg")
    print(f"Flower version    : 1.36.0")
    print("=" * 70)

    # --------------------------------------------------------
    # Create model directory
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create initial global model
    # --------------------------------------------------------

    global_model = create_global_model()

    print()
    print("Initial global model created.")

    # --------------------------------------------------------
    # Create FedAvg strategy
    # --------------------------------------------------------

    strategy = SaveModelFedAvg(

        fraction_fit=1.0,
        fraction_evaluate=1.0,

        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,

        on_fit_config_fn=lambda server_round: {
            "server_round": server_round,
            "local_epochs": 5,
            "batch_size": 32
        },

        on_evaluate_config_fn=lambda server_round: {
            "server_round": server_round
        },

        fit_metrics_aggregation_fn=weighted_average,

        evaluate_metrics_aggregation_fn=weighted_average,

        initial_parameters=fl.common.ndarrays_to_parameters(
            global_model.get_weights()
        )
    )

    # --------------------------------------------------------
    # Start Flower server
    # --------------------------------------------------------

    print()
    print("Starting Flower server...")
    print("Server address: 127.0.0.1:8080")
    print()

    history = fl.server.start_server(

        server_address="127.0.0.1:8080",

        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS
        ),

        strategy=strategy
    )

    # --------------------------------------------------------
    # SAVE FINAL GLOBAL MODEL
    # --------------------------------------------------------

    if strategy.final_parameters is not None:

        print()
        print("=" * 70)
        print("SAVING FINAL GLOBAL MODEL")
        print("=" * 70)

        final_weights = fl.common.parameters_to_ndarrays(
            strategy.final_parameters
        )

        global_model.set_weights(
            final_weights
        )

        global_model.save(
            GLOBAL_MODEL_PATH
        )

        print()
        print(
            f"Final global model saved to:"
        )
        print(
            f"  {GLOBAL_MODEL_PATH}"
        )

        print()
        print(
            f"Final global model size:"
            f" {len(final_weights)} weight arrays"
        )

    else:

        print()
        print(
            "ERROR: Final global parameters were not received."
        )

    # --------------------------------------------------------
    # DISPLAY TRAINING HISTORY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FEDERATED LEARNING COMPLETED")
    print("=" * 70)

    print()
    print("Federated rounds completed:", NUM_ROUNDS)

    if hasattr(history, "metrics_distributed"):

        print()
        print("Distributed evaluation metrics:")

        print(
            history.metrics_distributed
        )

    print()
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    start_server()