# ============================================================
# evaluate.py
# Genetic Disorder Risk Prediction
# Global Model Evaluation
# ============================================================

import os
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/global_model.keras"

PROCESSED_PATH = "data/processed"

NUM_CLASSES = 16


# ============================================================
# DISORDER CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "No Significant Genetic Risk",
    "Sickle Cell Disease",
    "Beta Thalassemia",
    "Cystic Fibrosis",
    "Hemophilia A",
    "Huntington Disease",
    "Phenylketonuria",
    "Fragile X Syndrome",
    "Duchenne Muscular Dystrophy",
    "Marfan Syndrome",
    "Tay-Sachs Disease",
    "Spinal Muscular Atrophy",
    "Wilson Disease",
    "Gaucher Disease",
    "Albinism",
    "Familial Hypercholesterolemia"
]


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    print()
    print("=" * 70)
    print("LOADING TEST DATA")
    print("=" * 70)

    # Client 1
    X_test_1 = np.load(
        os.path.join(
            PROCESSED_PATH,
            "client_1_X_test.npy"
        )
    )

    y_test_1 = np.load(
        os.path.join(
            PROCESSED_PATH,
            "client_1_y_test.npy"
        )
    )

    # Client 2
    X_test_2 = np.load(
        os.path.join(
            PROCESSED_PATH,
            "client_2_X_test.npy"
        )
    )

    y_test_2 = np.load(
        os.path.join(
            PROCESSED_PATH,
            "client_2_y_test.npy"
        )
    )

    print(
        f"Client 1 test data: "
        f"{X_test_1.shape}"
    )

    print(
        f"Client 2 test data: "
        f"{X_test_2.shape}"
    )

    # Combine both client test datasets
    X_test = np.concatenate(
        [X_test_1, X_test_2],
        axis=0
    )

    y_test = np.concatenate(
        [y_test_1, y_test_2],
        axis=0
    )

    print()
    print(
        f"Combined test data: "
        f"{X_test.shape}"
    )

    print(
        f"Combined test labels: "
        f"{y_test.shape}"
    )

    return X_test, y_test


# ============================================================
# LOAD GLOBAL MODEL
# ============================================================

def load_global_model():

    print()
    print("=" * 70)
    print("LOADING FINAL GLOBAL MODEL")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Global model not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        f"Global model loaded from:"
    )

    print(
        f"  {MODEL_PATH}"
    )

    return model


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):

    print()
    print("=" * 70)
    print("GLOBAL MODEL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Keras evaluation
    # --------------------------------------------------------

    loss, keras_accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    probabilities = model.predict(
        X_test,
        verbose=0
    )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # Display overall metrics
    # --------------------------------------------------------

    print()
    print("OVERALL PERFORMANCE")
    print("-" * 70)

    print(
        f"Test Loss              : {loss:.4f}"
    )

    print(
        f"Keras Accuracy         : {keras_accuracy:.4f}"
    )

    print(
        f"Accuracy               : {accuracy:.4f}"
    )

    print(
        f"Precision (Weighted)   : {precision:.4f}"
    )

    print(
        f"Recall (Weighted)      : {recall:.4f}"
    )

    print(
        f"F1 Score (Weighted)    : {f1:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0
    )

    print(report)

    return (
        y_pred,
        probabilities,
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def create_confusion_matrix(y_test, y_pred):

    print()
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=list(range(NUM_CLASSES))
    )

    print(cm)

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        "results",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Plot confusion matrix
    # --------------------------------------------------------

    plt.figure(
        figsize=(16, 13)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.title(
        "Global Model - Confusion Matrix"
    )

    plt.xlabel(
        "Predicted Disorder"
    )

    plt.ylabel(
        "Actual Disorder"
    )

    plt.xticks(
        rotation=90
    )

    plt.yticks(
        rotation=0
    )

    plt.tight_layout()

    confusion_path = (
        "results/confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print()
    print(
        f"Confusion matrix saved to:"
    )

    print(
        f"  {confusion_path}"
    )


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    y_test,
    y_pred,
    probabilities
):

    print()
    print("=" * 70)
    print("SAVING PREDICTIONS")
    print("=" * 70)

    os.makedirs(
        "results",
        exist_ok=True
    )

    prediction_path = (
        "results/predictions.csv"
    )

    # Create CSV data
    import pandas as pd

    prediction_data = {

        "Actual_Code": y_test,

        "Actual_Disorder": [
            CLASS_NAMES[i]
            for i in y_test
        ],

        "Predicted_Code": y_pred,

        "Predicted_Disorder": [
            CLASS_NAMES[i]
            for i in y_pred
        ],

        "Prediction_Confidence": np.max(
            probabilities,
            axis=1
        )
    }

    df = pd.DataFrame(
        prediction_data
    )

    df.to_csv(
        prediction_path,
        index=False
    )

    print(
        f"Predictions saved to:"
    )

    print(
        f"  {prediction_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("GENETIC DISORDER RISK PREDICTION")
    print("FEDERATED GLOBAL MODEL EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    X_test, y_test = load_test_data()

    # --------------------------------------------------------
    # Load global model
    # --------------------------------------------------------

    model = load_global_model()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    (
        y_pred,
        probabilities,
        accuracy,
        precision,
        recall,
        f1
    ) = evaluate_model(
        model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    create_confusion_matrix(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    save_predictions(
        y_test,
        y_pred,
        probabilities
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)

    print()
    print(
        f"Accuracy            : {accuracy * 100:.2f}%"
    )

    print(
        f"Weighted Precision  : {precision * 100:.2f}%"
    )

    print(
        f"Weighted Recall     : {recall * 100:.2f}%"
    )

    print(
        f"Weighted F1 Score   : {f1 * 100:.2f}%"
    )

    print()
    print("Generated files:")
    print(
        "  results/confusion_matrix.png"
    )
    print(
        "  results/predictions.csv"
    )

    print()
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()