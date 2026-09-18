# ============================================================
# Genetic Disorder Risk Prediction
# Data Preprocessing for Federated Learning
#
# Clients:
#   Client 1
#   Client 2
#
# Dataset:
#   30,000 total records
#   15,000 records per client
#
# Targets:
#   Disorder
#   Risk_Level
#   Stage
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
PREPROCESSOR_DIR = os.path.join(DATA_DIR, "preprocessors")

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# Input features
# ============================================================

NUMERICAL_FEATURES = [
    "Age",
    "Hemoglobin",
    "RBC",
    "WBC",
    "Platelets",
    "PCV",
    "MCV",
    "MCH",
    "MCHC",
    "RDW",
    "Neutrophils",
    "Lymphocytes",
    "Monocytes",
    "Eosinophils",
    "Basophils",
    "Gene_Count",
    "Variant_Count",
    "Variant_Length",
    "Allele_Frequency",
    "Pathogenicity_Score",
    "Conservation_Score",
    "Mutation_Count",
    "Heterozygosity"
]

CATEGORICAL_FEATURES = [
    "Sex",
    "Variant_Type",
    "Origin",
    "Chromosome"
]


# ============================================================
# Target
# ============================================================

TARGET_COLUMN = "Disorder"

TARGET_CODE_COLUMN = "Disorder_Code"


# ============================================================
# Expected disorder mapping
# ============================================================

DISORDER_MAPPING = {
    "No Significant Genetic Risk": 0,
    "Sickle Cell Disease": 1,
    "Beta Thalassemia": 2,
    "Cystic Fibrosis": 3,
    "Hemophilia A": 4,
    "Huntington Disease": 5,
    "Phenylketonuria": 6,
    "Fragile X Syndrome": 7,
    "Duchenne Muscular Dystrophy": 8,
    "Marfan Syndrome": 9,
    "Tay-Sachs Disease": 10,
    "Spinal Muscular Atrophy": 11,
    "Wilson Disease": 12,
    "Gaucher Disease": 13,
    "Albinism": 14,
    "Familial Hypercholesterolemia": 15
}


# ============================================================
# Create directories
# ============================================================

def create_directories():

    os.makedirs(
        PROCESSED_DIR,
        exist_ok=True
    )

    os.makedirs(
        PREPROCESSOR_DIR,
        exist_ok=True
    )


# ============================================================
# Load client dataset
# ============================================================

def load_client_data(client_id):

    file_path = os.path.join(
        DATA_DIR,
        f"client_{client_id}.csv"
    )

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"Client dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    print(
        f"Client {client_id} dataset loaded: "
        f"{df.shape}"
    )

    return df


# ============================================================
# Validate dataset
# ============================================================

def validate_dataset(df, client_id):

    print(
        f"\nValidating Client {client_id}..."
    )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    required_columns = (
        NUMERICAL_FEATURES
        + CATEGORICAL_FEATURES
        + [
            "Patient_ID",
            "Disorder",
            "Disorder_Code",
            "Risk_Level",
            "Risk_Code",
            "Stage",
            "Stage_Code"
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Client {client_id} is missing columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Check missing values
    # --------------------------------------------------------

    missing_values = df.isnull().sum().sum()

    print(
        f"Missing values: {missing_values}"
    )

    if missing_values > 0:

        raise ValueError(
            f"Client {client_id} contains missing values."
        )

    # --------------------------------------------------------
    # Check disorder labels
    # --------------------------------------------------------

    unknown_disorders = set(
        df["Disorder"].unique()
    ) - set(
        DISORDER_MAPPING.keys()
    )

    if unknown_disorders:

        raise ValueError(
            f"Unknown disorders found: "
            f"{unknown_disorders}"
        )

    # --------------------------------------------------------
    # Check disorder codes
    # --------------------------------------------------------

    expected_codes = df["Disorder"].map(
        DISORDER_MAPPING
    )

    if not np.array_equal(
        expected_codes.values,
        df["Disorder_Code"].values
    ):

        raise ValueError(
            f"Disorder codes do not match "
            f"disorder names in Client {client_id}."
        )

    print(
        f"Unique disorders: "
        f"{df['Disorder'].nunique()}"
    )

    print(
        f"Validation successful for Client {client_id}."
    )


# ============================================================
# Build preprocessing pipeline
# ============================================================

def build_preprocessor():

    numerical_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES
            )
        ],
        remainder="drop"
    )

    return preprocessor


# ============================================================
# Process one client
# ============================================================

def process_client(
    df,
    client_id,
    preprocessor
):

    print("\n" + "=" * 60)

    print(
        f"PROCESSING CLIENT {client_id}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X = df[
        NUMERICAL_FEATURES
        + CATEGORICAL_FEATURES
    ].copy()

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y = df[
        TARGET_CODE_COLUMN
    ].astype(int).copy()

    # --------------------------------------------------------
    # Stratified train/test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples : {len(X_test)}"
    )

    # --------------------------------------------------------
    # Fit preprocessor ONLY on training data
    # --------------------------------------------------------

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    X_train_processed = np.asarray(
        X_train_processed,
        dtype=np.float32
    )

    X_test_processed = np.asarray(
        X_test_processed,
        dtype=np.float32
    )

    y_train = np.asarray(
        y_train,
        dtype=np.int64
    )

    y_test = np.asarray(
        y_test,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Verify shapes
    # --------------------------------------------------------

    print(
        f"Processed training shape: "
        f"{X_train_processed.shape}"
    )

    print(
        f"Processed testing shape : "
        f"{X_test_processed.shape}"
    )

    # --------------------------------------------------------
    # Save NumPy arrays
    # --------------------------------------------------------

    np.save(
        os.path.join(
            PROCESSED_DIR,
            f"client_{client_id}_X_train.npy"
        ),
        X_train_processed
    )

    np.save(
        os.path.join(
            PROCESSED_DIR,
            f"client_{client_id}_X_test.npy"
        ),
        X_test_processed
    )

    np.save(
        os.path.join(
            PROCESSED_DIR,
            f"client_{client_id}_y_train.npy"
        ),
        y_train
    )

    np.save(
        os.path.join(
            PROCESSED_DIR,
            f"client_{client_id}_y_test.npy"
        ),
        y_test
    )

    # --------------------------------------------------------
    # Save preprocessor
    # --------------------------------------------------------

    preprocessor_path = os.path.join(
        PREPROCESSOR_DIR,
        f"client_{client_id}_preprocessor.pkl"
    )

    joblib.dump(
        preprocessor,
        preprocessor_path
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {
        "client_id": client_id,
        "original_samples": int(len(df)),
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "input_features_before_encoding": len(
            NUMERICAL_FEATURES
            + CATEGORICAL_FEATURES
        ),
        "processed_features": int(
            X_train_processed.shape[1]
        ),
        "number_of_classes": len(
            DISORDER_MAPPING
        ),
        "target": TARGET_COLUMN,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE
    }

    metadata_path = os.path.join(
        PROCESSED_DIR,
        f"client_{client_id}_metadata.json"
    )

    with open(
        metadata_path,
        "w"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Print class distribution
    # --------------------------------------------------------

    print(
        "\nTraining class distribution:"
    )

    unique, counts = np.unique(
        y_train,
        return_counts=True
    )

    for class_code, count in zip(
        unique,
        counts
    ):

        disorder_name = next(
            (
                name
                for name, code
                in DISORDER_MAPPING.items()
                if code == int(class_code)
            ),
            "Unknown"
        )

        print(
            f"{class_code:2d} | "
            f"{disorder_name:<35} | "
            f"{count}"
        )

    print(
        f"\nSaved Client {client_id} processed data."
    )

    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_test": y_test
    }


# ============================================================
# Save combined information
# ============================================================

def save_global_metadata(
    client1_data,
    client2_data
):

    global_metadata = {

        "total_records": 30000,

        "number_of_clients": 2,

        "records_per_client": {
            "client_1": 15000,
            "client_2": 15000
        },

        "train_test_ratio": "80/20",

        "client_1_training": int(
            len(client1_data["X_train"])
        ),

        "client_1_testing": int(
            len(client1_data["X_test"])
        ),

        "client_2_training": int(
            len(client2_data["X_train"])
        ),

        "client_2_testing": int(
            len(client2_data["X_test"])
        ),

        "input_features": (
            NUMERICAL_FEATURES
            + CATEGORICAL_FEATURES
        ),

        "number_of_input_features": 27,

        "number_of_disorder_classes": 16,

        "target": TARGET_COLUMN,

        "federated_clients": [
            "Client 1",
            "Client 2"
        ],

        "federated_algorithm": "FedAvg"
    }

    metadata_path = os.path.join(
        PROCESSED_DIR,
        "global_metadata.json"
    )

    with open(
        metadata_path,
        "w"
    ) as file:

        json.dump(
            global_metadata,
            file,
            indent=4
        )

    print(
        f"\nGlobal metadata saved: "
        f"{metadata_path}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("GENETIC DISORDER DATA PREPROCESSING")
    print("=" * 60)

    print(
        "\nConfiguration:"
    )

    print(
        "Total records : 30,000"
    )

    print(
        "Clients       : 2"
    )

    print(
        "Features      : 27"
    )

    print(
        "Classes       : 16"
    )

    print(
        "Train/Test    : 80/20"
    )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    create_directories()

    # --------------------------------------------------------
    # Load clients
    # --------------------------------------------------------

    client1_df = load_client_data(1)

    client2_df = load_client_data(2)

    # --------------------------------------------------------
    # Validate clients
    # --------------------------------------------------------

    validate_dataset(
        client1_df,
        1
    )

    validate_dataset(
        client2_df,
        2
    )

    # --------------------------------------------------------
    # Build separate preprocessors
    #
    # Each client keeps its own preprocessing object.
    # This avoids sharing raw client information.
    # --------------------------------------------------------

    client1_preprocessor = build_preprocessor()

    client2_preprocessor = build_preprocessor()

    # --------------------------------------------------------
    # Process Client 1
    # --------------------------------------------------------

    client1_data = process_client(
        client1_df,
        1,
        client1_preprocessor
    )

    # --------------------------------------------------------
    # Process Client 2
    # --------------------------------------------------------

    client2_data = process_client(
        client2_df,
        2,
        client2_preprocessor
    )

    # --------------------------------------------------------
    # Save global metadata
    # --------------------------------------------------------

    save_global_metadata(
        client1_data,
        client2_data
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL PREPROCESSING VERIFICATION")
    print("=" * 60)

    print(
        "\nClient 1:"
    )

    print(
        "X_train:",
        client1_data["X_train"].shape
    )

    print(
        "X_test :",
        client1_data["X_test"].shape
    )

    print(
        "y_train:",
        client1_data["y_train"].shape
    )

    print(
        "y_test :",
        client1_data["y_test"].shape
    )

    print(
        "\nClient 2:"
    )

    print(
        "X_train:",
        client2_data["X_train"].shape
    )

    print(
        "X_test :",
        client2_data["X_test"].shape
    )

    print(
        "y_train:",
        client2_data["y_train"].shape
    )

    print(
        "y_test :",
        client2_data["y_test"].shape
    )

    # --------------------------------------------------------
    # Check NaN
    # --------------------------------------------------------

    print("\nNaN verification:")

    print(
        "Client 1 X_train:",
        np.isnan(
            client1_data["X_train"]
        ).sum()
    )

    print(
        "Client 1 X_test :",
        np.isnan(
            client1_data["X_test"]
        ).sum()
    )

    print(
        "Client 2 X_train:",
        np.isnan(
            client2_data["X_train"]
        ).sum()
    )

    print(
        "Client 2 X_test :",
        np.isnan(
            client2_data["X_test"]
        ).sum()
    )

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        "\nProcessed files are available in:"
    )

    print(
        PROCESSED_DIR
    )

    print(
        "\nPreprocessors are available in:"
    )

    print(
        PREPROCESSOR_DIR
    )


# ============================================================
# Execute
# ============================================================

if __name__ == "__main__":
    main()