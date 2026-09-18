# ============================================================
# Genetic Disorder Risk Prediction
# Synthetic Dataset Generator for Federated Learning
#
# Dataset:
#   Total records : 30,000
#   Clients       : 2
#   Records/client: 15,000
#   Features      : 27
#   Disorders     : 15 + No Significant Genetic Risk
# ============================================================

import os
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

SEED = 42
TOTAL_RECORDS = 30000
NUM_CLIENTS = 2
RECORDS_PER_CLIENT = TOTAL_RECORDS // NUM_CLIENTS

OUTPUT_DIR = "data"

np.random.seed(SEED)

# ------------------------------------------------------------
# Genetic disorder classes
# ------------------------------------------------------------

DISORDERS = [
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

DISORDER_CODES = {
    disorder: i for i, disorder in enumerate(DISORDERS)
}

# ------------------------------------------------------------
# Feature definitions
# ------------------------------------------------------------

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

TARGET_FEATURES = [
    "Disorder",
    "Disorder_Code",
    "Risk_Level",
    "Risk_Code",
    "Stage",
    "Stage_Code"
]

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def clipped_normal(mean, std, low, high, size):
    """
    Generate normally distributed values within a range.
    """
    values = np.random.normal(mean, std, size)
    return np.clip(values, low, high)


def generate_variant_type(n):
    """
    Generate DNA variant types.
    """
    return np.random.choice(
        ["SNV", "Insertion", "Deletion", "Duplication"],
        size=n,
        p=[0.65, 0.12, 0.15, 0.08]
    )


def generate_origin(n):
    """
    Generate variant origin.
    """
    return np.random.choice(
        ["Germline", "Inherited", "De_Novo"],
        size=n,
        p=[0.45, 0.45, 0.10]
    )


def generate_chromosome(n):
    """
    Generate chromosome information.
    """
    chromosomes = [str(i) for i in range(1, 23)] + ["X", "Y"]

    return np.random.choice(
        chromosomes,
        size=n
    )


# ------------------------------------------------------------
# Generate disease-specific biological patterns
# ------------------------------------------------------------

def apply_disorder_pattern(df, disorder, indices):
    """
    Modify synthetic blood/genetic features according to
    disorder-specific patterns.

    NOTE:
    These are synthetic research patterns and are NOT
    clinically validated measurements.
    """

    n = len(indices)

    if n == 0:
        return

    # --------------------------------------------------------
    # Sickle Cell Disease
    # --------------------------------------------------------

    if disorder == "Sickle Cell Disease":

        df.loc[indices, "Hemoglobin"] = clipped_normal(
            9.2, 1.1, 6.5, 12.0, n
        )

        df.loc[indices, "RBC"] = clipped_normal(
            3.9, 0.5, 2.5, 5.2, n
        )

        df.loc[indices, "MCV"] = clipped_normal(
            82, 8, 65, 105, n
        )

        df.loc[indices, "RDW"] = clipped_normal(
            18, 2.5, 12, 25, n
        )

        df.loc[indices, "Mutation_Count"] = np.random.randint(
            1, 5, n
        )

    # --------------------------------------------------------
    # Beta Thalassemia
    # --------------------------------------------------------

    elif disorder == "Beta Thalassemia":

        df.loc[indices, "Hemoglobin"] = clipped_normal(
            8.8, 1.0, 6.0, 11.5, n
        )

        df.loc[indices, "RBC"] = clipped_normal(
            5.1, 0.6, 3.5, 6.8, n
        )

        df.loc[indices, "MCV"] = clipped_normal(
            67, 7, 50, 85, n
        )

        df.loc[indices, "MCH"] = clipped_normal(
            21, 3, 15, 28, n
        )

        df.loc[indices, "RDW"] = clipped_normal(
            19, 3, 13, 27, n
        )

    # --------------------------------------------------------
    # Cystic Fibrosis
    # --------------------------------------------------------

    elif disorder == "Cystic Fibrosis":

        df.loc[indices, "WBC"] = clipped_normal(
            11.0, 2.5, 4.0, 20.0, n
        )

        df.loc[indices, "Neutrophils"] = clipped_normal(
            72, 7, 50, 90, n
        )

        df.loc[indices, "Lymphocytes"] = clipped_normal(
            20, 5, 8, 40, n
        )

        df.loc[indices, "Mutation_Count"] = np.random.randint(
            1, 6, n
        )

    # --------------------------------------------------------
    # Hemophilia A
    # --------------------------------------------------------

    elif disorder == "Hemophilia A":

        df.loc[indices, "Platelets"] = clipped_normal(
            220, 45, 120, 400, n
        )

        df.loc[indices, "Mutation_Count"] = np.random.randint(
            1, 4, n
        )

        df.loc[indices, "Pathogenicity_Score"] = clipped_normal(
            0.85, 0.08, 0.60, 1.0, n
        )

    # --------------------------------------------------------
    # Huntington Disease
    # --------------------------------------------------------

    elif disorder == "Huntington Disease":

        df.loc[indices, "Age"] = clipped_normal(
            42, 10, 18, 70, n
        )

        df.loc[indices, "Mutation_Count"] = np.random.randint(
            2, 7, n
        )

        df.loc[indices, "Pathogenicity_Score"] = clipped_normal(
            0.90, 0.06, 0.65, 1.0, n
        )

    # --------------------------------------------------------
    # Phenylketonuria
    # --------------------------------------------------------

    elif disorder == "Phenylketonuria":

        df.loc[indices, "Mutation_Count"] = np.random.randint(
            1, 5, n
        )

        df.loc[indices, "Pathogenicity_Score"] = clipped_normal(
            0.82, 0.10, 0.55, 1.0, n
        )

        df.loc[indices, "Conservation_Score"] = clipped_normal(
            0.82, 0.08, 0.50, 1.0, n
        )

    # --------------------------------------------------------
    # Fragile X Syndrome
    # --------------------------------------------------------

    elif disorder == "Fragile X Syndrome":

        df.loc[indices, "Mutation_Count"] = np.random.randint(
            2, 8, n
        )

        df.loc[indices, "Variant_Length"] = np.random.randint(
            200, 1000, n
        )

        df.loc[indices, "Pathogenicity_Score"] = clipped_normal(
            0.90, 0.07, 0.60, 1.0, n
        )

    # --------------------------------------------------------
    # Duchenne Muscular Dystrophy
    # --------------------------------------------------------

    elif disorder == "Duchenne Muscular Dystrophy":

        df.loc[indices, "Sex"] = np.random.choice(
            ["Male", "Female"],
            size=n,
            p=[0.95, 0.05]
        )

        df.loc[indices, "Mutation_Count"] = np.random.randint(
            1, 5, n
        )

        df.loc[indices, "Pathogenicity_Score"] = clipped_normal(
            0.91, 0.06, 0.65, 1.0, n
        )

    # --------------------------------------------------------
    # Marfan Syndrome
    # --------------------------------------------------------

    elif disorder == "Marfan Syndrome":

        df.loc[indices, "Mutation_Count"] = np.random.randint(
            1, 4, n
        )

        df.loc[indices, "Pathogenicity_Score"] = clipped_normal(
            0.84, 0.10, 0.55, 1.0, n
        )

        df.loc[indices, "Conservation_Score"] = clipped_normal(
            0.84, 0.08, 0.55, 1.0, n
        )

    # --------------------------------------------------------
    # Tay-Sachs Disease
    # --------------------------------------------------------

    elif disorder == "Tay-Sachs Disease":

        df.loc[indices, "Mutation_Count"] = np.random.randint(
            1, 5, n
        )

        df.loc[indices, "Pathogenicity_Score"] = clipped_normal(
            0.90, 0.07, 0.65, 1.0, n
        )

        df.loc[indices, "Allele_Frequency"] = clipped_normal(
            0.005, 0.003, 0.0001, 0.02, n
        )

    # --------------------------------------------------------
    # Spinal Muscular Atrophy
    # --------------------------------------------------------

    elif disorder == "Spinal Muscular Atrophy":

        df.loc[indices, "Mutation_Count"] = np.random.randint(
            1, 5, n
        )

        df.loc[indices, "Pathogenicity_Score"] = clipped_normal(
            0.92, 0.06, 0.65, 1.0, n
        )

        df.loc[indices, "Conservation_Score"] = clipped_normal(
            0.87, 0.08, 0.55, 1.0, n
        )

    # --------------------------------------------------------
    # Wilson Disease
    # --------------------------------------------------------

    elif disorder == "Wilson Disease":

        df.loc[indices, "Mutation_Count"] = np.random.randint(
            1, 5, n
        )

        df.loc[indices, "Pathogenicity_Score"] = clipped_normal(
            0.84, 0.10, 0.55, 1.0, n
        )

    # --------------------------------------------------------
    # Gaucher Disease
    # --------------------------------------------------------

    elif disorder == "Gaucher Disease":

        df.loc[indices, "Platelets"] = clipped_normal(
            155, 35, 80, 300, n
        )

        df.loc[indices, "Hemoglobin"] = clipped_normal(
            10.8, 1.2, 7.5, 14, n
        )

        df.loc[indices, "Mutation_Count"] = np.random.randint(
            1, 5, n
        )

    # --------------------------------------------------------
    # Albinism
    # --------------------------------------------------------

    elif disorder == "Albinism":

        df.loc[indices, "Mutation_Count"] = np.random.randint(
            1, 4, n
        )

        df.loc[indices, "Pathogenicity_Score"] = clipped_normal(
            0.82, 0.10, 0.55, 1.0, n
        )

    # --------------------------------------------------------
    # Familial Hypercholesterolemia
    # --------------------------------------------------------

    elif disorder == "Familial Hypercholesterolemia":

        df.loc[indices, "Mutation_Count"] = np.random.randint(
            1, 4, n
        )

        df.loc[indices, "Pathogenicity_Score"] = clipped_normal(
            0.83, 0.10, 0.55, 1.0, n
        )

    # --------------------------------------------------------
    # No significant risk
    # --------------------------------------------------------

    elif disorder == "No Significant Genetic Risk":

        df.loc[indices, "Pathogenicity_Score"] = clipped_normal(
            0.20, 0.12, 0.01, 0.50, n
        )

        df.loc[indices, "Mutation_Count"] = np.random.randint(
            0, 2, n
        )

        df.loc[indices, "Variant_Count"] = np.random.randint(
            0, 3, n
        )


# ------------------------------------------------------------
# Generate one client dataset
# ------------------------------------------------------------

def generate_client_dataset(client_id, n_records):
    """
    Generate a synthetic dataset for one federated client.
    """

    # --------------------------------------------------------
    # Basic features
    # --------------------------------------------------------

    df = pd.DataFrame()

    df["Patient_ID"] = [
        f"C{client_id}_P{i:05d}"
        for i in range(1, n_records + 1)
    ]

    df["Age"] = np.random.randint(
    1, 81, n_records
    ).astype(float)

    df["Sex"] = np.random.choice(
        ["Male", "Female"],
        size=n_records,
        p=[0.50, 0.50]
    )

    # --------------------------------------------------------
    # Blood features
    # --------------------------------------------------------

    df["Hemoglobin"] = clipped_normal(
        13.2, 1.8, 7.0, 18.0, n_records
    )

    df["RBC"] = clipped_normal(
        4.7, 0.7, 2.5, 6.5, n_records
    )

    df["WBC"] = clipped_normal(
        7.2, 1.8, 3.0, 15.0, n_records
    )

    df["Platelets"] = clipped_normal(
        250, 55, 100, 450, n_records
    )

    df["PCV"] = clipped_normal(
        40, 5, 25, 55, n_records
    )

    df["MCV"] = clipped_normal(
        86, 8, 55, 110, n_records
    )

    df["MCH"] = clipped_normal(
        28, 3, 18, 35, n_records
    )

    df["MCHC"] = clipped_normal(
        33, 2, 25, 37, n_records
    )

    df["RDW"] = clipped_normal(
        13.5, 1.5, 10, 25, n_records
    )

    df["Neutrophils"] = clipped_normal(
        58, 8, 30, 85, n_records
    )

    df["Lymphocytes"] = clipped_normal(
        32, 7, 10, 55, n_records
    )

    df["Monocytes"] = clipped_normal(
        7, 2, 2, 15, n_records
    )

    df["Eosinophils"] = clipped_normal(
        2.5, 1.2, 0, 8, n_records
    )

    df["Basophils"] = clipped_normal(
        0.7, 0.3, 0, 2, n_records
    )

    # --------------------------------------------------------
    # Genetic features
    # --------------------------------------------------------

    df["Gene_Count"] = np.random.randint(
        1, 10, n_records
    )

    df["Variant_Count"] = np.random.randint(
        0, 8, n_records
    )

    df["Variant_Length"] = np.random.randint(
        50, 501, n_records
    )

    df["Allele_Frequency"] = np.round(
        np.random.beta(1.5, 15, n_records) * 0.20,
        5
    )

    df["Pathogenicity_Score"] = np.round(
        np.random.beta(2, 5, n_records),
        3
    )

    df["Conservation_Score"] = np.round(
        np.random.beta(5, 2, n_records),
        3
    )

    df["Mutation_Count"] = np.random.randint(
        0, 5, n_records
    )

    df["Heterozygosity"] = np.round(
        np.random.beta(3, 4, n_records),
        3
    )

    # --------------------------------------------------------
    # Variant categorical features
    # --------------------------------------------------------

    df["Variant_Type"] = generate_variant_type(
        n_records
    )

    df["Origin"] = generate_origin(
        n_records
    )

    df["Chromosome"] = generate_chromosome(
        n_records
    )

    # --------------------------------------------------------
    # Generate disorder labels
    #
    # Client 1 and Client 2 have slightly different
    # distributions to simulate non-IID federated data.
    # --------------------------------------------------------

    if client_id == 1:

        probabilities = [
            0.10,  # No risk
            0.09,  # Sickle Cell
            0.09,  # Beta Thalassemia
            0.07,  # Cystic Fibrosis
            0.06,  # Hemophilia A
            0.05,  # Huntington
            0.05,  # PKU
            0.05,  # Fragile X
            0.06,  # DMD
            0.06,  # Marfan
            0.05,  # Tay-Sachs
            0.06,  # SMA
            0.06,  # Wilson
            0.05,  # Gaucher
            0.05,  # Albinism
            0.05   # Familial Hypercholesterolemia
        ]

    else:

        probabilities = [
            0.12,  # No risk
            0.06,  # Sickle Cell
            0.06,  # Beta Thalassemia
            0.09,  # Cystic Fibrosis
            0.06,  # Hemophilia A
            0.07,  # Huntington
            0.06,  # PKU
            0.06,  # Fragile X
            0.05,  # DMD
            0.05,  # Marfan
            0.07,  # Tay-Sachs
            0.05,  # SMA
            0.06,  # Wilson
            0.06,  # Gaucher
            0.04,  # Albinism
            0.04   # Familial Hypercholesterolemia
        ]

    disorders = np.random.choice(
        DISORDERS,
        size=n_records,
        p=probabilities
    )

    df["Disorder"] = disorders
    df["Disorder_Code"] = df["Disorder"].map(
        DISORDER_CODES
    )

    # --------------------------------------------------------
    # Apply disorder-specific synthetic patterns
    # --------------------------------------------------------

    for disorder in DISORDERS:

        indices = df.index[
            df["Disorder"] == disorder
        ]

        apply_disorder_pattern(
            df,
            disorder,
            indices
        )

    # --------------------------------------------------------
    # Calculate genetic risk score
    # --------------------------------------------------------

    risk_score = (
        df["Pathogenicity_Score"] * 0.45
        + df["Conservation_Score"] * 0.15
        + np.clip(df["Mutation_Count"] / 5, 0, 1) * 0.20
        + np.clip(df["Variant_Count"] / 8, 0, 1) * 0.10
        + df["Heterozygosity"] * 0.10
    )

    # Add disease effect
    disease_mask = (
        df["Disorder"] != "No Significant Genetic Risk"
    )

    risk_score = np.where(
        disease_mask,
        risk_score + 0.15,
        risk_score
    )

    risk_score = np.clip(
        risk_score,
        0,
        1
    )

    # --------------------------------------------------------
    # Risk Level
    # --------------------------------------------------------

    risk_level = np.where(
        risk_score < 0.40,
        "Low",
        np.where(
            risk_score < 0.70,
            "Moderate",
            "High"
        )
    )

    # Force no-risk patients toward low risk
    no_risk_mask = (
        df["Disorder"] == "No Significant Genetic Risk"
    )

    risk_level = np.where(
        no_risk_mask,
        "Low",
        risk_level
    )

    df["Risk_Level"] = risk_level

    risk_code = {
        "Low": 0,
        "Moderate": 1,
        "High": 2
    }

    df["Risk_Code"] = df["Risk_Level"].map(
        risk_code
    )

    # --------------------------------------------------------
    # Stage
    # --------------------------------------------------------

    stage = []

    for risk in df["Risk_Level"]:

        if risk == "Low":
            stage.append("Normal")

        elif risk == "Moderate":

            stage.append(
                np.random.choice(
                    ["Early", "Moderate"],
                    p=[0.65, 0.35]
                )
            )

        else:

            stage.append(
                np.random.choice(
                    ["Moderate", "High"],
                    p=[0.40, 0.60]
                )
            )

    df["Stage"] = stage

    stage_code = {
        "Normal": 0,
        "Early": 1,
        "Moderate": 2,
        "High": 3
    }

    df["Stage_Code"] = df["Stage"].map(
        stage_code
    )

    # --------------------------------------------------------
    # Disorder code
    # --------------------------------------------------------

    df["Disorder_Code"] = df["Disorder"].map(
        DISORDER_CODES
    )

    # --------------------------------------------------------
    # Round numerical values
    # --------------------------------------------------------

    decimal_columns = [
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
        "Allele_Frequency",
        "Pathogenicity_Score",
        "Conservation_Score",
        "Heterozygosity"
    ]

    df[decimal_columns] = df[
        decimal_columns
    ].round(3)

    # --------------------------------------------------------
    # Reorder columns
    # --------------------------------------------------------

    columns = [
        "Patient_ID",
        "Age",
        "Sex",
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
        "Heterozygosity",
        "Variant_Type",
        "Origin",
        "Chromosome",
        "Disorder",
        "Disorder_Code",
        "Risk_Level",
        "Risk_Code",
        "Stage",
        "Stage_Code"
    ]

    df = df[columns]

    return df


# ------------------------------------------------------------
# Main dataset generation
# ------------------------------------------------------------

def main():

    print("=" * 60)
    print("GENETIC DISORDER SYNTHETIC DATASET GENERATOR")
    print("=" * 60)

    print(f"\nTotal records : {TOTAL_RECORDS}")
    print(f"Clients       : {NUM_CLIENTS}")
    print(f"Per client    : {RECORDS_PER_CLIENT}")
    print(f"Disorders     : {len(DISORDERS)}")
    print(f"Features      : 27")

    # Create output directory
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Generate Client 1
    # --------------------------------------------------------

    print("\nGenerating Client 1...")

    client1 = generate_client_dataset(
        client_id=1,
        n_records=RECORDS_PER_CLIENT
    )

    client1_path = os.path.join(
        OUTPUT_DIR,
        "client_1.csv"
    )

    client1.to_csv(
        client1_path,
        index=False
    )

    print(
        f"Client 1 saved: {client1_path}"
    )

    # --------------------------------------------------------
    # Generate Client 2
    # --------------------------------------------------------

    print("\nGenerating Client 2...")

    client2 = generate_client_dataset(
        client_id=2,
        n_records=RECORDS_PER_CLIENT
    )

    client2_path = os.path.join(
        OUTPUT_DIR,
        "client_2.csv"
    )

    client2.to_csv(
        client2_path,
        index=False
    )

    print(
        f"Client 2 saved: {client2_path}"
    )

    # --------------------------------------------------------
    # Combine both datasets
    # --------------------------------------------------------

    full_dataset = pd.concat(
        [client1, client2],
        ignore_index=True
    )

    full_path = os.path.join(
        OUTPUT_DIR,
        "genetic_disorder_dataset.csv"
    )

    full_dataset.to_csv(
        full_path,
        index=False
    )

    print(
        f"\nComplete dataset saved: {full_path}"
    )

    # --------------------------------------------------------
    # Dataset verification
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)

    print(
        f"\nDataset shape: {full_dataset.shape}"
    )

    print(
        f"\nExpected rows: {TOTAL_RECORDS}"
    )

    print(
        f"Actual rows  : {len(full_dataset)}"
    )

    print(
        f"\nMissing values:"
    )

    print(
        full_dataset.isnull().sum().sum()
    )

    # --------------------------------------------------------
    # Disorder distribution
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("DISORDER DISTRIBUTION")
    print("-" * 60)

    disorder_counts = (
        full_dataset["Disorder"]
        .value_counts()
        .sort_index()
    )

    print(disorder_counts)

    # --------------------------------------------------------
    # Risk distribution
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("RISK DISTRIBUTION")
    print("-" * 60)

    print(
        full_dataset["Risk_Level"]
        .value_counts()
    )

    # --------------------------------------------------------
    # Stage distribution
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("STAGE DISTRIBUTION")
    print("-" * 60)

    print(
        full_dataset["Stage"]
        .value_counts()
    )

    # --------------------------------------------------------
    # Client distribution
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("CLIENT DISTRIBUTION")
    print("-" * 60)

    print(
        "Client 1:",
        len(client1)
    )

    print(
        "Client 2:",
        len(client2)
    )

    # --------------------------------------------------------
    # Show sample records
    # --------------------------------------------------------

    print("\n" + "-" * 60)
    print("SAMPLE RECORDS")
    print("-" * 60)

    print(
        full_dataset.head(5).to_string(
            index=False
        )
    )

    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETED")
    print("=" * 60)

    print("\nFiles created:")

    print(
        "1.",
        full_path
    )

    print(
        "2.",
        client1_path
    )

    print(
        "3.",
        client2_path
    )


# ------------------------------------------------------------
# Execute
# ------------------------------------------------------------

if __name__ == "__main__":
    main()