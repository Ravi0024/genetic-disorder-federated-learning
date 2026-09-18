# ============================================================
# app.py
# Genetic Disorder Risk Prediction
# Streamlit Application
# ============================================================

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/global_model.keras"

PREPROCESSOR_PATH = (
    "data/preprocessors/client_1_preprocessor.pkl"
)

CLIENT_DATA_PATH = "data/client_1.csv"

INPUT_DIM = 56
NUM_CLASSES = 16


# ============================================================
# DISORDER MAPPING
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
# CATEGORIES
# ============================================================

VARIANT_TYPES = [
    "SNV",
    "Insertion",
    "Deletion",
    "Duplication"
]

# IMPORTANT:
# These values MUST exactly match the dataset.
ORIGINS = [
    "De_Novo",
    "Inherited",
    "Germline"
]

CHROMOSOMES = [
    "1", "2", "3", "4", "5", "6", "7", "8",
    "9", "10", "11", "12", "13", "14", "15",
    "16", "17", "18", "19", "20", "21", "22",
    "X", "Y"
]

SEXES = [
    "Male",
    "Female"
]


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Genetic Disorder Risk Predictor",
    page_icon="🧬",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Global model not found: {MODEL_PATH}"
        )

    # compile=False because we only need prediction
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# ============================================================
# LOAD PREPROCESSOR
# ============================================================

@st.cache_resource
def load_preprocessor():

    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            f"Preprocessor not found: {PREPROCESSOR_PATH}"
        )

    return joblib.load(
        PREPROCESSOR_PATH
    )


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_client_data():

    if not os.path.exists(CLIENT_DATA_PATH):
        return None

    return pd.read_csv(
        CLIENT_DATA_PATH
    )


# ============================================================
# RISK LEVEL
# ============================================================

def calculate_risk_level(confidence):

    if confidence >= 0.75:
        return "High"

    elif confidence >= 0.50:
        return "Moderate"

    else:
        return "Low"


# ============================================================
# STAGE
# ============================================================

def calculate_stage(risk_level):

    if risk_level == "High":
        return "High Risk"

    elif risk_level == "Moderate":
        return "Moderate Risk"

    else:
        return "Early Risk"


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

def create_input_dataframe(
    age,
    sex,
    hemoglobin,
    rbc,
    wbc,
    platelets,
    pcv,
    mcv,
    mch,
    mchc,
    rdw,
    neutrophils,
    lymphocytes,
    monocytes,
    eosinophils,
    basophils,
    gene_count,
    variant_count,
    variant_length,
    allele_frequency,
    pathogenicity_score,
    conservation_score,
    mutation_count,
    heterozygosity,
    variant_type,
    origin,
    chromosome
):

    data = {

        "Age": [age],

        "Hemoglobin": [hemoglobin],

        "RBC": [rbc],

        "WBC": [wbc],

        "Platelets": [platelets],

        "PCV": [pcv],

        "MCV": [mcv],

        "MCH": [mch],

        "MCHC": [mchc],

        "RDW": [rdw],

        "Neutrophils": [neutrophils],

        "Lymphocytes": [lymphocytes],

        "Monocytes": [monocytes],

        "Eosinophils": [eosinophils],

        "Basophils": [basophils],

        "Gene_Count": [gene_count],

        "Variant_Count": [variant_count],

        "Variant_Length": [variant_length],

        "Allele_Frequency": [allele_frequency],

        "Pathogenicity_Score": [pathogenicity_score],

        "Conservation_Score": [conservation_score],

        "Mutation_Count": [mutation_count],

        "Heterozygosity": [heterozygosity],

        "Sex": [sex],

        "Variant_Type": [variant_type],

        "Origin": [origin],

        "Chromosome": [chromosome]
    }

    return pd.DataFrame(data)


# ============================================================
# PREDICT FUNCTION
# ============================================================

def make_prediction(
    input_data,
    model,
    preprocessor
):

    # -----------------------------
    # Preprocess
    # -----------------------------

    processed_input = preprocessor.transform(
        input_data
    )

    processed_input = np.asarray(
        processed_input,
        dtype=np.float32
    )

    # -----------------------------
    # Verify dimensions
    # -----------------------------

    if processed_input.shape[1] != INPUT_DIM:

        raise ValueError(
            f"Expected {INPUT_DIM} processed features "
            f"but received {processed_input.shape[1]}"
        )

    # -----------------------------
    # Prediction
    # -----------------------------

    probabilities = model.predict(
        processed_input,
        verbose=0
    )[0]

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    # Normalize just in case
    probabilities = (
        probabilities /
        probabilities.sum()
    )

    predicted_class = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_class]
    )

    disorder = CLASS_NAMES[
        predicted_class
    ]

    # -----------------------------
    # Risk
    # -----------------------------

    if disorder == "No Significant Genetic Risk":

        risk_level = "Low"
        stage = "Normal / No Risk"

    else:

        risk_level = calculate_risk_level(
            confidence
        )

        stage = calculate_stage(
            risk_level
        )

    return (
        probabilities,
        predicted_class,
        disorder,
        confidence,
        risk_level,
        stage,
        processed_input
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    st.title(
        "🧬 Genetic Disorder Risk Prediction System"
    )

    st.markdown(
        """
        ### Artificial Intelligence + Federated Learning

        This system predicts a genetic disorder category
        using blood, genetic and variant-related features.
        """
    )

    st.warning(
        "⚠️ Academic/Research Demonstration Only — "
        "This system uses synthetic data and is NOT a "
        "clinical diagnostic tool."
    )

    st.divider()

    # --------------------------------------------------------
    # Load resources
    # --------------------------------------------------------

    try:

        model = load_model()

        preprocessor = load_preprocessor()

        dataset = load_client_data()

    except Exception as e:

        st.error(
            f"Unable to load model or preprocessor: {e}"
        )

        st.stop()

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    with st.expander(
        "🔧 Model / Preprocessing Information"
    ):

        st.write(
            f"Model input dimension: {INPUT_DIM}"
        )

        st.write(
            f"Number of classes: {NUM_CLASSES}"
        )

        st.write(
            "Preprocessor: Client 1 preprocessor"
        )

        if dataset is not None:

            st.write(
                f"Client 1 dataset records: {len(dataset):,}"
            )

    # --------------------------------------------------------
    # TEST MODE
    # --------------------------------------------------------

    st.header(
        "🧪 Prediction Mode"
    )

    test_mode = st.radio(
        "Choose input method:",
        [
            "Manual Patient Input",
            "Test with Dataset Sample"
        ],
        horizontal=True
    )

    # ========================================================
    # DATASET SAMPLE MODE
    # ========================================================

    if test_mode == "Test with Dataset Sample":

        if dataset is None:

            st.error(
                "data/client_1.csv was not found."
            )

            st.stop()

        st.info(
            "This mode uses an actual row from Client 1 "
            "to verify that the saved preprocessor and "
            "global model are working correctly."
        )

        sample_number = st.number_input(
            "Dataset Row Number",
            min_value=0,
            max_value=len(dataset) - 1,
            value=0,
            step=1
        )

        sample = dataset.iloc[
            int(sample_number)
        ]

        st.subheader(
            "Original Dataset Record"
        )

        display_columns = [
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
            "Chromosome"
        ]

        sample_display = pd.DataFrame(
            sample[display_columns]
        ).T

        st.dataframe(
            sample_display,
            use_container_width=True,
            hide_index=False
        )

        input_data = pd.DataFrame(
            [sample[display_columns].to_dict()]
        )

        predict_button = st.button(
            "🔍 Predict Dataset Sample",
            type="primary",
            use_container_width=True
        )

    # ========================================================
    # MANUAL MODE
    # ========================================================

    else:

        # ----------------------------------------------------
        # Patient Information
        # ----------------------------------------------------

        st.header(
            "Patient Information"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            age = st.number_input(
                "Age",
                min_value=0.0,
                max_value=120.0,
                value=25.0,
                step=1.0
            )

            sex = st.selectbox(
                "Sex",
                SEXES
            )

            hemoglobin = st.number_input(
                "Hemoglobin (g/dL)",
                min_value=1.0,
                max_value=25.0,
                value=13.0
            )

            rbc = st.number_input(
                "RBC",
                min_value=1.0,
                max_value=10.0,
                value=4.5
            )

            wbc = st.number_input(
                "WBC",
                min_value=1000.0,
                max_value=30000.0,
                value=7000.0
            )

            platelets = st.number_input(
                "Platelets",
                min_value=50000.0,
                max_value=1000000.0,
                value=250000.0
            )

        with col2:

            pcv = st.number_input(
                "PCV (%)",
                min_value=10.0,
                max_value=70.0,
                value=42.0
            )

            mcv = st.number_input(
                "MCV",
                min_value=50.0,
                max_value=150.0,
                value=85.0
            )

            mch = st.number_input(
                "MCH",
                min_value=10.0,
                max_value=50.0,
                value=28.0
            )

            mchc = st.number_input(
                "MCHC",
                min_value=20.0,
                max_value=40.0,
                value=33.0
            )

            rdw = st.number_input(
                "RDW",
                min_value=5.0,
                max_value=30.0,
                value=13.0
            )

            neutrophils = st.number_input(
                "Neutrophils (%)",
                min_value=0.0,
                max_value=100.0,
                value=55.0
            )

            lymphocytes = st.number_input(
                "Lymphocytes (%)",
                min_value=0.0,
                max_value=100.0,
                value=35.0
            )

        with col3:

            monocytes = st.number_input(
                "Monocytes (%)",
                min_value=0.0,
                max_value=30.0,
                value=7.0
            )

            eosinophils = st.number_input(
                "Eosinophils (%)",
                min_value=0.0,
                max_value=30.0,
                value=2.0
            )

            basophils = st.number_input(
                "Basophils (%)",
                min_value=0.0,
                max_value=10.0,
                value=1.0
            )

            gene_count = st.number_input(
                "Gene Count",
                min_value=0.0,
                max_value=100.0,
                value=1.0,
                step=1.0
            )

            variant_count = st.number_input(
                "Variant Count",
                min_value=0.0,
                max_value=100.0,
                value=1.0,
                step=1.0
            )

            variant_length = st.number_input(
                "Variant Length",
                min_value=1.0,
                max_value=10000.0,
                value=100.0
            )

        # ----------------------------------------------------
        # Genetic Information
        # ----------------------------------------------------

        st.divider()

        st.header(
            "Genetic & Variant Information"
        )

        col4, col5, col6 = st.columns(3)

        with col4:

            allele_frequency = st.number_input(
                "Allele Frequency",
                min_value=0.0,
                max_value=1.0,
                value=0.01,
                format="%.4f"
            )

            pathogenicity_score = st.number_input(
                "Pathogenicity Score",
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                format="%.4f"
            )

            conservation_score = st.number_input(
                "Conservation Score",
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                format="%.4f"
            )

        with col5:

            mutation_count = st.number_input(
                "Mutation Count",
                min_value=0.0,
                max_value=100.0,
                value=1.0,
                step=1.0
            )

            heterozygosity = st.number_input(
                "Heterozygosity",
                min_value=0.0,
                max_value=1.0,
                value=0.50,
                format="%.4f"
            )

            variant_type = st.selectbox(
                "Variant Type",
                VARIANT_TYPES
            )

        with col6:

            origin = st.selectbox(
                "Origin",
                ORIGINS
            )

            chromosome = st.selectbox(
                "Chromosome",
                CHROMOSOMES
            )

        st.divider()

        # ----------------------------------------------------
        # Create input dataframe
        # ----------------------------------------------------

        input_data = create_input_dataframe(
            age,
            sex,
            hemoglobin,
            rbc,
            wbc,
            platelets,
            pcv,
            mcv,
            mch,
            mchc,
            rdw,
            neutrophils,
            lymphocytes,
            monocytes,
            eosinophils,
            basophils,
            gene_count,
            variant_count,
            variant_length,
            allele_frequency,
            pathogenicity_score,
            conservation_score,
            mutation_count,
            heterozygosity,
            variant_type,
            origin,
            chromosome
        )

        predict_button = st.button(
            "🔍 Predict Genetic Disorder Risk",
            type="primary",
            use_container_width=True
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:

        try:

            # ------------------------------------------------
            # Show raw input
            # ------------------------------------------------

            with st.expander(
                "View Input Data"
            ):

                st.dataframe(
                    input_data,
                    use_container_width=True,
                    hide_index=True
                )

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            (
                probabilities,
                predicted_class,
                disorder,
                confidence,
                risk_level,
                stage,
                processed_input
            ) = make_prediction(
                input_data,
                model,
                preprocessor
            )

            # ------------------------------------------------
            # Debug information
            # ------------------------------------------------

            with st.expander(
                "🔧 Prediction Debug Information"
            ):

                st.write(
                    "Processed input shape:",
                    processed_input.shape
                )

                st.write(
                    "Processed input minimum:",
                    float(processed_input.min())
                )

                st.write(
                    "Processed input maximum:",
                    float(processed_input.max())
                )

                st.write(
                    "Probability sum:",
                    float(probabilities.sum())
                )

                st.write(
                    "Predicted class code:",
                    predicted_class
                )

            # ------------------------------------------------
            # Result
            # ------------------------------------------------

            st.divider()

            st.header(
                "Prediction Result"
            )

            result_col1, result_col2, result_col3 = st.columns(3)

            with result_col1:

                st.metric(
                    "Predicted Disorder",
                    disorder
                )

            with result_col2:

                st.metric(
                    "Prediction Confidence",
                    f"{confidence * 100:.2f}%"
                )

            with result_col3:

                st.metric(
                    "Risk Level",
                    risk_level
                )

            st.subheader(
                "Risk Progression Stage"
            )

            st.info(
                stage
            )

            # ------------------------------------------------
            # Actual dataset label
            # ------------------------------------------------

            if test_mode == "Test with Dataset Sample":

                actual_disorder = sample[
                    "Disorder"
                ]

                st.subheader(
                    "Dataset Comparison"
                )

                if disorder == actual_disorder:

                    st.success(
                        f"Prediction matches dataset label: "
                        f"{actual_disorder}"
                    )

                else:

                    st.warning(
                        f"Dataset label: {actual_disorder} | "
                        f"Model prediction: {disorder}"
                    )

            # ------------------------------------------------
            # Probability distribution
            # ------------------------------------------------

            st.subheader(
                "Prediction Probability Distribution"
            )

            probability_df = pd.DataFrame({

                "Disorder": CLASS_NAMES,

                "Probability (%)":
                    probabilities * 100

            })

            probability_df = (
                probability_df
                .sort_values(
                    "Probability (%)",
                    ascending=False
                )
                .reset_index(drop=True)
            )

            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # Top 5
            # ------------------------------------------------

            st.subheader(
                "Top 5 Predictions"
            )

            top_indices = np.argsort(
                probabilities
            )[::-1][:5]

            top_predictions = pd.DataFrame({

                "Rank": range(1, 6),

                "Disorder": [
                    CLASS_NAMES[i]
                    for i in top_indices
                ],

                "Probability (%)": [
                    probabilities[i] * 100
                    for i in top_indices
                ]
            })

            st.dataframe(
                top_predictions,
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------
            # Disclaimer
            # ------------------------------------------------

            st.warning(
                "This prediction is generated by an AI model "
                "trained on synthetic academic data. It should "
                "not be used for medical diagnosis, treatment, "
                "or clinical decision-making."
            )

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()