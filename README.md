🧬 Genetic Disorder Risk Prediction Using AI and Federated Learning
📌 Overview
Genetic Disorder Risk Prediction Using AI and Federated Learning is an academic/research project that demonstrates how Artificial Intelligence (AI), Deep Learning, and Federated Learning can be combined to classify genetic-disorder categories using blood-related, genetic, and variant-related features.
The system uses a synthetic dataset containing 30,000 records, distributed across 2 federated clients.
Each client performs local model training, and the Flower Federated Learning server aggregates the client model updates using Federated Averaging (FedAvg).
The final global model predicts one of 16 classes.
⚠️ Academic/Research Demonstration Only: This project uses synthetic data and is not a clinical diagnostic tool.
________________________________________
🎯 Aim
To develop an AI-based genetic-disorder classification system using Federated Learning, where multiple clients can train a shared neural-network model without combining their local datasets into one centralized training dataset.
________________________________________
🧠 Role of AI
AI is the main prediction component of this project.
A neural-network classifier receives processed blood, genetic, and variant-related features and learns patterns associated with the 16 disorder classes.
AI Pipeline
Patient / Biological Data
          ↓
Data Preprocessing
          ↓
27 Original Features
          ↓
StandardScaler + OneHotEncoder
          ↓
56 Processed Features
          ↓
Neural Network
          ↓
Softmax Output
          ↓
16 Disorder Classes
          ↓
Prediction
The neural network architecture is:
56 Input Features
       ↓
Dense(128)
       ↓
Batch Normalization
       ↓
Dropout
       ↓
Dense(64)
       ↓
Batch Normalization
       ↓
Dropout
       ↓
Dense(32)
       ↓
Dense(16)
       ↓
Softmax
________________________________________
🔐 Why Federated Learning?
In conventional machine learning, data from different sources may be combined into a central dataset.
Federated Learning follows a different approach:
                    Global Model
                         │
              ┌──────────┴──────────┐
              ↓                     ↓
          Client 1              Client 2
       15,000 records         15,000 records
              ↓                     ↓
       Local Training          Local Training
              ↓                     ↓
       Model Updates           Model Updates
              └──────────┬──────────┘
                         ↓
                       FedAvg
                         ↓
                   Global Model
                         ↓
                  Next FL Round
This project uses exactly 2 federated clients.
________________________________________
📊 Dataset
The dataset is synthetically generated for academic experimentation.
Parameter	Value
Total records	30,000
Federated clients	2
Records per client	15,000
Original features	27
Processed features	56
Classes	16
Training data	80%
Testing data	20%
Dataset Files
data/
├── genetic_disorder_dataset.csv
├── client_1.csv
└── client_2.csv
________________________________________
🧬 Predicted Classes
The model contains 16 output classes:
Code	Class
0	No Significant Genetic Risk
1	Sickle Cell Disease
2	Beta Thalassemia
3	Cystic Fibrosis
4	Hemophilia A
5	Huntington Disease
6	Phenylketonuria
7	Fragile X Syndrome
8	Duchenne Muscular Dystrophy
9	Marfan Syndrome
10	Tay-Sachs Disease
11	Spinal Muscular Atrophy
12	Wilson Disease
13	Gaucher Disease
14	Albinism
15	Familial Hypercholesterolemia
________________________________________
🔢 Input Features
The model starts with 27 input features.
Blood / Patient Features
•	Age
•	Hemoglobin
•	RBC
•	WBC
•	Platelets
•	PCV
•	MCV
•	MCH
•	MCHC
•	RDW
•	Neutrophils
•	Lymphocytes
•	Monocytes
•	Eosinophils
•	Basophils
Genetic / Variant Features
•	Gene Count
•	Variant Count
•	Variant Length
•	Allele Frequency
•	Pathogenicity Score
•	Conservation Score
•	Mutation Count
•	Heterozygosity
Categorical Features
•	Sex
•	Variant Type
•	Origin
•	Chromosome
________________________________________
⚙️ Data Preprocessing
The preprocessing pipeline separates features into numerical and categorical groups.
Numerical Features
Numerical values are standardized using:
StandardScaler
Categorical Features
Categorical values are converted into numerical representation using:
OneHotEncoder
After preprocessing:
27 original features
        ↓
56 processed features
Each client has its own preprocessing object.
data/preprocessors/
├── client_1_preprocessor.pkl
└── client_2_preprocessor.pkl
________________________________________
🏗️ Project Architecture
                   SYNTHETIC DATASET
                    30,000 RECORDS
                          │
              ┌───────────┴───────────┐
              ↓                       ↓
          CLIENT 1                CLIENT 2
       15,000 records           15,000 records
              │                       │
              ↓                       ↓
       PREPROCESSING             PREPROCESSING
              │                       │
              ↓                       ↓
        56 FEATURES              56 FEATURES
              │                       │
              ↓                       ↓
       LOCAL TRAINING            LOCAL TRAINING
              │                       │
              └───────────┬───────────┘
                          ↓
                    FLOWER SERVER
                          ↓
                        FedAvg
                          ↓
                  GLOBAL MODEL
                          ↓
              global_model.keras
                          ↓
                     EVALUATION
                          ↓
                     STREAMLIT
                          ↓
                  USER PREDICTION
________________________________________
📁 Project Structure
GENETICRISK_PREDICTOR/
│
├── app.py
├── dataset_generator.py
├── preprocessing.py
├── model.py
├── federated_client.py
├── federated_server.py
├── evaluate.py
├── test_prediction.py
│
├── data/
│   ├── genetic_disorder_dataset.csv
│   ├── client_1.csv
│   ├── client_2.csv
│   │
│   ├── processed/
│   │   ├── client_1_X_train.npy
│   │   ├── client_1_X_test.npy
│   │   ├── client_1_y_train.npy
│   │   ├── client_1_y_test.npy
│   │   ├── client_2_X_train.npy
│   │   ├── client_2_X_test.npy
│   │   ├── client_2_y_train.npy
│   │   ├── client_2_y_test.npy
│   │   └── global_metadata.json
│   │
│   └── preprocessors/
│       ├── client_1_preprocessor.pkl
│       └── client_2_preprocessor.pkl
│
└── models/
    └── global_model.keras
________________________________________
🛠️ Technologies
Technology	Purpose
Python	Programming language
Pandas	Dataset processing
NumPy	Numerical computation
Scikit-learn	Data preprocessing
TensorFlow	Deep Learning
Keras	Neural Network
Flower	Federated Learning
Joblib	Saving preprocessors
Streamlit	Web application
________________________________________
💻 Requirements
Recommended environment used for the project:
Python 3.11.9
Windows 64-bit
AMD64
TensorFlow 2.21.0
Flower 1.36.0
________________________________________
📦 Installation
Create a virtual environment:
python -m venv myenv
Activate it:
myenv\Scripts\activate
Install the required packages:
pip install numpy pandas scikit-learn tensorflow flwr joblib streamlit
Check Python:
python --version
Check TensorFlow:
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
Check Flower:
python -m pip show flwr
________________________________________
▶️ Running the Project
Step 1 — Generate Dataset
Run:
python dataset_generator.py
Expected configuration:
Total records : 30000
Clients       : 2
Per client    : 15000
Disorders     : 16
Features      : 27
The generator creates:
data/client_1.csv
data/client_2.csv
data/genetic_disorder_dataset.csv
________________________________________
Step 2 — Preprocess Dataset
Run:
python preprocessing.py
Expected output:
Client 1:
X_train: (12000, 56)
X_test : (3000, 56)
y_train: (12000,)
y_test : (3000,)

Client 2:
X_train: (12000, 56)
X_test : (3000, 56)
y_train: (12000,)
y_test : (3000,)
The preprocessing stage also checks for missing values.
Expected:
NaN verification:
Client 1 X_train: 0
Client 1 X_test : 0
Client 2 X_train: 0
Client 2 X_test : 0
________________________________________
Step 3 — Test AI Model
Run:
python model.py
Expected configuration:
Input features : 56
Output classes : 16
The model should successfully generate predictions with:
Prediction shape : (5, 16)
The 16 output values represent the model's probability distribution over the 16 classes.
________________________________________
Step 4 — Run Federated Learning
Run:
python federated_server.py
The project uses:
Number of clients : 2
Federated rounds  : 5
Input features    : 56
Output classes    : 16
Aggregation       : FedAvg
The Flower server coordinates training between Client 1 and Client 2.
After training, the global model is saved as:
models/global_model.keras
________________________________________
🔄 Federated Learning Rounds
The project performs 5 rounds.
Round 1
Global Model
     ↓
Client 1 + Client 2
     ↓
Local Training
     ↓
FedAvg
Round 2
Updated Global Model
     ↓
Client 1 + Client 2
     ↓
Local Training
     ↓
FedAvg
The same process continues through Round 5.
________________________________________
📈 Federated Learning Results
The completed experiment produced:
Round	Distributed Evaluation Accuracy
1	55.35%
2	56.95%
3	57.43%
4	57.48%
5	57.77%
Final distributed evaluation accuracy:
57.77%
________________________________________
🧪 Global Model Evaluation
Run:
python evaluate.py
The final model was evaluated using:
Client 1 test data : 3000
Client 2 test data : 3000
Combined test data : 6000
Overall Results
Metric	Result
Test Loss	1.1387
Accuracy	57.77%
Weighted Precision	58.47%
Weighted Recall	57.77%
Weighted F1 Score	55.99%
________________________________________
🔬 Prediction Testing
The test_prediction.py script can be used to directly test the global model against existing test records.
Example:
Actual:
Duchenne Muscular Dystrophy

Predicted:
Duchenne Muscular Dystrophy

Confidence:
46.16%
The model also generates probabilities for all 16 classes.
Example:
Duchenne Muscular Dystrophy       46.16%
Familial Hypercholesterolemia     14.77%
Albinism                           13.09%
Wilson Disease                     11.39%
Huntington Disease                  7.70%
...
________________________________________
🌐 Streamlit Application
The project includes a graphical web interface.
Start it using:
streamlit run app.py
The application loads:
models/global_model.keras
and:
data/preprocessors/client_1_preprocessor.pkl
The user can enter patient, blood, genetic, and variant information.
The application then:
User Input
    ↓
Create DataFrame
    ↓
Preprocessor
    ↓
56 Features
    ↓
Global Neural Network
    ↓
Softmax Probabilities
    ↓
Predicted Class
________________________________________
🖥️ Application Output
The Streamlit application displays:
•	Predicted Disorder
•	Prediction Confidence
•	Risk Level
•	Risk Progression Stage
•	Probability Distribution
•	Top 5 Predictions
________________________________________
📊 Risk Level
The current application converts model confidence into a displayed risk level.
Confidence ≥ 75%
        ↓
      High

50% ≤ Confidence < 75%
        ↓
    Moderate

Confidence < 50%
        ↓
       Low
For a predicted class of:
No Significant Genetic Risk
the application displays:
Normal / No Risk
________________________________________
⚠️ Important Note About Confidence
The neural network's Softmax value is a model confidence score, not a medically validated probability.
For example:
Duchenne Muscular Dystrophy
Confidence: 46.16%
means the model assigned the highest Softmax probability to that class.
It does not mean that a patient has a medically validated 46.16% probability of having that disorder.
________________________________________
🔒 Federated Learning and Privacy
The project demonstrates the basic Federated Learning concept by maintaining two separate client datasets.
Client 1 Data
      ↓
Client 1 Local Model
      ↓
      │
      ├────── FedAvg ──────→ Global Model
      │
      ↓
Client 2 Local Model
      ↑
Client 2 Data
The client datasets are not merged into a single training dataset for the federated training process.
________________________________________
📌 Important Project Observation
The current model has an overall test accuracy of approximately 57.77%.
Some classes perform substantially better than others.
Therefore, the model should be presented as an experimental academic prototype, not as a clinically reliable diagnostic model.
________________________________________
🚧 Limitations
1. Synthetic Dataset
The current dataset is synthetically generated.
It does not represent the complete complexity of real patient genomic and clinical data.
2. Limited Number of Clients
Only two federated clients are used.
3. Model Performance
The final accuracy is:
57.77%
Therefore, the model requires further research and validation before any real-world application.
4. Class Imbalance
The number of records varies between disorder classes.
This can affect classification performance.
5. No Clinical Validation
The model has not been clinically validated.
6. No Medical Decision Making
The output should not be used to diagnose or treat a person.
________________________________________
🔮 Future Scope
Possible future improvements include:
•	Larger real-world datasets
•	Clinically validated genetic datasets
•	More federated clients
•	Better class balancing
•	Hyperparameter optimization
•	Advanced neural networks
•	Transformer-based genomic models
•	Graph Neural Networks
•	Graph Attention Networks
•	Knowledge Graph integration
•	Explainable AI
•	Differential Privacy
•	Secure Aggregation
•	More extensive external validation
________________________________________
🎓 Academic Workflow
The complete project workflow can be summarized as:
1. Dataset Generation
          ↓
2. Dataset Validation
          ↓
3. Train/Test Split
          ↓
4. Feature Scaling
          ↓
5. Categorical Encoding
          ↓
6. Local Client Data
          ↓
7. Neural Network Creation
          ↓
8. Client Local Training
          ↓
9. Federated Aggregation
          ↓
10. FedAvg
          ↓
11. Global Model
          ↓
12. Model Evaluation
          ↓
13. Prediction
          ↓
14. Streamlit Interface

 
📜 Disclaimer
This project is an academic/research demonstration.
The system uses synthetic data and an experimental AI model. Its predictions are not medically validated and must not be used for diagnosis, treatment, or clinical decision-making.
________________________________________
👨‍💻 Project Status
Dataset Generation       ✅ Completed
Data Preprocessing       ✅ Completed
AI Model                 ✅ Completed
Federated Client Setup   ✅ Completed
Federated Server         ✅ Completed
FedAvg Training          ✅ Completed
5 FL Rounds              ✅ Completed
Global Model             ✅ Saved
Model Evaluation         ✅ Completed
Prediction Testing       ✅ Completed
Streamlit Application    ✅ Completed
________________________________________
⭐ Project Summary
Genetic Disorder Risk Prediction Using AI and Federated Learning
30,000 Synthetic Records
          ↓
2 Federated Clients
          ↓
27 Original Features
          ↓
56 Processed Features
          ↓
Neural Network
          ↓
16 Disorder Classes
          ↓
5 Federated Rounds
          ↓
FedAvg
          ↓
Global Model
          ↓
57.77% Experimental Accuracy
          ↓
Streamlit Prediction Application
Technologies: Python • TensorFlow • Keras • Scikit-learn • Flower • Pandas • NumPy • Streamlit

