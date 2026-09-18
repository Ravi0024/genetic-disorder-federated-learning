import numpy as np
import tensorflow as tf

MODEL_PATH = "models/global_model.keras"

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

print("=" * 60)
print("DIRECT GLOBAL MODEL TEST")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)

# Use an actual processed test record
X_test = np.load(
    "data/processed/client_1_X_test.npy"
)

y_test = np.load(
    "data/processed/client_1_y_test.npy"
)

# Test first 10 real processed records
X_sample = X_test[:10]
y_actual = y_test[:10]

probabilities = model.predict(
    X_sample,
    verbose=0
)

for i in range(10):

    predicted = int(
        np.argmax(probabilities[i])
    )

    confidence = float(
        probabilities[i][predicted]
    )

    print()
    print(f"Sample {i + 1}")
    print(
        "Actual    :",
        CLASS_NAMES[int(y_actual[i])]
    )
    print(
        "Predicted :",
        CLASS_NAMES[predicted]
    )
    print(
        "Confidence:",
        f"{confidence * 100:.2f}%"
    )

    print("Top 3:")

    top3 = np.argsort(
        probabilities[i]
    )[::-1][:3]

    for rank, index in enumerate(top3, 1):

        print(
            f"  {rank}. "
            f"{CLASS_NAMES[index]} - "
            f"{probabilities[i][index] * 100:.2f}%"
        )

print()
print("=" * 60)