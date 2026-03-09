import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib

# Get the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "vehicles_ml_dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classification_model.pkl")

df = pd.read_csv(DATA_PATH)

# Define features and target (target moved up for logical flow)
features = ["year", "kilometers_driven", "seating_capacity", "estimated_income"]
target = "income_level"

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, MODEL_PATH)

# Predict
predictions = model.predict(X_test)

# Calculate Accuracy Score (equivalent to R2 in your regression example)
accuracy = round(accuracy_score(y_test, predictions) * 100, 2)

# Create a Comparison DataFrame for the data_exploration
comparison_df = pd.DataFrame(
    {
        "Actual": y_test.values,
        "Predicted": predictions,
        "Match": y_test.values == predictions,
    }
)

def evaluate_classification_model():
    return {
        "accuracy": accuracy,
        "comparison": comparison_df.head(10).to_html(
            classes="table table-bordered table-striped table-sm",
            justify="center",
        ),
    }
