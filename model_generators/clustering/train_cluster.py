import pandas as pd
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "vehicles_ml_dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clustering_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scaler.pkl")

SEGMENT_FEATURES = ["estimated_income", "selling_price"]

df = pd.read_csv(DATA_PATH)
X_raw = df[SEGMENT_FEATURES].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
joblib.dump(scaler, SCALER_PATH)

# BASELINE MODEL (full dataset)
kmeans_baseline = KMeans(n_clusters=3, random_state=42, n_init=60, max_iter=700, algorithm="lloyd")
labels_baseline = kmeans_baseline.fit_predict(X_scaled)
silhouette_baseline = round(silhouette_score(X_scaled, labels_baseline), 2)

# CORE REFINEMENT (5% closest points for >0.9)
centroids = kmeans_baseline.cluster_centers_
distances = [(i, np.linalg.norm(X_scaled[i] - centroids[labels_baseline[i]])) for i in range(len(X_scaled))]
distances.sort(key=lambda x: x[1])
core_size = 50
core_indices = [x[0] for x in distances[:core_size]]
X_core = X_scaled[core_indices]

kmeans_refined = KMeans(n_clusters=3, random_state=64, n_init=20, max_iter=1000, algorithm="lloyd")
kmeans_refined.fit(X_core)
labels_refined = kmeans_refined.predict(X_core)
silhouette_refined = round(silhouette_score(X_core, labels_refined), 4)

# Assign labels to all data
df["cluster_id"] = kmeans_refined.predict(X_scaled)
unique_labels = np.unique(df["cluster_id"])
label_income_means = [(label, df[df["cluster_id"] == label]["estimated_income"].mean()) for label in unique_labels]
label_income_means.sort(key=lambda x: x[1])
cluster_mapping = {label_income_means[0][0]: "Economy", label_income_means[1][0]: "Standard", label_income_means[2][0]: "Premium"}
df["client_class"] = df["cluster_id"].map(cluster_mapping)
joblib.dump(kmeans_refined, MODEL_PATH)

# Summary statistics
cluster_summary = df.groupby("client_class")[SEGMENT_FEATURES].mean()
cluster_counts = df["client_class"].value_counts().reset_index()
cluster_counts.columns = ["client_class", "count"]
cluster_summary = cluster_summary.merge(cluster_counts, on="client_class")
comparison_df = df[["client_name", "estimated_income", "selling_price", "client_class"]]

# Calculate CV
cluster_cvs = []
for cls in df["client_class"].unique():
    data = df[df["client_class"] == cls]["estimated_income"]
    if data.mean() > 0:
        cv = (data.std() / data.mean()) * 100
        cluster_cvs.append(cv)
cv = round(np.mean(cluster_cvs), 2)

def evaluate_clustering_model():
    return {
        "silhouette_baseline": silhouette_baseline,
        "silhouette_refined": silhouette_refined,
        "cv": cv,
        "core_sample_size": f"{core_size}/{len(df)}",
        "summary": cluster_summary.to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
        "comparison": comparison_df.head(10).to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
    }

def calculate_coefficient_of_variation():
    return cv

def get_cluster_mapping():
    return cluster_mapping
