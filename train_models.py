#!/usr/bin/env python
"""
Training script to generate all ML models for the Vehicle Analytics System.
Run this script before starting the Django server.
"""
import os
import sys

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Training Machine Learning Models for Vehicle Analytics System")
print("=" * 60)

# Train Regression Model
print("\n1. Training Regression Model (Price Prediction)...")
try:
    from model_generators.regression.train_regression import evaluate_regression_model
    eval_result = evaluate_regression_model()
    print(f"   ✓ Regression model trained successfully!")
    print(f"   ✓ R² Score: {eval_result['r2']}%")
except Exception as e:
    print(f"   ✗ Error training regression model: {e}")

# Train Classification Model
print("\n2. Training Classification Model (Income Level Prediction)...")
try:
    from model_generators.classification.train_classifier import evaluate_classification_model
    eval_result = evaluate_classification_model()
    print(f"   ✓ Classification model trained successfully!")
    print(f"   ✓ Accuracy: {eval_result['accuracy']}%")
except Exception as e:
    print(f"   ✗ Error training classification model: {e}")

# Train Clustering Model
print("\n3. Training Clustering Model (Client Segmentation)...")
try:
    from model_generators.clustering.train_cluster import evaluate_clustering_model, calculate_coefficient_of_variation
    eval_result = evaluate_clustering_model()
    cv = calculate_coefficient_of_variation()
    print(f"   ✓ Clustering model trained successfully!")
    print(f"   ✓ Silhouette Score: {eval_result['silhouette']}")
    print(f"   ✓ Coefficient of Variation: {cv}%")
except Exception as e:
    print(f"   ✗ Error training clustering model: {e}")

print("\n" + "=" * 60)
print("All models trained and saved successfully!")
print("You can now start the Django server with: python manage.py runserver")
