import pandas as pd
import joblib
import os
from django.shortcuts import render
from predictor.data_exploration import dataset_exploration, data_exploration

# Import evaluation functions
from model_generators.clustering.train_cluster import (
    evaluate_clustering_model,
    get_cluster_mapping,
)
from model_generators.classification.train_classifier import evaluate_classification_model
from model_generators.regression.train_regression import evaluate_regression_model

# Get base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "vehicles_ml_dataset.csv")

# Load models once with error handling
try:
    regression_model = joblib.load(
        os.path.join(BASE_DIR, "model_generators", "regression", "regression_model.pkl")
    )
except:
    regression_model = None

try:
    classification_model = joblib.load(
        os.path.join(BASE_DIR, "model_generators", "classification", "classification_model.pkl")
    )
except:
    classification_model = None

try:
    clustering_model = joblib.load(
        os.path.join(BASE_DIR, "model_generators", "clustering", "clustering_model.pkl")
    )
except:
    clustering_model = None


def data_exploration_view(request):
    df = pd.read_csv(DATA_PATH)
    
    # Calculate real district data from CSV
    district_counts = df['district'].value_counts().to_dict()
    
    # Rwanda district coordinates (lat, lon)
    district_coords = {
        'Gasabo': (-1.9441, 30.0619),
        'Kicukiro': (-2.0000, 30.1000),
        'Nyarugenge': (-1.9500, 30.0500),
        'Musanze': (-1.5000, 29.6000),
        'Rubavu': (-1.6700, 29.3500),
        'Huye': (-2.6000, 29.7333),
        'Rwamagana': (-1.9500, 30.4333),
        'Muhanga': (-2.0833, 29.7500),
        'Kayonza': (-1.9333, 30.5000),
        'Bugesera': (-2.2333, 30.1667),
        'Gicumbi': (-1.6667, 29.8500),
        'Nyanza': (-2.3500, 29.7500),
        'Ruhango': (-2.2333, 29.8000),
        'Nyagatare': (-1.3000, 30.3333),
        'Gakenke': (-1.7000, 29.6333),
        'Kirehe': (-2.2500, 30.6500),
        'Ngoma': (-2.1667, 30.5500),
        'Nyaruguru': (-2.6000, 29.6333),
        'Rusizi': (-2.4833, 28.9000),
        'Rutsiro': (-1.8167, 29.2833),
        'Burera': (-1.4667, 29.8333),
        'Gatsibo': (-1.5833, 30.4333),
        'Kamonyi': (-2.0667, 29.8833),
        'Karongi': (-2.1667, 29.3667),
        'Nyabihu': (-1.6333, 29.4333),
        'Nyamasheke': (-2.5000, 29.1667),
        'Rulindo': (-1.7500, 29.9833),
        'Ngororero': (-1.8333, 29.6500),
        'Gisagara': (-2.5833, 29.8333),
        'Nyamagabe': (-2.3333, 29.6333),
    }
    
    # Build district data list
    district_data = []
    for district, count in district_counts.items():
        if district in district_coords:
            lat, lon = district_coords[district]
            district_data.append({
                'name': district,
                'lat': lat,
                'lon': lon,
                'clients': int(count)
            })
    
    context = {
        "data_exploration": data_exploration(df),
        "dataset_exploration": dataset_exploration(df),
        "district_data": district_data,
    }
    return render(request, "predictor/index.html", context)


def regression_analysis(request):
    context = {"evaluations": evaluate_regression_model()}
    if request.method == "POST":
        year = int(request.POST["year"])
        km = float(request.POST["km"])
        seats = int(request.POST["seats"])
        income = float(request.POST["income"])
        
        if regression_model:
            prediction = regression_model.predict([[year, km, seats, income]])[0]
            context["price"] = prediction
        else:
            context["error"] = "Regression model not loaded"
    
    return render(request, "predictor/regression_analysis.html", context)


def classification_analysis(request):
    context = {"evaluations": evaluate_classification_model()}
    if request.method == "POST":
        year = int(request.POST["year"])
        km = float(request.POST["km"])
        seats = int(request.POST["seats"])
        income = float(request.POST["income"])
        
        if classification_model:
            prediction = classification_model.predict([[year, km, seats, income]])[0]
            context["prediction"] = prediction
        else:
            context["error"] = "Classification model not loaded"
    
    return render(request, "predictor/classification_analysis.html", context)


def clustering_analysis(request):
    eval_results = evaluate_clustering_model()
    context = {"evaluations": eval_results}
    
    if request.method == "POST":
        try:
            year = int(request.POST["year"])
            km = float(request.POST["km"])
            seats = int(request.POST["seats"])
            income = float(request.POST["income"])
            
            if regression_model and clustering_model:
                # Step 1: Predict price
                predicted_price = regression_model.predict([[year, km, seats, income]])[0]
                
                # Step 2: Predict cluster
                cluster_id = clustering_model.predict([[income, predicted_price]])[0]
                
                mapping = get_cluster_mapping()
                
                context.update({
                    "prediction": mapping.get(cluster_id, "Unknown"),
                    "price": predicted_price
                })
            else:
                context["error"] = "Models not loaded"
                
        except Exception as e:
            context["error"] = str(e)
    
    return render(request, "predictor/clustering_analysis.html", context)
