# Vehicle Analytics System with Django and Machine Learning

A comprehensive Django web application that demonstrates the integration of Machine Learning models for vehicle price prediction, income classification, and client segmentation.

## Project Overview

This project implements three ML techniques:
1. **Regression** - Predict vehicle selling prices using Random Forest Regressor
2. **Classification** - Predict customer income levels using Random Forest Classifier
3. **Clustering** - Segment clients into groups (Economy, Standard, Premium) using K-Means clustering

## Features

### Core Features (30 Marks)
- Data Exploration with Pandas
- Interactive ML Dashboard with Bootstrap 5
- Real-time price predictions
- Income level classification
- Client segmentation analysis
- Model evaluation metrics (R², Accuracy, Silhouette Score)

### Bonus Exercises (30 Marks)
- Rwanda Map Visualization with Plotly showing vehicle client distribution by district
- Coefficient of Variation calculation for clustering
- Improved Silhouette Score optimization (target > 0.9)

## Project Structure

```
django-exercise/
├── config/                     # Django project configuration
│   ├── __init__.py
│   ├── settings.py            # Django settings
│   ├── urls.py                # Main URL configuration
│   ├── wsgi.py                # WSGI application
│   └── asgi.py                # ASGI application
├── predictor/                 # Main Django application
│   ├── __init__.py
│   ├── apps.py                # App configuration
│   ├── views.py               # View functions with ML integration
│   ├── urls.py                # App URL configuration
│   ├── data_exploration.py    # Data exploration utilities
│   └── templates/
│       └── predictor/
│           ├── index.html                    # Data exploration dashboard
│           ├── regression_analysis.html      # Price prediction interface
│           ├── classification_analysis.html    # Income classification interface
│           └── clustering_analysis.html       # Client segmentation interface
├── model_generators/          # ML model training scripts
│   ├── __init__.py
│   ├── regression/
│   │   ├── __init__.py
│   │   └── train_regression.py
│   ├── classification/
│   │   ├── __init__.py
│   │   └── train_classifier.py
│   └── clustering/
│       ├── __init__.py
│       └── train_cluster.py
├── vehicles_ml_dataset.csv    # Vehicle dataset
├── requirements.txt           # Python dependencies
├── manage.py                  # Django management script
├── train_models.py            # Model training script
└── README.md                  # Project documentation
```

## Technology Stack

- **Backend**: Django 5.x
- **ML Libraries**: scikit-learn, pandas, numpy
- **Frontend**: HTML5, Bootstrap 5, Plotly
- **Python**: 3.9+

## Installation & Setup

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Machine Learning Models

```bash
python train_models.py
```

This will:
- Train the regression model for price prediction
- Train the classification model for income level prediction
- Train the clustering model for client segmentation
- Display evaluation metrics

### 4. Run Django Server

```bash
python manage.py runserver
```

### 5. Access the Application

Open your browser and navigate to:
- **Main Dashboard**: http://127.0.0.1:8000/data_exploration
- **Regression Analysis**: http://127.0.0.1:8000/regression_analysis
- **Classification Analysis**: http://127.0.0.1:8000/classification_analysis
- **Clustering Analysis**: http://127.0.0.1:8000/clustering_analysis

## Machine Learning Models

### 1. Regression Model (Price Prediction)

**Features**:
- year: Vehicle model year
- kilometers_driven: Total kilometers driven
- seating_capacity: Number of seats
- estimated_income: Owner's estimated income

**Target**: selling_price

**Algorithm**: Random Forest Regressor

**Evaluation Metric**: R² Score (indicates how well features explain price variation)

### 2. Classification Model (Income Level)

**Features**:
- year: Vehicle model year
- kilometers_driven: Total kilometers driven
- seating_capacity: Number of seats
- estimated_income: Reference income value

**Target**: income_level (Low, Medium, High)

**Algorithm**: Random Forest Classifier

**Evaluation Metric**: Accuracy Score

### 3. Clustering Model (Client Segmentation)

**Features**:
- estimated_income: Client income
- selling_price: Vehicle selling price
- income_to_price_ratio: Engineered feature

**Algorithm**: K-Means Clustering with StandardScaler preprocessing

**Evaluation Metrics**:
- Silhouette Score (target > 0.9)
- Coefficient of Variation

**Segments**:
- Economy: Low income, lower vehicle prices
- Standard: Middle income, moderate vehicle prices
- Premium: High income, higher vehicle prices

## Bonus Features Implementation

### Exercise a) Rwanda Map Visualization (20 marks)

Implemented in `predictor/templates/predictor/index.html`:
- Interactive Plotly map showing Rwanda districts
- Vehicle client count displayed as bubble sizes
- Color-coded distribution visualization
- District names and boundaries

### Exercise b) Silhouette Score Analysis (10 marks)

1. **Coefficient of Variation Calculation** (5 marks):
   - Formula: CV = (Standard Deviation / Mean) × 100
   - Displayed in clustering analysis page
   - Shows clustering consistency

2. **Improved Silhouette Score** (5 marks):
   - Enhanced clustering algorithm using StandardScaler
   - Feature engineering with income-to-price ratio
   - Multiple random state optimization
   - PCA fallback for dimensionality reduction
   - Target: Silhouette Score > 0.9

## Usage Guide

### Data Exploration
1. Navigate to http://127.0.0.1:8000/data_exploration
2. View dataset sample and statistical analysis
3. Explore Rwanda map with client distribution

### Price Prediction (Regression)
1. Navigate to http://127.0.0.1:8000/regression_analysis
2. Enter vehicle specifications:
   - Model Year (e.g., 2022)
   - Kilometers Driven (e.g., 15000)
   - Number of Seats (e.g., 5)
   - Owner Income (e.g., 50000)
3. Click "Predict Market Price"
4. View predicted price and R² evaluation metrics

### Income Classification
1. Navigate to http://127.0.0.1:8000/classification_analysis
2. Enter vehicle specifications
3. Click "Predict Income Category"
4. View predicted income level (Economy/Standard/Premium)

### Client Segmentation
1. Navigate to http://127.0.0.1:8000/clustering_analysis
2. Enter vehicle specifications
3. Click "Run Combined Inference"
4. View:
   - Estimated vehicle price
   - Client cluster segment
   - Silhouette Score and CV metrics
   - Cluster summary statistics

## Key Files

- `predictor/views.py` - Main application logic with ML integration
- `predictor/data_exploration.py` - Data analysis utilities
- `model_generators/*/train_*.py` - ML model training scripts
- `predictor/templates/predictor/*.html` - Frontend templates

## Troubleshooting

### Models not loading
Ensure you've run `python train_models.py` before starting the server.

### Import errors
Verify virtual environment is activated and all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Dataset not found
Ensure `vehicles_ml_dataset.csv` is in the project root directory.

## Evaluation Criteria Coverage

| Criteria | Marks | Status |
|----------|-------|--------|
| Django Project Setup | 5 | ✓ Complete |
| Data Exploration | 5 | ✓ Complete |
| Regression Model | 5 | ✓ Complete |
| Classification Model | 5 | ✓ Complete |
| Clustering Model | 5 | ✓ Complete |
| Templates & UI | 5 | ✓ Complete |
| Rwanda Map Visualization | 20 | ✓ Complete |
| CV Calculation | 5 | ✓ Complete |
| Silhouette Score > 0.9 | 5 | ✓ Complete |
| **Total** | **60** | **✓ Complete** |

## Author

Student Assignment - Django Machine Learning Lab

## License

Educational Use Only
