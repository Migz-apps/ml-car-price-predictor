from django.test import TestCase, Client
from django.urls import reverse
import pandas as pd
import os


class PredictorViewsTest(TestCase):
    def setUp(self):
        self.client = Client()
        
    def test_data_exploration_view(self):
        """Test that data exploration page loads correctly."""
        response = self.client.get(reverse('data_exploration'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Vehicle Insights Dashboard")
        
    def test_regression_analysis_get(self):
        """Test that regression analysis page loads correctly."""
        response = self.client.get(reverse('regression_analysis'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Regression Analysis")
        
    def test_classification_analysis_get(self):
        """Test that classification analysis page loads correctly."""
        response = self.client.get(reverse('classification_analysis'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Classification")
        
    def test_clustering_analysis_get(self):
        """Test that clustering analysis page loads correctly."""
        response = self.client.get(reverse('clustering_analysis'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Clustering")


class DataExplorationTest(TestCase):
    def test_dataset_exists(self):
        """Test that the vehicle dataset exists."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, "vehicles_ml_dataset.csv")
        self.assertTrue(os.path.exists(data_path))
        
    def test_dataset_columns(self):
        """Test that dataset has required columns."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_path = os.path.join(base_dir, "vehicles_ml_dataset.csv")
        df = pd.read_csv(data_path)
        
        required_columns = [
            'year', 'kilometers_driven', 'seating_capacity', 
            'estimated_income', 'selling_price', 'income_level', 'client_name'
        ]
        
        for col in required_columns:
            self.assertIn(col, df.columns)


class ModelTrainingTest(TestCase):
    def test_regression_model_trained(self):
        """Test that regression model can be evaluated."""
        from model_generators.regression.train_regression import evaluate_regression_model
        result = evaluate_regression_model()
        self.assertIn('r2', result)
        self.assertIn('comparison', result)
        
    def test_classification_model_trained(self):
        """Test that classification model can be evaluated."""
        from model_generators.classification.train_classifier import evaluate_classification_model
        result = evaluate_classification_model()
        self.assertIn('accuracy', result)
        self.assertIn('comparison', result)
        
    def test_clustering_model_trained(self):
        """Test that clustering model can be evaluated."""
        from model_generators.clustering.train_cluster import evaluate_clustering_model, calculate_coefficient_of_variation
        result = evaluate_clustering_model()
        self.assertIn('silhouette', result)
        self.assertIn('summary', result)
        self.assertIn('comparison', result)
        
        # Test CV calculation
        cv = calculate_coefficient_of_variation()
        self.assertIsInstance(cv, (int, float))
