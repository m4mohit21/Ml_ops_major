# iris_experiment.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.exceptions import NotFittedError
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

# Data processing class
class IrisDataProcessor:
    def __init__(self, test_size=0.2, random_state=42):
        iris = load_iris()
        self.data = pd.DataFrame(iris.data, columns=iris.feature_names)
        self.target = pd.Series(iris.target, name="species")
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.experiment_log = []

    def prepare_data(self):
        self.data[self.data.columns] = self.scaler.fit_transform(self.data)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data, self.target, test_size=self.test_size, random_state=self.random_state
        )
        experiment = {
            "test_size": self.test_size,
            "random_state": self.random_state,
            "X_train_shape": self.X_train.shape,
            "X_test_shape": self.X_test.shape,
        }
        self.experiment_log.append(experiment)

    def get_feature_stats(self):
        return self.data.describe()

# Experiment tracking and model training class
class IrisExperiment:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.models = {
            "LogisticRegression": LogisticRegression(max_iter=200),
            "RandomForest": RandomForestClassifier(n_estimators=100)
        }
        self.results = {}

    def run_experiment(self):
        self.data_processor.prepare_data()
        X_train, X_test = self.data_processor.X_train, self.data_processor.X_test
        y_train, y_test = self.data_processor.y_train, self.data_processor.y_test
        
        mlflow.set_experiment("Iris Classification Experiment")
        
        for model_name, model in self.models.items():
            with mlflow.start_run(run_name=model_name):
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="weighted")
                recall = recall_score(y_test, y_pred, average="weighted")
                
                cv_accuracy = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
                cv_precision = cross_val_score(model, X_train, y_train, cv=5, scoring="precision_weighted").mean()
                cv_recall = cross_val_score(model, X_train, y_train, cv=5, scoring="recall_weighted").mean()
                
                self.results[model_name] = {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "cv_accuracy": cv_accuracy,
                    "cv_precision": cv_precision,
                    "cv_recall": cv_recall
                }
                
                self.log_results(model_name, model, accuracy, precision, recall, cv_accuracy, cv_precision, cv_recall)

    def log_results(self, model_name, model, accuracy, precision, recall, cv_accuracy, cv_precision, cv_recall):
        mlflow.log_param("Model", model_name)
        mlflow.log_metric("Test Accuracy", accuracy)
        mlflow.log_metric("Test Precision", precision)
        mlflow.log_metric("Test Recall", recall)
        mlflow.log_metric("CV Accuracy", cv_accuracy)
        mlflow.log_metric("CV Precision", cv_precision)
        mlflow.log_metric("CV Recall", cv_recall)
        
        mlflow.sklearn.log_model(model, model_name)

# Model optimization and testing class
class IrisModelOptimizer:
    def __init__(self, experiment):
        self.experiment = experiment
        self.quantized_model = None
        self.original_model = self.experiment.models["LogisticRegression"]
    
    def quantize_model(self):
        if not hasattr(self.original_model, "coef_"):
            raise NotFittedError("The Logistic Regression model must be trained before quantization.")
        
        self.quantized_model = LogisticRegression(max_iter=self.original_model.max_iter)
        self.quantized_model.coef_ = self.original_model.coef_.astype(np.float16)
        self.quantized_model.intercept_ = self.original_model.intercept_.astype(np.float16)
        self.quantized_model.classes_ = self.original_model.classes_
        
        print("Quantization complete. Model coefficients converted to float16.")
    
  # Add at the top of the file

# Inside the IrisModelOptimizer class
    def run_tests(self):
        try:
            # Ensure quantized model exists and has float16 attributes
            assert self.quantized_model is not None, "Quantized model not created."
            assert self.quantized_model.coef_.dtype == np.float16, "Model quantization failed: coefficients are not in float16."
            assert self.quantized_model.intercept_.dtype == np.float16, "Model quantization failed: intercepts are not in float16."
            print("Quantization test passed.")
        except AssertionError as e:
            print(f"Quantization test failed: {e}")

        # Preparing data for testing
        X_test, y_test = self.experiment.data_processor.X_test, self.experiment.data_processor.y_test

        # Prediction with original model
        original_pred = self.original_model.predict(X_test)
        original_accuracy = accuracy_score(y_test, original_pred)

        # Prediction with quantized model
        quantized_pred_proba = np.dot(X_test, self.quantized_model.coef_.T) + self.quantized_model.intercept_
        quantized_pred = quantized_pred_proba.argmax(axis=1)  # Get the index of the max score as class prediction
        
        # Map quantized predictions to class labels
        label_encoder = LabelEncoder()
        label_encoder.fit(y_test)  # Fit encoder on true labels
        quantized_pred_labels = label_encoder.inverse_transform(quantized_pred)

        quantized_accuracy = accuracy_score(y_test, quantized_pred_labels)
        
        print(f"Original model accuracy: {original_accuracy:.2f}")
        print(f"Quantized model accuracy: {quantized_accuracy:.2f}")

        try:
            assert abs(original_accuracy - quantized_accuracy) < 0.05, "Quantized model accuracy differs significantly from the original."
            print("Accuracy test passed.")
        except AssertionError as e:
            print(f"Accuracy test failed: {e}")

# Main function to run experiment and optimizer
def main():
    # Initialize processor
    processor = IrisDataProcessor()
    processor.prepare_data()
    
    # Run experiments
    experiment = IrisExperiment(data_processor=processor)
    experiment.run_experiment()
    print("Experiment Results:\n", experiment.results)
    
    # Optimize and test model
    optimizer = IrisModelOptimizer(experiment=experiment)
    optimizer.quantize_model()
    optimizer.run_tests()

if __name__ == "__main__":
    main()
