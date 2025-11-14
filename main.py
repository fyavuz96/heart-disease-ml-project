from src.preprocess import load_data, split_data
from src.model import create_pipeline_model, create_polynomial_model
from src.pipeline import build_pipeline, evaluate_model

import os
import pandas as pd

# Çalışma dizinine göre data klasörünü oluştur
file_path = os.path.join(os.getcwd(), "data", "heart.csv")

df=load_data(file_path)

X_train, X_test, y_train, y_test=split_data(df,"target")

poly_model=create_polynomial_model(degree=2)
ridge_model=create_pipeline_model(alpha=0.1, degree=2)

def run_experiment(model, name):
    pipe = build_pipeline(model)
    scores = evaluate_model(pipe, X_train, y_train)

    print(f"Cross-validated score: {scores:.4f}")
    print("CV Scores:", scores)
    print("Mean Score:",scores.mean())
    print("-----")

run_experiment(poly_model,"Polynomial Regression ")
run_experiment(ridge_model,"Ridge Regression")
