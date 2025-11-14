from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.pipeline import Pipeline

def create_polynomial_model(degree=2):
    return Pipeline([
        ('poly_features', PolynomialFeatures(degree=degree, include_bias=False)),
        ('ridge_regression', LinearRegression())
    ])

def create_pipeline_model(alpha=0.1,degree=2):
    return Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('ridge', Ridge(alpha=alpha))
    ])