from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score


def build_pipeline(model):
    """
    Build a machine learning pipeline with data scaling and the provided model.

    Parameters:
    model: The machine learning model to be included in the pipeline.

    Returns:
    A sklearn Pipeline object.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    return pipeline


def evaluate_model(pipeline, X, y, cv=5):
    """
    Evaluate the provided pipeline using cross-validation.

    Parameters:
    pipeline: The sklearn Pipeline object to be evaluated.
    X: Features dataset.
    y: Target variable.
    cv: Number of cross-validation folds.

    Returns:
    Mean cross-validation score.
    """
    scores = cross_val_score(pipeline, X, y, cv=cv)
    return scores.mean()

