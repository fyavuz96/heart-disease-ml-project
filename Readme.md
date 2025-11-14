# Heart Disease Prediction Project (Sklearn + Polynomial + Ridge + CV)

Aşağıda tam bir GitHub projesinin genel yapısı, klasör düzeni, açıklamalar ve ana Python dosyaları yer alıyor. Bunu direkt GitHub reposu olarak kullanabilirsin.

## Proje Yapısı

```
heart-disease-ml-project/
│
├── README.md
├── requirements.txt
├── data/
│   └── heart.csv
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── preprocess.py
│   ├── model.py
│   └── pipeline.py
└── main.py
```

---

# README.md

## Heart Disease Prediction Project

Bu proje, Heart Disease veri seti kullanılarak lineer regresyon, polynomial regresyon, Ridge regresyon ve cross-validation uygulamalarını gösterir. Amaç tam bir makine öğrenimi pipeline'ı kurmak ve iyi dokümante edilmiş bir portföy çalışması sunmaktır.

### Özellikler

* Polynomial Regression
* Ridge Regression
* Scikit-learn Pipeline kullanımı
* StandardScaler ile ölçekleme
* Cross-Validation (KFold)
* R² değerlendirmesi

### Veri Seti

```
data/heart.csv
```

Kaggle'daki Cleveland Heart Disease dataset kullanılabilir.

### Kurulum

```
pip install -r requirements.txt
python main.py
```

### Gereksinimler

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

# requirements.txt

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

---

# src/preprocess.py

```python
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path="data/heart.csv"):
    df = pd.read_csv(path)
    return df

def split_data(df, target="target"):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)
```

---

# src/model.py

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

def create_polynomial_model(degree=2):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("linear", LinearRegression())
    ])

def create_ridge_model(alpha=1.0, degree=2):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("ridge", Ridge(alpha=alpha))
    ])
```

---

# src/pipeline.py

```python
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

def build_pipeline(model):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

def evaluate_model(pipeline, X, y, cv=5):
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")
    return scores
```

---

# main.py

```python
from src.preprocess import load_data, split_data
from src.model import create_polynomial_model, create_ridge_model
from src.pipeline import build_pipeline, evaluate_model

# Veri yükleme
df = load_data()
X_train, X_test, y_train, y_test = split_data(df)

# Modeller
poly_model = create_polynomial_model(degree=2)
ridge_model = create_ridge_model(alpha=1.0, degree=2)

# Pipeline
def run_experiment(model, name):
    pipe = build_pipeline(model)
    scores = evaluate_model(pipe, X_train, y_train)
    print(f"Model: {name}")
    print("CV Scores:", scores)
    print("Mean Score:", scores.mean())
    print("---")

run_experiment(poly_model, "Polynomial Regression")
run_experiment(ridge_model, "Ridge Regression")
```
