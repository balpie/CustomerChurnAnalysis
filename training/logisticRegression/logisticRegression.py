import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from pathlib import Path
from sklearn.model_selection import  GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
 
import joblib 
from training.save_metadata import save_metadata

 

PRJ_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
df = pd.read_csv(PRJ_ROOT_DIR / "data/clean/training_set.csv")

y = df["Churn"]
X = df.drop(columns=["Churn"])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
  

categorical_cols = X.select_dtypes(include=["object", "str","category"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object", "str", "category"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ]
)


pipe = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=5000, random_state=42))
])
 
 

param_grid = [
    {
        "clf__solver": ["liblinear"],
        "clf__l1_ratio": [0.0, 1.0],          # L2 e L1
        "clf__C": np.logspace(-4, 4, 9)
    },
    {
        "clf__solver": ["lbfgs"],
        "clf__l1_ratio": [0.0],               # solo L2
        "clf__C": np.logspace(-4, 4, 9)
    },
    {
        "clf__solver": ["saga"],
        "clf__l1_ratio": [0.0, 0.3, 0.5, 0.7, 1.0],  # L2, elasticnet, L1
        "clf__C": np.logspace(-4, 4, 9)
    },
]
 
 
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
 
grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv_strategy,
    scoring="f1",     
    n_jobs=-1,
    verbose=1,
    refit=True
)
 
grid_search.fit(X, y)
 
 
# ---------------------------------------------------------------------------
# Risultati della ricerca
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("MIGLIORI IPERPARAMETRI")
print("=" * 60)
print(grid_search.best_params_)
print(f"\nMiglior score medio (CV): {grid_search.best_score_:.4f}")
 
# Tabella con tutti i risultati, ordinata per rank
results_df = pd.DataFrame(grid_search.cv_results_)
cols_to_show = ["params", "mean_test_score", "std_test_score", "rank_test_score"]
top_results = results_df[cols_to_show].sort_values("rank_test_score").head(10)
 
print("\nTop 10 combinazioni:")
print(top_results.to_string(index=False))
 




# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # Salva su disco la pipeline addestrata (preprocessing + modello) e il
    # label encoder, in modo da poterli ricaricare da un altro script senza
    # dover rieseguire il training da capo.
    # Creo la nuova directory se non esiste, altrimenti continuo
    Path(PRJ_ROOT_DIR / "models/logisticRegression").mkdir(parents = True, exist_ok = True)

    pipeline_path = PRJ_ROOT_DIR / "models/logisticRegression/churn_pipeline_logisticRegression.joblib"
    label_encoder_path = PRJ_ROOT_DIR / "models/logisticRegression/churn_label_encoder_logisticRegression.joblib"
    features_path = PRJ_ROOT_DIR / "models/logisticRegression/churn_feature_columns_logisticRegression.joblib"
    risultati_grid_search = PRJ_ROOT_DIR / "models/logisticRegression/risultati_grid_search.joblib"

    joblib.dump(grid_search.best_estimator_, pipeline_path)
    joblib.dump(label_encoder, label_encoder_path)
    joblib.dump(list(X.columns), features_path)
    joblib.dump(grid_search.cv_results_,risultati_grid_search)


    # Aggiorno il json dei modelli disponibili
    model_metadata = {
            "name": "LogisticRegression", 
            "desc": "Logistic Regression",
            "pipeline_path": str(pipeline_path.relative_to(PRJ_ROOT_DIR)),
            "label_encoder_path": str(label_encoder_path.relative_to(PRJ_ROOT_DIR)),
            "features_path": str(features_path.relative_to(PRJ_ROOT_DIR))
        }

    save_metadata(model_metadata, PRJ_ROOT_DIR / "models/metadata.json")
