"""
GridSearchCV per Logistic Regression
======================================
Script completo che mostra come:
- preparare i dati (split + scaling)
- definire una griglia di iperparametri valida (con vincoli solver/penalty)
- eseguire GridSearchCV con cross-validation
- analizzare i risultati (best params, cv_results_, heatmap)
- valutare il modello finale sul test set
"""
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
 
import joblib 
from training.save_metadata import save_metadata

 
# ---------------------------------------------------------------------------
# 1. Caricamento e split dei dati
# ---------------------------------------------------------------------------
PRJ_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
df = pd.read_csv(PRJ_ROOT_DIR / "data/clean/telco_customer_churn_clean.csv")

y = df["Churn"]
X = df.drop(columns=["Churn"])

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y,test_size=0.2,stratify=y,random_state=42
    )

 
 
# ---------------------------------------------------------------------------
# 2. Pipeline: scaling + modello
# ---------------------------------------------------------------------------
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
 
 
# ---------------------------------------------------------------------------
# 3. Griglia di iperparametri
# ---------------------------------------------------------------------------

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
 
 
# ---------------------------------------------------------------------------
# 4. GridSearchCV
# ---------------------------------------------------------------------------
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
 
grid_search.fit(X_train, y_train)
 
 
# ---------------------------------------------------------------------------
# 5. Risultati della ricerca
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
# 6. Valutazione sul test set con il modello migliore
# ---------------------------------------------------------------------------
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
 
print("\n" + "=" * 60)
print("VALUTAZIONE SUL TEST SET")
print("=" * 60)
print(f"Accuracy sul test set: {best_model.score(X_test, y_test):.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred))
 
 
# ---------------------------------------------------------------------------
# 7. Visualizzazioni
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Confusion Matrix")
 
# Heatmap C vs score, filtrata sul miglior solver/l1_ratio trovato
best_solver = grid_search.best_params_["clf__solver"]
best_l1_ratio = grid_search.best_params_["clf__l1_ratio"]
 
mask = results_df["params"].apply(
    lambda p: p.get("clf__solver") == best_solver and p.get("clf__l1_ratio") == best_l1_ratio
)
subset = results_df[mask].copy()
subset["C"] = subset["params"].apply(lambda p: p["clf__C"])
subset = subset.sort_values("C")
 
axes[1].plot(subset["C"], subset["mean_test_score"], marker="o")
axes[1].fill_between(
    subset["C"],
    subset["mean_test_score"] - subset["std_test_score"],
    subset["mean_test_score"] + subset["std_test_score"],
    alpha=0.2
)
axes[1].set_xscale("log")
axes[1].set_xlabel("C (scala log)")
axes[1].set_ylabel("Accuracy media (CV)")
axes[1].set_title(f"Score vs C (solver={best_solver}, l1_ratio={best_l1_ratio})")
axes[1].grid(alpha=0.3)
 
plt.tight_layout()
plt.show()
 
 
# ---------------------------------------------------------------------------
# 8. Numero totale di fit eseguiti (utile per stimare i tempi)
# ---------------------------------------------------------------------------

n_combinations = sum(
    np.prod([len(v) for v in grid.values()]) for grid in param_grid
)
n_folds = cv_strategy.get_n_splits()
print(f"\nCombinazioni totali testate: {int(n_combinations)}")
print(f"Fold di cross-validation: {n_folds}")
print(f"Totale fit eseguiti: {int(n_combinations) * n_folds}")


# ---------------------------------------------------------------------------
# 9. Esporto il best model 
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # Salva su disco la pipeline addestrata (preprocessing + modello) e il
    # label encoder, in modo da poterli ricaricare da un altro script senza
    # dover rieseguire il training da capo.
    # Creo la nuova directory se non esiste, altrimenti continuo
    Path(PRJ_ROOT_DIR / "models/logisticRegression_best_params").mkdir(parents = True, exist_ok = True)

    pipeline_path = PRJ_ROOT_DIR / "models/logisticRegression_best_params/churn_pipeline_logisticRegression_best_params.joblib"
    label_encoder_path = PRJ_ROOT_DIR / "models/logisticRegression_best_params/churn_label_encoder_logisticRegression_best_params.joblib"
    features_path = PRJ_ROOT_DIR / "models/logisticRegression_best_params/churn_feature_columns_logisticRegression_best_params.joblib"

    joblib.dump(best_model, pipeline_path)
    joblib.dump(label_encoder, label_encoder_path)
    joblib.dump(list(X.columns), features_path)


    # Aggiorno il json dei modelli disponibili
    model_metadata = {
            "name": "logisticRegression_best_params", 
            "desc": "logistic Regression (migliori parametri ottenuti dalla gridSearch)",
            "pipeline_path": str(pipeline_path.relative_to(PRJ_ROOT_DIR)),
            "label_encoder_path": str(label_encoder_path.relative_to(PRJ_ROOT_DIR)),
            "features_path": str(features_path.relative_to(PRJ_ROOT_DIR))
        }

    save_metadata(model_metadata, PRJ_ROOT_DIR / "models/metadata.json")