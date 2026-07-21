from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path
from training.save_metadata import save_metadata
import pandas as pd
import joblib 

PRJ_ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Separa target dalle feature
df = pd.read_csv(PRJ_ROOT_DIR / "data/clean/training_set.csv")
y = df["Churn"]
X = df.drop(columns=["Churn"])

# Encoding del target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Individua automaticamente le colonne categoriche e numeriche
categorical_cols = X.select_dtypes(include=["object", "str", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object", "str", "category"]).columns.tolist()


# Preprocessing: le categoriche vengono codificate con OneHotEncoder,
# le numeriche vengono standardizzate. In questo modo l'encoding
# viene "imparato" sul train e riapplicato in modo coerente a qualsiasi
# nuovo record passato in seguito (anche con categorie non viste).
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols),
    ]
)

# Pipeline unica: preprocessing + modello.
# Allenandola su X_train "grezzo" (senza get_dummies fatto a mano),
# la stessa pipeline può poi essere riutilizzata per predire su nuovi record.
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", SGDClassifier(max_iter=1000, tol=1e-4, random_state=42)),
    ]
)

param_grid = [
    {
        "classifier__loss": ["hinge", "log_loss", "modified_huber"],
        "classifier__penalty": ["l2", "l1"],
        "classifier__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "classifier__class_weight": ["balanced", None]
    },
    {
        "classifier__loss": ["hinge", "log_loss", "modified_huber"],
        "classifier__penalty": ["elasticnet"],
        "classifier__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
        "classifier__l1_ratio": [0.15, 0.5, 0.85],
        "classifier__class_weight": ["balanced", None]
    },
]

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        scoring="f1", 
        n_jobs=None, 
        refit=True, 
        cv=cv_strategy, 
        verbose=1)

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


if __name__ == "__main__":

    # Salva su disco la pipeline addestrata (preprocessing + modello) e il
    # label encoder, in modo da poterli ricaricare da un altro script senza
    # dover rieseguire il training da capo.
    # Creo la nuova directory se non esiste, altrimenti continuo
    Path(PRJ_ROOT_DIR / "models/sgdc").mkdir(parents = True, exist_ok = True)

    pipeline_path = PRJ_ROOT_DIR / "models/sgdc/churn_pipeline_sgdc.joblib"
    label_encoder_path = PRJ_ROOT_DIR / "models/sgdc/churn_label_encoder_sgdc.joblib"
    features_path = PRJ_ROOT_DIR / "models/sgdc/churn_feature_columns_sgdc.joblib"
    risultati_grid_search = PRJ_ROOT_DIR / "models/sgdc/risultati_grid_search.joblib"


    joblib.dump(grid_search.best_estimator_, pipeline_path)
    joblib.dump(label_encoder, label_encoder_path)
    joblib.dump(list(X.columns), features_path)
    joblib.dump(grid_search.cv_results_,risultati_grid_search)


    # Aggiorno il json dei modelli disponibili
    model_metadata = {
            "name": "sgdc", 
            "desc": "Stochastic Gradient Descent Classifier",
            "pipeline_path": str(pipeline_path.relative_to(PRJ_ROOT_DIR)),
            "label_encoder_path": str(label_encoder_path.relative_to(PRJ_ROOT_DIR)),
            "features_path": str(features_path.relative_to(PRJ_ROOT_DIR))
        }

    save_metadata(model_metadata, PRJ_ROOT_DIR / "models/metadata.json")
