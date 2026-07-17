from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from math import isqrt 
from training.save_metadata import save_metadata
import joblib 

PRJ_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
df = pd.read_csv(PRJ_ROOT_DIR / "data/clean/training_set.csv")

# Separa target dalle feature
y = df["Churn"]
X = df.drop(columns=["Churn"])

# Encoding del target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

categorical_cols = X.select_dtypes(include=["object", "str","category"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object", "str", "category"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(), categorical_cols), #ho rimosso handle_unknown="ignore" per provare
        ("num", StandardScaler(), numeric_cols),
    ]
)


pipe = Pipeline(
    steps=[
        ("preprocessor",preprocessor),
        ("knn",KNeighborsClassifier())
    ]
)

k = isqrt(len(X))
#lista_n_neighbors = list(range(k - 20, k - 10)) + list(range(k + 10, k + 20))

lista_n_neighbors = list(range(1,k + 1)) #best = 37 
#lista_n_neighbors = list(range(k, 2*k))  #best = 87
best = 37
lista_n_neighbors = list(range(best,k + 1, 3))

#a partire dal valore k, facendo diverse iterazioni su un intervallo totale di [1, 2*k],
#siamo arrivati alla conclusione che 37 è il miglior numero per questo problema tenendo conto delle distanze euclidean e manhattan
# e in un intervallo [1, k + 1] con la metrica minkowski utilizzando p = [1.5, 3, 4]
param_grid = [
    {
        'knn__n_neighbors': lista_n_neighbors,
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan']
    },

    {
        'knn__n_neighbors': lista_n_neighbors,
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['minkowski'],
        'knn__p': [1.5, 3, 4]
    }
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

grid_search.fit(X,y)


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



#------------------------------------------------------------------------------

if __name__ == "__main__":

    # Salva su disco la pipeline addestrata (preprocessing + modello) e il
    # label encoder, in modo da poterli ricaricare da un altro script senza
    # dover rieseguire il training da capo.
    # Creo la nuova directory se non esiste, altrimenti continuo
    Path(PRJ_ROOT_DIR / "models/knn").mkdir(parents = True, exist_ok = True)

    pipeline_path = PRJ_ROOT_DIR / "models/knn/churn_pipeline_knn.joblib"
    label_encoder_path = PRJ_ROOT_DIR / "models/knn/churn_label_encoder_knn.joblib"
    features_path = PRJ_ROOT_DIR / "models/knn/churn_feature_columns_knn.joblib"

    joblib.dump(grid_search.best_estimator_, pipeline_path)
    joblib.dump(label_encoder, label_encoder_path)
    joblib.dump(list(X.columns), features_path)


    # Aggiorno il json dei modelli disponibili
    model_metadata = {
            "name": "knn", 
            "desc": "K nearest neighborg classifier",
            "pipeline_path": str(pipeline_path.relative_to(PRJ_ROOT_DIR)),
            "label_encoder_path": str(label_encoder_path.relative_to(PRJ_ROOT_DIR)),
            "features_path": str(features_path.relative_to(PRJ_ROOT_DIR))
        }

    save_metadata(model_metadata, PRJ_ROOT_DIR / "models/metadata.json")