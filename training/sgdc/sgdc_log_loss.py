from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from training.save_metadata import save_metadata
import pandas as pd
import joblib 

PRJ_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
df = pd.read_csv(PRJ_ROOT_DIR / "data/clean/telco_customer_churn_clean.csv")
#print(df)

# Separa target dalle feature
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
        ("classifier", SGDClassifier(loss="log_loss", penalty="l2", max_iter=100, random_state=42)),
    ]
)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.19, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

accuracy = pipeline.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")

y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred),"\n")
print(confusion_matrix(y_test, y_pred))


def predict_churn(record: dict) -> dict:
    """
    Prende in input un singolo record del dataset churn (come dizionario
    con le stesse colonne/feature usate in fase di training, ESCLUSO il
    target "Churn"), applica lo stesso encoding usato in training tramite
    la pipeline, e restituisce l'esito della predizione.

    Esempio d'uso:
        nuovo_cliente = {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            ...  # tutte le altre colonne presenti in X
        }
        risultato = predict_churn(nuovo_cliente)
        print(risultato)
    """
    # Controlla che tutte le colonne del dataframe siano presenti in record
    missing_cols = set(X.columns) - set(record.keys())
    if missing_cols:
        raise ValueError(f"Mancano queste colonne nel record: {missing_cols}")
    
    # Trasforma il dizionario in un DataFrame con una sola riga,
    # rispettando le stesse colonne viste durante il training
    record_df = pd.DataFrame([record], columns=X.columns)

    # La pipeline applica automaticamente OneHotEncoder + StandardScaler
    # esattamente come fatto sul train, poi passa i dati al classificatore
    pred_encoded = pipeline.predict(record_df)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    # decision_function restituisce la distanza dall'iperpiano di separazione:
    # valori più alti (in valore assoluto) indicano maggiore confidenza
    score = pipeline.decision_function(record_df)[0]

    return {
        "prediction": pred_label,
        "decision_score": float(score),
    }



if __name__ == "__main__":

    # Salva su disco la pipeline addestrata (preprocessing + modello) e il
    # label encoder, in modo da poterli ricaricare da un altro script senza
    # dover rieseguire il training da capo.
    # Creo la nuova directory se non esiste, altrimenti continuo
    Path(PRJ_ROOT_DIR / "models/sgdc_log_loss").mkdir(parents = True, exist_ok = True)

    pipeline_path = PRJ_ROOT_DIR / "models/sgdc_log_loss/churn_pipeline_sgdc_log_loss.joblib"
    label_encoder_path = PRJ_ROOT_DIR / "models/sgdc_log_loss/churn_label_encoder_sgdc.joblib"
    features_path = PRJ_ROOT_DIR / "models/sgdc_log_loss/churn_feature_columns_sgdc.joblib"

    joblib.dump(pipeline, pipeline_path)
    joblib.dump(label_encoder, label_encoder_path)
    joblib.dump(list(X.columns), features_path)


    # Aggiorno il json dei modelli disponibili
    model_metadata = {
            "name": "sgdc_log_loss", 
            "desc": "Stochastic Gradient Descent Classifier (log loss)",
            "pipeline_path": str(pipeline_path.relative_to(PRJ_ROOT_DIR)),
            "label_encoder_path": str(label_encoder_path.relative_to(PRJ_ROOT_DIR)),
            "features_path": str(features_path.relative_to(PRJ_ROOT_DIR))
        }

    save_metadata(model_metadata, PRJ_ROOT_DIR / "models/metadata.json")


