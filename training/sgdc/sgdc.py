from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import pandas as pd
import joblib 

SCRIPT_DIR = Path(__file__).resolve().parent
df = pd.read_csv(SCRIPT_DIR / "../../data/clean/telco_customer_churn_clean.csv")
print(df)

# Separa target dalle feature
y = df["Churn"]
X = df.drop(columns=["Churn"])

# Encoding del target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Individua automaticamente le colonne categoriche e numeriche
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

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
        ("classifier", SGDClassifier(loss="hinge", penalty="l2", max_iter=1000)),
    ]
)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

accuracy = pipeline.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")

y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))
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
    # Esempio: prendo un record esistente dal test set e lo passo alla pipeline
    esempio_record = X_test.iloc[0].to_dict()
    print("\nEsempio di predizione su un singolo record:")
    print(predict_churn(esempio_record))

if __name__ == "__main__":
    # ... (predizione di esempio)

    # Salva su disco la pipeline addestrata (preprocessing + modello) e il
    # label encoder, in modo da poterli ricaricare da un altro script senza
    # dover rieseguire il training da capo.
    joblib.dump(pipeline, SCRIPT_DIR / "../../models/sgdc/churn_pipeline_sgdc.joblib")
    joblib.dump(label_encoder, SCRIPT_DIR / "../../models/sgdc/churn_label_encoder_sgdc.joblib")
    joblib.dump(list(X.columns), SCRIPT_DIR / "../../models/sgdc/churn_feature_columns_sgdc.joblib")
    print("\nModello salvato in: ../../models/sgdc/churn_pipeline_sgdc.joblib")
