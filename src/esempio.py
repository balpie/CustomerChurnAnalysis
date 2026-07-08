import pandas as pd
import joblib

from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent

# Carica gli oggetti salvati da sgdc.py
pipeline = joblib.load(SCRIPT_DIR / "../models/sgdc/churn_pipeline_sgdc.joblib")
label_encoder = joblib.load(SCRIPT_DIR / "../models/sgdc/churn_label_encoder_sgdc.joblib")
feature_columns = joblib.load(SCRIPT_DIR / "../models/sgdc/churn_feature_columns_sgdc.joblib")


def predict_churn(record: dict) -> dict:
    """
    Prende un dizionario con le feature di un cliente (stesse colonne
    usate in training, ESCLUSO "Churn") e restituisce la predizione.
    """
    record_df = pd.DataFrame([record], columns=feature_columns)

    pred_encoded = pipeline.predict(record_df)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]
    score = pipeline.decision_function(record_df)[0]

    return {
        "prediction": pred_label,
        "decision_score": float(score),
    }


if __name__ == "__main__":
    df = pd.read_csv(SCRIPT_DIR / "../data/clean/telco_customer_churn_clean.csv")
    nuovo_cliente = dict(df.iloc[0][0:20])
    

    risultato = predict_churn(nuovo_cliente)
    print(risultato)
