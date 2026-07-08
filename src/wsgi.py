from flask import Flask, request, render_template
import joblib
import pandas as pd
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent

# Carica gli oggetti salvati da sgdc.py
pipeline = joblib.load(SCRIPT_DIR / "../models/sgdc/churn_pipeline_sgdc.joblib")
label_encoder = joblib.load(SCRIPT_DIR / "../models/sgdc/churn_label_encoder_sgdc.joblib")
feature_columns = joblib.load(SCRIPT_DIR / "../models/sgdc/churn_feature_columns_sgdc.joblib")
print(f"feature_cols = {feature_columns}")

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

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def customer_info():
    if request.method == 'POST':
        # HACK
        record = request.form.to_dict()
        record["Unnamed: 0"] = 3

        print(record)
        pred = predict_churn(record)
        
        return f"predizione: {pred['prediction']}"
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
