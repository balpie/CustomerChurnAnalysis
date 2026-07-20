import pandas as pd
import joblib
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def get_available_models():
    """
    get_available_models restituisce i modelli disponibili.
    I modelli disponibili hanno come identificatore il nome della propria
    directory, all'interno della cartella models
    """
    try: 
        with open(ROOT_DIR / "models/metadata.json") as f:
            models = json.load(f)
    except Exception: 
        return []
    return models


def predict(form: dict):
    """ Predizione dell'abbandono di un'utente
    Argomenti: 
    form: dizionario con le risposte del form.
    form['model'] contiene l'identificatore del modello da utilizzare
    """

    # FIXME: una volta caricato il modello dovrebbe rimanere in memoria
    with open(ROOT_DIR / "models/metadata.json") as f:
        models = json.load(f)
    curr_model = None
    for mm in models:
        if mm["name"] == form["model"]:
            curr_model = mm
            break
    if curr_model is None:
        print("ERRORE: modello invalido selezionato")
        raise ValueError

    label_encoder = joblib.load(ROOT_DIR / curr_model["label_encoder_path"])
    pipeline = joblib.load(ROOT_DIR / curr_model["pipeline_path"])
    features = joblib.load(ROOT_DIR / curr_model["features_path"])

    df_record = pd.DataFrame([form], columns = features)

    pred_encoded = pipeline.predict(df_record )[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    result = {
        "prediction": pred_label,
    }

    if hasattr(pipeline, "decision_function"):
        result["confidence"] = float(
            pipeline.decision_function(df_record)[0]
        )
    elif hasattr(pipeline, "predict_proba"):
        result["confidence"] = float(
            pipeline.predict_proba(df_record)[0, pred_encoded]
        )

    return result

