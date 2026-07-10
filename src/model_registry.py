import pandas as pd
import joblib
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# FIXME
def get_path(func: str, model: str):
    """
    restituisce il path che esegue la funzionalità func.
    Accettati solo i seguenti valori: "pipeline", "label_encoder", "feature_columns"
    """
    return str(ROOT_DIR) + "/models/" + model + "/churn_" + func + "_" + model + ".joblib"

def get_available_models():
    """
    get_available_models restituisce i modelli disponibili.
    I modelli disponibili hanno come identificatore il nome della propria
    directory, all'interno della cartella models
    """
    try: 
        models = os.listdir(ROOT_DIR / 'models')
    except Exception: 
        return []
    return models

def model_to_dict(model_name: str):
    """
    model_to_dict a partire da model_name restutuisce un dizionario contenente
    il label encoder, le colonne contenenti le features, e la pipeline per la predizione, 
    nel formato utilizzato dalla funzione predict
    """
    selected_model = {
            "pipeline" : joblib.load(get_path('pipeline', model_name)),
            "label_encoder" : joblib.load(get_path('label_encoder', model_name)),
            "feature_columns" : joblib.load(get_path('feature_columns', model_name))
        }
    return selected_model



def predict(model: dict, record: dict):
    """ Predizione dell'abbandono di un'utente
    Argomenti: 
    model -- dizionario con all'interno: 
        label_encoder
        feature_columns
        pipeline
    
    record -- dizionario contenente le informazioni dell'utente
    """
    df_record = pd.DataFrame([record], columns = model['feature_columns'])

    pred_encoded = model['pipeline'].predict(df_record )[0]
    pred_label = model['label_encoder'].inverse_transform([pred_encoded])[0]
    score = model['pipeline'].decision_function(df_record)[0]

    return {
        "prediction": pred_label,
        "decision_score": float(score),
    }
    
# TEST
df = pd.read_csv(ROOT_DIR / "data/clean/telco_customer_churn_clean.csv")
dummy = df.iloc[0]
models = get_available_models()

print(predict(model_to_dict(models[0]), dummy))

