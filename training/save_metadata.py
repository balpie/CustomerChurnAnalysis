"""
Questo modulo provvede le funzioni per esportare i metadati del modello 
allenato nel file /models/metadata.json mantenendo il file non ridondante
"""
import json
from pathlib import Path


def save_metadata(metadata: dict, dest: Path):
    """
    argomenti: 
    metadata: dizionario di metadati, con attributo 
        'name' univoco che identifica il modello
    dest: file .json destinazione per i metadati
    """
    # Aggiorno il json dei modelli disponibili
    try:
        with open(dest) as f:
            models = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        models = []

    if not isinstance(models, list):
        models = []

    if any(x["name"] == metadata["name"] for x in models):
        print(f"Modello di nome {metadata['name']} già presente")
        return

    models.append(metadata)
    with open(dest, "w") as f:
        json.dump(models, f)


