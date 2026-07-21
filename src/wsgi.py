from flask import Flask, request, render_template
import pandas as pd
import json
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent

from src.model_registry import get_available_models, predict 

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def customer_info():
    """
    Gestisce la pagina principale dell'applicazione.

    - GET: visualizza il form per l'inserimento dei dati del cliente.
    - POST: valida i dati ricevuti, esegue la predizione del modello
      selezionato e restituisce la pagina con il risultato oppure un
      messaggio di errore in caso di input non valido o eccezioni.

    Returns:
        Response: Pagina HTML renderizzata con il form, la predizione
        oppure un messaggio di errore.
    """
    if request.method == 'POST':

        print(f"Arrivata richiesta: \n{dict(request.form)}")
        # Sanifico l'ingresso
        if any(val == '' for val in request.form.values()):
            return render_template(
                    "index.html",
                    models=get_available_models(),
                    error="Compila tutti i campi."
                ), 400
        try:
            pred = predict(request.form.to_dict())
        except Exception as e:
            return render_template(
                "index.html",
                models=get_available_models(),
                error=f"Qualcosa è andato storto: {type(e).__name__}: {e}"
            ), 400

        return render_template("index.html", models = get_available_models(), prediction = dict(pred))

    # Altrimenti, se viene fatto GET assumo che la richiesta arrivi da browser
    return render_template("index.html", models = get_available_models())


if __name__ == '__main__':
    app.run(debug=True)
