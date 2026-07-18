from flask import Flask, request, render_template
import pandas as pd
import json
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent

from model_registry import get_available_models, predict 

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def customer_info():
    if request.method == 'POST':

        print(f"Arrivata richiesta: \n{dict(request.form)}")
        # Sanifico l'ingresso
        if any(val == '' for val in request.form.values()):
            return render_template(
                    "index.html",
                    models=get_available_models(),
                    error="Compila tutti i campi."
                ), 400
        pred = predict(request.form.to_dict())
        try:
            pred = predict(request.form.to_dict())
        except Exception as e:
            return render_template(
                "index.html",
                models=get_available_models(),
                error=f"Qualcosa è andato storto: {type(e).__name__}: {e}"
            ), 400

        return render_template("index.html", models = get_available_models(), prediction = pred)
    models = []
    for mm in get_available_models():
        print(mm)
        models.append({"name": mm["name"], "desc": mm["desc"]})

    # Altrimenti, se viene fatto GET assumo che la richiesta arrivi da browser
    return render_template("index.html", models = get_available_models())

@app.route('/models', methods=['POST'])
def api_get_models():
    if request.method == 'POST':
        return(get_available_models())


if __name__ == '__main__':
    app.run(debug=True)
