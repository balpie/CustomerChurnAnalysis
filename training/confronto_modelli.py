from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report,ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
from pathlib import Path

import matplotlib as plt

def calcola_risultati():
    """
    Funzione che calcola le metriche di valutazione di alcuni modelli e li restituisce sottoforma di array di dizionari
    dove un singolo dizionario è della forma:
        
        {
            "Modello": nome,
            "ConfusionMatrix": cm,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            #"ClassificationReport":
        }

    L'array di dizionari viene utilizzato per fare grafici e valutazioni delle performance 

    In futuro può essere possibile non dover eseguire encoding, preprocessor e addestramento,
    ma utilizzare dei joblib.


    """

    SCRIPT_DIR = Path(__file__).resolve().parent
    df = pd.read_csv(SCRIPT_DIR / "../data/clean/telco_customer_churn_clean.csv")

    # Separa target dalle feature
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    # Encoding del target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    print(label_encoder.classes_)

    # Individua automaticamente le colonne categoriche e numeriche
    categorical_cols = X.select_dtypes(include=["object", "str","category"]).columns.tolist()
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

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )



    modelli = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier()
    }




    risultati = []
    for nome, modello in modelli.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessore", preprocessor),
                ("classificatore",modello)
            ]
        )


        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        #calcolo confusion matrix e metriche di valutazione
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred,pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)
        classification_report(y_test,y_pred,)

        risultati.append({
            "Modello": nome,
            "ConfusionMatrix": cm,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            #"ClassificationReport":
        })

        print(f"\n--- {nome} ---")
        print("Confusion Matrix:")
        print(cm)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1:        {f1:.4f}")

    return risultati
    

import seaborn as sns
import numpy as np
from math import ceil


def grafica_risultati(risultati):
    df_metriche = pd.DataFrame(risultati)[
        ["Modello", "Accuracy", "Precision", "Recall", "F1"]
    ].set_index("Modello")

    # --- Grafico 1: barre con le metriche a confronto ---
    ax = df_metriche.plot(kind="bar", figsize=(10, 6), rot=45)
    ax.set_title("Confronto metriche tra modelli")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # --- Grafico 2: confusion matrix per ciascun modello ---
    n = len(risultati)
    cols = 3
    rows = ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)  # gestisce anche il caso di un solo modello

    for i, r in enumerate(risultati):
        sns.heatmap(
            r["ConfusionMatrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[i],
            cbar=False,
        )
        axes[i].set_title(r["Modello"])
        axes[i].set_xlabel("Predetto")
        axes[i].set_ylabel("Reale")

    # nasconde eventuali assi vuoti in eccesso
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
    print("fine grafica")




def grafica_metriche_confronto(risultati):
    df = pd.DataFrame(risultati)[
        ["Modello", "Accuracy", "Precision", "Recall", "F1"]
    ].set_index("Modello")

    metriche = df.columns.tolist()      # ["Accuracy", "Precision", "Recall", "F1"]
    modelli = df.index.tolist()

    n_metriche = len(metriche)
    n_modelli = len(modelli)

    x = np.arange(n_metriche)           # posizione dei gruppi (una per metrica)
    larghezza_barra = 0.8 / n_modelli   # larghezza di ogni singola barra

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, modello in enumerate(modelli):
        valori = df.loc[modello, metriche].values
        posizione = x - 0.4 + larghezza_barra * (i + 0.5)
        barre = ax.bar(posizione, valori, larghezza_barra, label=modello)
        ax.bar_label(barre, fmt="%.2f", fontsize=7, rotation=90, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(metriche)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_title("Confronto metriche per tutti i modelli")
    ax.legend(title="Modello", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    risultati = calcola_risultati()
    grafica_risultati(risultati)
    grafica_metriche_confronto(risultati)

    


