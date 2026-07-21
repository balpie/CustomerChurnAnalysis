import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

 
def plot_gridsearch_results(df, score_label="F1 score"):
     
    df["C"] = df["param_clf__C"].astype(float)
    df["l1_ratio"] = df["param_clf__l1_ratio"].astype(float)
    df["solver"] = df["param_clf__solver"].astype(str)
 
    solvers = sorted(df["solver"].unique())
    n_solvers = len(solvers)
 
    fig, axes = plt.subplots(1, n_solvers, figsize=(6 * n_solvers, 5), sharey=True)
    if n_solvers == 1:
        axes = [axes]
 
    for ax, solver in zip(axes, solvers):
        sub = df[df["solver"] == solver]
 
        for l1r in sorted(sub["l1_ratio"].unique()):
            line = sub[sub["l1_ratio"] == l1r].sort_values("C")
            ax.plot(
                line["C"],
                line["mean_test_score"],
                marker="o",
                label=f"l1_ratio={l1r}",
            )
 
        ax.set_xscale("log")
        ax.set_xlabel("C (scala log)")
        ax.set_title(f"solver = {solver}")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, title="Regolarizzazione")
 
    axes[0].set_ylabel(score_label)
    fig.suptitle("GridSearchCV — score al variare di C, l1_ratio e solver", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
 
    return fig

import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import pandas as pd

def plot_gridsearch_results_cw(df, score_label="F1 score"):
    
    # 1. Estrazione dei parametri
    df["C"] = df["param_clf__C"].astype(float)
    df["l1_ratio"] = df["param_clf__l1_ratio"].astype(float)
    df["solver"] = df["param_clf__solver"].astype(str)
    
    # Gestione sicura di class_weight
    if "param_clf__class_weight" in df.columns:
        df["class_weight"] = df["param_clf__class_weight"].fillna("None").astype(str)
    else:
        df["class_weight"] = "None"

    # 2. Identificazione dei valori unici
    solvers = sorted(df["solver"].unique())
    n_solvers = len(solvers)
    
    # Preleviamo una palette di colori standard da matplotlib
    cmap = plt.get_cmap("tab10")

    # 3. Creazione della griglia (1 riga, N colonne per i solver)
    fig, axes = plt.subplots(1, n_solvers, figsize=(6 * n_solvers, 5), sharey=True)
    if n_solvers == 1:
        axes = [axes]

    # 4. Popolamento dei grafici
    for ax, solver in zip(axes, solvers):
        sub_solver = df[df["solver"] == solver]
        
        l1_ratios = sorted(sub_solver["l1_ratio"].dropna().unique())

        # Iteriamo PRIMA sugli l1_ratio per bloccare il colore
        for i, l1r in enumerate(l1_ratios):
            
            # Assegniamo un colore fisso per questo l1_ratio
            colore = cmap(i % 10) 
            
            # Ora iteriamo sui pesi per questo specifico l1_ratio
            for weight in sorted(sub_solver["class_weight"].unique()):
                
                # Filtriamo i dati esatti per solver, l1_ratio e weight
                line = sub_solver[(sub_solver["l1_ratio"] == l1r) & (sub_solver["class_weight"] == weight)].sort_values("C")
                
                # Se non ci sono dati per questa combinazione, saltiamo
                if line.empty:
                    continue
                
                # Stile linea: tratteggiato se "None", continuo altrimenti
                ls = "--" if weight == "None" else "-"
                
                label_str = f"l1={l1r} | cw={weight}"
                
                ax.plot(
                    line["C"],
                    line["mean_test_score"],
                    marker="o",
                    linestyle=ls,
                    color=colore,      # <--- Forza lo STESSO colore per la coppia
                    label=label_str,
                )

        # Formattazione del singolo subplot
        ax.set_xscale("log")
        ax.set_xlabel("C (scala log)")
        ax.set_title(f"solver = {solver}")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, title="l1_ratio | class_weight")

    # 5. Formattazione finale globale
    axes[0].set_ylabel(score_label)
    fig.suptitle("GridSearchCV — score al variare di C, l1_ratio e class_weight", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return fig
 
def plot_best_C_per_solver(cv_results):
    """
    Grafico riassuntivo: per ogni combinazione (solver, l1_ratio),
    mostra il miglior score ottenuto (massimizzando su C) come barre.
    Utile per un confronto rapido tra configurazioni.
    """
    df = pd.DataFrame(cv_results)
    df["l1_ratio"] = df["param_clf__l1_ratio"].astype(float)
    df["solver"] = df["param_clf__solver"].astype(str)
 
    best = (
        df.groupby(["solver", "l1_ratio"])["mean_test_score"]
        .max()
        .reset_index()
    )
    best["label"] = best["solver"] + " / l1=" + best["l1_ratio"].astype(str)
    best = best.sort_values("mean_test_score", ascending=True)
 
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(best) + 2))
    ax.barh(best["label"], best["mean_test_score"])
    ax.set_xlabel("Miglior score CV (su tutti i C)")
    ax.set_title("Confronto solver / l1_ratio (best over C)")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
 
    return fig
 
 
if __name__ == "__main__":

    PRJ_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    grid_cv_result = joblib.load(PRJ_ROOT_DIR / "models/logisticRegression/risultati_grid_search.joblib")
    results_df = pd.DataFrame(grid_cv_result)
     
 
    fig1 = plot_gridsearch_results(results_df)
    #fig1.savefig("score_vs_C_per_solver.png", dpi=150)
 
    fig2 = plot_best_C_per_solver(results_df)
    #fig2.savefig("best_score_per_config.png", dpi=150)

    grid_cv_result_CW = joblib.load(PRJ_ROOT_DIR / "models/logisticRegression/risultati_grid_search_cw.joblib")
    results_df_CW = pd.DataFrame(grid_cv_result_CW)


 
    fig3 = plot_gridsearch_results_cw(results_df_CW)
    #fig1.savefig("score_vs_C_per_solver.png", dpi=150)
 
    fig4 = plot_best_C_per_solver(results_df_CW)
    #fig2.savefig("best_score_per_config.png", dpi=150)
 
    plt.show()