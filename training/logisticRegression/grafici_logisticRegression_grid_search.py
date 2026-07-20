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
 
    plt.show()