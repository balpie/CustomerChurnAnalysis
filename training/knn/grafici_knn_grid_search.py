import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

PRJ_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
grid_cv_result = joblib.load(PRJ_ROOT_DIR / "models/knn/risultati_grid_search_2_k_piu_10.joblib")
results_df = pd.DataFrame(grid_cv_result)

# Creo una colonna 'label' che identifica la configurazione di distanza
def make_label(row):
    if row['param_knn__metric'] == 'minkowski':
        return f"minkowski (p={row['param_knn__p']})"
    else:
        return row['param_knn__metric']

results_df['distance_label'] = results_df.apply(make_label, axis=1)

# --- Un grafico separato e più grande per ogni valore di weights ---
weights_values = results_df['param_knn__weights'].unique()

for w in weights_values:
    # Crea una nuova finestra per ogni iterazione (es. 12x8 pollici)
    plt.figure(figsize=(12, 8))
    
    # Ottiene l'asse della figura appena creata
    ax = plt.gca() 
    
    subset = results_df[results_df['param_knn__weights'] == w]
    
    for label, group in subset.groupby('distance_label'):
        group_sorted = group.sort_values('param_knn__n_neighbors')
        
        ax.plot(
            group_sorted['param_knn__n_neighbors'],
            group_sorted['mean_test_score'],
            marker='o',
            label=label
        )
    
    # Formattazione del grafico singolo
    ax.set_title(f"weights = '{w}'", fontsize=16) 
    ax.set_xlabel("n_neighbors", fontsize=14)
    ax.set_ylabel("mean_test_score", fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(title="Distanza", fontsize=12, title_fontsize=12)
    
    plt.tight_layout()

plt.show()

# --- Grafico con un subplot per ogni valore di weights ---
weights_values = results_df['param_knn__weights'].unique()

fig, axes = plt.subplots(1, len(weights_values), figsize=(20, 8), sharey=True)

if len(weights_values) == 1:
    axes = [axes]

for ax, w in zip(axes, weights_values):
    subset = results_df[results_df['param_knn__weights'] == w]
    
    for label, group in subset.groupby('distance_label'):
        group_sorted = group.sort_values('param_knn__n_neighbors')
        ax.plot(
            group_sorted['param_knn__n_neighbors'],
            group_sorted['mean_test_score'],
            marker='o',
            label=label
        )
    
    ax.set_title(f"weights = '{w}'")
    ax.set_xlabel("n_neighbors")
    ax.grid(alpha=0.3)

axes[0].set_ylabel("mean_test_score")
axes[0].legend(title="Distanza")
plt.tight_layout()
plt.show()