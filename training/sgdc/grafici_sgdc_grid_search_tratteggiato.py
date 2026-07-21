import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib


def plot_grid_search(cv_results_):
    df = pd.DataFrame(cv_results_)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=True)
    losses = ['hinge', 'log_loss', 'modified_huber']
    penalties = ['l1', 'l2', 'elasticnet']
    class_weights = ['balanced', None]
    
    # --- Configurazione Colori ---
    def get_trace_color(penalty, l1_ratio=None):
        if penalty == 'l1': return 'tab:blue'
        if penalty == 'l2': return 'tab:orange'
        if penalty == 'elasticnet':
            if l1_ratio == 0.15: return 'tab:green'
            if l1_ratio == 0.5: return 'tab:red'
            if l1_ratio == 0.85: return 'tab:purple'
        return 'black'
    
    # --- Nuova Configurazione Stili (class_weight) ---
    cw_styles = {
        'balanced': '-',   # Linea continua
        'None': '--'       # Linea tratteggiata
    }
    
    for ax, loss in zip(axes, losses):
        subset = df[df['param_classifier__loss'] == loss]
        
        for penalty in penalties:
            for cw in class_weights:
                
                if cw is None:
                    cw_mask = subset['param_classifier__class_weight'].isna()
                    cw_label = "None"
                else:
                    cw_mask = subset['param_classifier__class_weight'] == cw
                    cw_label = cw
                
                # Assegniamo lo stile in base al class_weight
                trace_style = cw_styles.get(cw_label, '-')
                
                if penalty == 'elasticnet':
                    for l1_r in subset['param_classifier__l1_ratio'].unique():
                        if pd.isna(l1_r): continue
                            
                        trace_data = subset[
                            (subset['param_classifier__penalty'] == penalty) &
                            cw_mask &
                            (subset['param_classifier__l1_ratio'] == l1_r)
                        ].sort_values('param_classifier__alpha')
                        
                        if not trace_data.empty:
                            label = f"{penalty} | l1_r: {l1_r} | cw: {cw_label}"
                            base_color = get_trace_color(penalty, l1_r)
                            
                            ax.plot(trace_data['param_classifier__alpha'], 
                                    trace_data['mean_test_score'], 
                                    marker='o', markersize=4,
                                    color=base_color, 
                                    linestyle=trace_style, # Continua o tratteggiata
                                    alpha=1.0,             # Colore pieno per tutte
                                    label=label)
                else:
                    trace_data = subset[
                        (subset['param_classifier__penalty'] == penalty) & cw_mask
                    ].sort_values('param_classifier__alpha')
                    
                    if not trace_data.empty:
                        label = f"{penalty} | cw: {cw_label}"
                        base_color = get_trace_color(penalty)
                        
                        ax.plot(trace_data['param_classifier__alpha'], 
                                trace_data['mean_test_score'], 
                                marker='o', markersize=4,
                                color=base_color, 
                                linestyle=trace_style, # Continua o tratteggiata
                                alpha=1.0,             # Colore pieno per tutte
                                label=label)
        
        ax.set_xscale('log')
        ax.set_title(f'Loss: {loss.upper()}', fontweight='bold', fontsize=14)
        ax.set_xlabel('Alpha (log scale)')
        ax.grid(True, linestyle=':', alpha=0.6) # Ho reso la griglia puntinata per non confonderla con i tratteggi

    axes[0].set_ylabel('F1 Score', fontweight='bold')
    
    # --- CREAZIONE E ORDINAMENTO LEGENDA UNICA ---
    handles_dict = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in handles_dict:
                handles_dict[label] = handle
                
    # Ordine forzato: prima balanced (continua), poi None (tratteggiata) sotto al corrispettivo colore
    ordered_label_names = [
        "l1 | cw: balanced",
        "l1 | cw: None",
        "l2 | cw: balanced",
        "l2 | cw: None",
        "elasticnet | l1_r: 0.15 | cw: balanced",
        "elasticnet | l1_r: 0.15 | cw: None",
        "elasticnet | l1_r: 0.5 | cw: balanced",
        "elasticnet | l1_r: 0.5 | cw: None",
        "elasticnet | l1_r: 0.85 | cw: balanced",
        "elasticnet | l1_r: 0.85 | cw: None"
    ]
    
    final_handles = []
    final_labels = []
    for label in ordered_label_names:
        if label in handles_dict:
            final_labels.append(label)
            final_handles.append(handles_dict[label])
            
    fig.legend(final_handles, final_labels, 
               loc='lower center', 
               bbox_to_anchor=(0.5, 0.02),
               ncol=5, 
               fontsize='10',
               frameon=True)
    
    fig.subplots_adjust(bottom=0.25, wspace=0.15)
    
    plt.show()


    
if __name__ == "__main__":

    PRJ_ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    grid_cv_results = joblib.load(PRJ_ROOT_DIR / "models/sgdc/risultati_grid_search.joblib")
    plot_grid_search(grid_cv_results)

