import os
import sys
import re
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import plotly.graph_objs as go
import plotly.io as pio


# ==========================
# Acquisition Functions
# ==========================

def expected_improvement(X, X_sample, Y_sample, model, xi=0.01):
    mu, sigma = model.predict(X, return_std=True)
    mu_sample_opt = np.max(Y_sample)
    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei

def propose_batch_greedy_index(acquisition, X_sample, Y_sample, model, candidate_pool, batch_size=3):
    ei = acquisition(candidate_pool, model)
    top_indices = np.argsort(ei)[-batch_size:][::-1]
    return candidate_pool[top_indices], top_indices

def upper_confidence_bound(X, model, kappa=2):
    mu, sigma = model.predict(X, return_std=True)
    return mu + kappa * sigma


# ==========================
# Plot Functions
# ==========================

def plot_with_errorbars(x, y, yerr, label, color, alpha=0.3, highlight_indices=None):
    plt.errorbar(x, y, yerr=yerr, fmt='o', markersize=2, color=color, alpha=alpha,
                 ecolor=color, elinewidth=0.5, capsize=2, label=label)
    if highlight_indices is not None:
        plt.errorbar(x[highlight_indices], y[highlight_indices], yerr=yerr[highlight_indices],
                     fmt='o', markersize=4, color='red', alpha=0.6,
                     ecolor='red', elinewidth=1, capsize=3, label='Highlighted')

def plot_parity_generic(y_true, y_pred, y_std, highlight_indices, iteration, filename, title_suffix=""):
    plt.figure(figsize=(10, 10))
    plot_with_errorbars(y_true, y_pred, y_std, label='Predictions', color='gray', highlight_indices=highlight_indices)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Parity line (y = x)')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Parity Plot - Iteration {iteration} {title_suffix}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_mean_std(y_true, y_pred, y_std, iteration, highlight_indices):
    x_axis = np.arange(len(y_pred))
    plt.figure(figsize=(10, 6))
    plot_with_errorbars(x_axis, y_pred, y_std, label='Prediction ±1 Std Dev', color='gray', highlight_indices=highlight_indices)
    plt.title(f'Mean ± Std Dev - Iteration {iteration}')
    plt.xlabel('Sample Index')
    plt.ylabel('Predicted Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plot/mean_std_predictions_iteration_{iteration}.png')
    plt.close()

def plot_parity(y_true, y_pred, y_std, highlight_indices, iteration):
    plot_parity_generic(y_true, y_pred, y_std, highlight_indices, iteration,
                        filename=f'plot/mean_parity_plot_iteration_{iteration}.png')

def plot_parity_tot(y_true, y_pred, y_std, highlight_indices, iteration):
    plot_parity_generic(y_true, y_pred, y_std, highlight_indices, iteration,
                        filename=f'plot/mean_parity_plot_tot_iteration_{iteration}.png',
                        title_suffix="(Total)")

def plot_parity_interactive(y_true, mean_preds, std_preds, highlight_idx, iter_num, structure_names):
    fig = go.Figure()

    # All predictions
    fig.add_trace(go.Scatter(
        x=y_true,
        y=mean_preds,
        error_y=dict(type='data', array=std_preds, visible=True),
        mode='markers',
        marker=dict(color='gray', size=6, opacity=0.3),
        name='Predictions',
        text=structure_names,
        hoverinfo='text+x+y'
    ))

    # Highlighted uncertain points
    fig.add_trace(go.Scatter(
        x=y_true[highlight_idx],
        y=mean_preds[highlight_idx],
        error_y=dict(type='data', array=std_preds[highlight_idx], visible=True),
        mode='markers',
        marker=dict(color='red', size=8, opacity=0.6),
        name='Most Uncertain Points',
        text=structure_names[highlight_idx],
        hoverinfo='text+x+y'
    ))

    # Parity line
    min_val = min(y_true.min(), mean_preds.min())
    max_val = max(y_true.max(), mean_preds.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Parity Line'
    ))

    fig.update_layout(
        title=f'Iteration {iter_num}',
        xaxis_title='True Values',
        yaxis_title='Predicted Values',
        template='plotly_white',
        width=700,
        height=700
    )

    filename = f'plot/interactive_parity_plot_iteration_{iter_num}.html'
    pio.write_html(fig, file=filename, auto_open=False)

def plot_learning_curve():
    learning_df = pd.read_csv('data/learning_curves.csv')
    plt.figure(figsize=(10, 6))
    for metric in ['R2', 'MSE', 'MAE']:
        plt.errorbar(learning_df['Num_Training_Samples'],
                     learning_df[metric],
                     yerr=learning_df[f'{metric}_std'],
                     label=metric, marker='o', capsize=5, linewidth=2)
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Metric Value')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plot/learning_curves_plot.png', dpi=300)
    plt.close()


# ==========================
# Utility Functions
# ==========================

def alphanumeric_sort(s):
    """Sort strings in natural alphanumeric order (e.g., x2 < x10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def build_structure_per_sample(main_folder: str, bin_index: int) -> np.ndarray:
    structure_list = []
    folders_lvl1 = sorted(
        [name for name in os.listdir(main_folder)
         if os.path.isdir(os.path.join(main_folder, name)) and '-' in name],
        key=alphanumeric_sort
    )
    for folder in folders_lvl1:
        folder_path = os.path.join(main_folder, folder)
        subdirs = sorted(
            [name for name in os.listdir(folder_path)
             if os.path.isdir(os.path.join(folder_path, name))],
            key=alphanumeric_sort
        )
        for subdir in subdirs:
            subdir_path = os.path.join(folder_path, subdir)
            sub_sub_dirs = sorted(
                [name for name in os.listdir(subdir_path)
                 if os.path.isdir(os.path.join(subdir_path, name))],
                key=alphanumeric_sort
            )
            for sub_sub_dir in sub_sub_dirs:
                path = os.path.join(subdir_path, sub_sub_dir)
                gcn_path = os.path.join(path, "GCN/gcn_genome.dat")
                try:
                    df = pd.read_csv(gcn_path, sep=r'\s+', comment='#', header=None)
                    if df.shape[0] > bin_index and df.shape[1] > 2:
                        structure_name = f"{folder}/{subdir}/{sub_sub_dir}"
                        structure_list.append(structure_name)
                except Exception:
                    pass
    return np.array(structure_list)


