import os
import sys
import re
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import plotly.graph_objs as go
import plotly.io as pio
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go


def alphanumeric_sort(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


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


def save_learning_curves_plot(learning_df, save_prefix='plot/learning_curves_plot'):
    plt.figure(figsize=(10, 6))
    plt.errorbar(learning_df['Num_Training_Samples'], learning_df['R2'],
                 yerr=learning_df['R2_std'], label='R²', marker='o', capsize=5, linewidth=2)
    plt.xlabel('Number of Training Samples')
    plt.ylabel('R² Score')
    plt.title('Learning Curve - R²')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_r2.png', dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.errorbar(learning_df['Num_Training_Samples'], learning_df['MSE'],
                 yerr=learning_df['MSE_std'], label='MSE', marker='o', capsize=5, linewidth=2)
    plt.errorbar(learning_df['Num_Training_Samples'], learning_df['MAE'],
                 yerr=learning_df['MAE_std'], label='MAE', marker='o', capsize=5, linewidth=2)
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Error')
    plt.title('Learning Curves - MSE and MAE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_mae.png', dpi=300)
    plt.close()


def plot_parity_interactive(df_plot, iteration, output_path):
    fig = px.scatter(
        df_plot,
        x='True Value', y='Predicted Value',
        error_y='Uncertainty (±1σ)',
        hover_name='Structure',
        title=f'Interactive Parity Plot – Iteration {iteration}',
        height=800, width=600
    )

    fig.update_traces(marker=dict(size=4, opacity=0.8, color='royalblue'))

    lim_min = df_plot[['True Value', 'Predicted Value']].min().min()
    lim_max = df_plot[['True Value', 'Predicted Value']].max().max()
    fig.add_shape(type='line', x0=lim_min, y0=lim_min, x1=lim_max, y1=lim_max,
                  line=dict(color='red', dash='dash'))
    fig.update_xaxes(range=[lim_min, lim_max])
    fig.update_yaxes(range=[lim_min, lim_max], scaleanchor="x", scaleratio=1)

    fig.update_layout(template='plotly_white', showlegend=False)
    fig.write_html(output_path)


def plot_parity_detailed(y_true, mean_preds, std_preds, deviating_points, structure_all,
                         structure_uncertain, structure_deviating, idx_uncertain, iteration, output_path):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=y_true[:, 0], y=mean_preds[:, 0],
        error_y=dict(type='data', array=std_preds[:, 0], visible=True),
        mode='markers', marker=dict(size=4, color='gray', opacity=0.35),
        name='All Points ±1σ', text=structure_all,
        hovertemplate='Structure: %{text}<br>True: %{x:.3f}<br>Pred: %{y:.3f}<br>±σ : %{error_y.array:.3f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=np.array(deviating_points['True Values']),
        y=np.array(deviating_points['Predicted Values']),
        error_y=dict(type='data', array=np.array(deviating_points['Pred Std Dev']), visible=True),
        mode='markers', marker=dict(size=7, color='blue', opacity=0.6),
        name='Deviating Points', text=structure_deviating,
        hovertemplate='Structure: %{text}<br>True: %{x:.3f}<br>Pred: %{y:.3f}<br>±σ : %{error_y.array:.3f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=y_true[idx_uncertain, 0], y=mean_preds[idx_uncertain, 0],
        error_y=dict(type='data', array=std_preds[idx_uncertain, 0], visible=True),
        mode='markers', marker=dict(size=7, color='red', opacity=0.8),
        name='Most Uncertain', text=structure_uncertain,
        hovertemplate='Structure: %{text}<br>True: %{x:.3f}<br>Pred: %{y:.3f}<br>±σ : %{error_y.array:.3f}<extra></extra>'
    ))

    lim_min = min(y_true[:, 0].min(), mean_preds[:, 0].min())
    lim_max = max(y_true[:, 0].max(), mean_preds[:, 0].max())
    fig.add_shape(type='line', x0=lim_min, y0=lim_min, x1=lim_max, y1=lim_max,
                  line=dict(color='black', dash='dash'))

    fig.update_xaxes(range=[lim_min, lim_max])
    fig.update_yaxes(range=[lim_min, lim_max], scaleanchor="x", scaleratio=1)

    fig.update_layout(
        template='plotly_white',
        title=f'Interactive Parity Plot – Iteration {iteration}',
        xaxis_title='True Value',
        yaxis_title='Predicted Value',
        width=800, height=600
    )
    fig.write_html(output_path)

    
    
