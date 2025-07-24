import os
import glob
import ast
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 20,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 16
})


def plot_learning_curve_shaded_error(csv_path, output_path):
    df = pd.read_csv(csv_path)
    x = df['Num_Training_Samples']
    y = df['R2']
    yerr = df['R2_std']

    plt.figure(figsize=(6, 4))
    plt.plot(x, y, label='Validation', marker='o', linewidth=2, color='olivedrab')
    plt.fill_between(x, y - yerr, y + yerr, color='olivedrab', alpha=0.3)
    plt.xlabel('Number of training samples')
    plt.ylabel('$R^2$')
    plt.legend(loc='lower right')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def get_latest_iteration_metrics(df):
    latest_iter = df['Iteration'].max()
    row = df[df['Iteration'] == latest_iter]
    if not row.empty:
        return latest_iter, {
            'r2_val': row['R2'].values[0],
            'r2_std': row['R2_std'].values[0],
            'mae_val': row['MAE'].values[0],
            'mae_std': row['MAE_std'].values[0]
        }
    return latest_iter, {}


def read_uncertainty_indices(filepath):
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        if first_line.startswith('#'):
            indices_str = first_line.split(':', 1)[1].strip()
            return ast.literal_eval(indices_str)
    return []


def plot_parity(df, metrics, iteration, output_path):
    df_val = df[df['Split'] == 'validation']
    df_train = df[df['Split'] == 'train']

    min_val = min(df['True Values'].min(), df['Predicted Values'].min())
    max_val = max(df['True Values'].max(), df['Predicted Values'].max())

    plt.figure(figsize=(6, 6))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')

    plt.errorbar(df_val['True Values'], df_val['Predicted Values'], yerr=df_val['Std Dev'],
                 fmt='o', markersize=3, color='royalblue', alpha=0.6,
                 ecolor='cornflowerblue', elinewidth=0.5, capsize=3, label='Validation')

    plt.errorbar(df_train['True Values'], df_train['Predicted Values'], yerr=df_train['Std Dev'],
                 fmt='o', markersize=3, color='indianred', alpha=0.6,
                 ecolor='lightcoral', elinewidth=0.5, capsize=3, label='Train')

    plt.xlabel('MA$_{@0.9V}$ True [A/mg]')
    plt.ylabel('MA$_{@0.9V}$ Predicted [A/mg]')
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.legend(loc='lower right')

    if metrics and 'r2_val' in metrics and 'mae_val' in metrics:
        plt.text(0.05, 0.95,
                 f"$R^2$ = {metrics['r2_val']:.3f} ± {metrics['r2_std']:.3f}\n"
                 f"MAE = {metrics['mae_val']:.3f} ± {metrics['mae_std']:.3f} [A/mg]",
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_top_metrics(experiment_dirs, filename, ylabel, scale=1, output_path=None):
    plt.figure(figsize=(6, 4))

    for i, exp_dir in enumerate(experiment_dirs):
        csv_path = os.path.join(exp_dir, filename)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            plt.plot(df['Iteration'], df['Percentage'] * scale, marker='o', linestyle='--',
                     markersize=4, linewidth=1, label=f'run_{i+1}')

    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.legend(loc='lower right')
    plt.ylim(0, 100)
    plt.xlim(0, 400)
    plt.yticks(range(0, 101, 20))
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    learning_curve_path = 'data/learning_curves.csv'
    plot_learning_curve_shaded_error(learning_curve_path, 'R2_learning_curves_plot_shaded.png')

    df_learning_curves = pd.read_csv(learning_curve_path)
    latest_iteration, metrics = get_latest_iteration_metrics(df_learning_curves)

    data_iter_path = f'data/data_iteration_{latest_iteration}.csv'
    most_uncertain_indices = read_uncertainty_indices(data_iter_path)
    df_data = pd.read_csv(data_iter_path, skiprows=1)

    plot_parity(df_data, metrics, latest_iteration, f'parity_plot_colored_{latest_iteration}.png')

    experiment_dirs = glob.glob('data/')
    plot_top_metrics(experiment_dirs, 'top_100_metrics.csv', 'Percentage', scale=1, output_path='Top_100.png')
    plot_top_metrics(experiment_dirs, 'top_10_metrics.csv', 'Percentage', scale=10, output_path='Top_10.png')


if __name__ == '__main__':
    main()

