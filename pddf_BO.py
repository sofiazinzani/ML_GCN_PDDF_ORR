
import os
import sys
import csv
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from BO_utils import (
    plot_mean_std,
    plot_parity,
    plot_parity_tot,
    plot_parity_interactive,
    plot_learning_curve,
    expected_improvement,
    propose_batch_greedy_index,
    upper_confidence_bound,
    build_structure_per_sample
)

from data_loading import (
    pdf_loading_01,
    MA_loading
)

# === Data Loading ===
main_folder = '/Users/sofiazinzani/Documents/Dottorato/Unimi/Daemon_Cost_Action/Structures/AuM_solo_pddf_01/AuM_Sub'
data_array_x, folders = pdf_loading_01(main_folder)

main_folder = '/Users/sofiazinzani/Documents/Dottorato/Unimi/Daemon_Cost_Action/Structures/Anisotropic/AuM_Sub/'
data_array_y = MA_loading(main_folder, folders)

data_array_x = data_array_x[:, 0:198]

scaler_x = StandardScaler()
data_array_x = scaler_x.fit_transform(data_array_x)

structure_per_sample = build_structure_per_sample(main_folder, bin_index=0)
assert len(structure_per_sample) == data_array_x.shape[0], 'Mismatch in structure-per-sample and data shape.'

structure_names_full = np.array(folders)

all_indices = np.arange(data_array_x.shape[0])
fixed_random = 3
num_active_points = 10
num_iterations = 10
save_every = 1
train_pool_number = 10
train_pool_size = 1 - train_pool_number / len(data_array_x)

idx_pool_train, idx_pool_other, data_array_x_pool_train, data_array_x_pool_other, data_array_y_pool_train, data_array_y_pool_other = train_test_split(
    all_indices, data_array_x, data_array_y, test_size=train_pool_size, random_state=fixed_random)

os.makedirs('data', exist_ok=True)
os.makedirs('plot', exist_ok=True)


learning_curve_path = 'data/learning_curves.csv'
with open(learning_curve_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Iteration', 'Num_Training_Samples', 'R2', 'R2_std', 'MSE', 'MSE_std', 'MAE', 'MAE_std'])

bayes_metrics_path = 'data/bayes_metrics.csv'
with open(bayes_metrics_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Iteration', 'Index_Structure', 'Structure_Name', 'Y', 'Y_max', 'EI'])

bayes_metrics_path_tot = 'data/bayes_y_tot.csv'
with open(bayes_metrics_path_tot, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Iteration', 'Y_max', 'Y_pred_max'])

# === Active Learning Loop ===
all_selected_points = {'True Values': [], 'Predicted Values': [], 'Std Dev': []}

for it in range(num_iterations):
    print(f'Iteration {it}\n{"_" * 22}')

    kernel = C(0.1, (1e-3, 1e3)) * RBF(length_scale=2.0)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=5)
    gp.fit(data_array_x_pool_train, data_array_y_pool_train.ravel())

    preds_tot, stds_tot = gp.predict(data_array_x, return_std=True)
    y_true_tot = data_array_y.ravel()
    preds_flat_tot = preds_tot.ravel()

    max_tot = max(y_true_tot)
    max_pred_tot = preds_flat_tot.max()

    with open(bayes_metrics_path_tot, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([it, max_tot, max_pred_tot])

    # Compute Top-100 match
    top_100_pred = np.argsort(preds_flat_tot)[-100:][::-1]
    top_100_true = np.argsort(y_true_tot)[-100:][::-1]
    top_100_match = len(set(top_100_pred) & set(top_100_true)) / 100 * 100
    print(f'Top 100 match: {top_100_match:.2f}%')

    top_100_metrics_path = 'data/top_100_metrics.csv'
    if it == 0:
        with open(top_100_metrics_path, 'w', newline='') as f:
            csv.writer(f).writerow(['Iteration', 'Percentage'])
    with open(top_100_metrics_path, 'a', newline='') as f:
        csv.writer(f).writerow([it, top_100_match])

    # Compute Top-10 match
    top_10_pred = np.argsort(preds_flat_tot)[-10:][::-1]
    top_10_true = np.argsort(y_true_tot)[-10:][::-1]
    top_10_match = len(set(top_10_pred) & set(top_10_true)) / 10 * 100
    print(f'Top 10 match: {top_10_match:.2f}%')

    top_10_metrics_path = 'data/top_10_metrics.csv'
    if it == 0:
        with open(top_10_metrics_path, 'w', newline='') as f:
            csv.writer(f).writerow(['Iteration', 'Percentage'])
    with open(top_10_metrics_path, 'a', newline='') as f:
        csv.writer(f).writerow([it, top_10_match])

    # Predict on pool
    preds, stds = gp.predict(data_array_x_pool_other, return_std=True)
    y_true = data_array_y_pool_other.ravel()
    preds_flat = preds.ravel()

    r2 = r2_score(y_true, preds_flat)
    mse = mean_squared_error(y_true, preds_flat)
    mae = mean_absolute_error(y_true, preds_flat)

    with open(learning_curve_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([it, len(data_array_x_pool_train), r2, 0.0, mse, 0.0, mae, 0.0])

    selected_points, selected_indices = propose_batch_greedy_index(
        upper_confidence_bound,
        data_array_x_pool_train,
        data_array_y_pool_train,
        gp,
        data_array_x_pool_other,
        batch_size=num_active_points
    )

    y_selected = gp.predict(selected_points)
    y_max_selected = max(y_selected)
    ei_selected = upper_confidence_bound(data_array_x_pool_other, gp)

    with open(f'data/points_to_simulate_iter_{it}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Index', 'Structure_Name'])
        for idx in selected_indices:
            writer.writerow([idx, structure_per_sample[idx_pool_other][idx]])

    with open(bayes_metrics_path, 'a', newline='') as f:
        writer = csv.writer(f)
        for i, idx in enumerate(selected_indices):
            writer.writerow([it, idx, structure_per_sample[idx_pool_other][idx], y_selected[i], y_max_selected, ei_selected[i]])

    for i in selected_indices:
        all_selected_points['True Values'].append(float(data_array_y_pool_other[i][0]))
        all_selected_points['Predicted Values'].append(float(preds[i]))
        all_selected_points['Std Dev'].append(float(stds[i]))

    pd.DataFrame(all_selected_points).to_csv(f'data/deviating_points_{it}.csv', index=False)

    structure_names_pool_other = structure_per_sample[idx_pool_other]

    if it == 0 or it % save_every == 0 or it == num_iterations - 1:
        plot_mean_std(y_true, preds_flat, stds, it, selected_indices)
        plot_parity(y_true, preds_flat, stds, selected_indices, it)
        plot_parity_interactive(y_true, preds_flat, stds, selected_indices, it, structure_names_pool_other)

    
    # Save data for this iteration
    data_save_path = f'data/data_iteration_{it}.csv'
    with open(data_save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"# Most uncertain indices: {list(selected_indices)}"])
        writer.writerow(['Index', 'Structure_Name', 'True Values', 'Predicted Values', 'Std Dev', 'Is_Selected', 'Split'])

        # Validation set
        for i, idx in enumerate(idx_pool_other):
            writer.writerow([
                idx,
                structure_per_sample[idx],
                float(y_true[i]),
                float(preds_flat[i]),
                float(stds[i]),
                int(i in selected_indices),
                'validation'
            ])

        # Training set
        y_train_pred, y_train_std = gp.predict(data_array_x_pool_train, return_std=True)
        for i, idx in enumerate(idx_pool_train):
            writer.writerow([
                idx,
                structure_per_sample[idx],
                float(data_array_y_pool_train[i][0]),
                float(y_train_pred[i]),
                float(y_train_std[i]),
                0,
                'train'
            ])


    selected_actual_indices = idx_pool_other[selected_indices]
    idx_pool_train = np.append(idx_pool_train, selected_actual_indices)
    idx_pool_other = np.delete(idx_pool_other, selected_indices)

    new_x = data_array_x_pool_other[selected_indices]
    new_y = data_array_y_pool_other[selected_indices]

    data_array_x_pool_train = np.vstack([data_array_x_pool_train, new_x])
    data_array_y_pool_train = np.vstack([data_array_y_pool_train, new_y])

    data_array_x_pool_other = np.delete(data_array_x_pool_other, selected_indices, axis=0)
    data_array_y_pool_other = np.delete(data_array_y_pool_other, selected_indices, axis=0)

# Final learning curve plot
plot_learning_curve()
