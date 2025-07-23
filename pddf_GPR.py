import os
import sys
import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from GPR_utils import (
    build_structure_per_sample,
    save_learning_curves_plot,
    plot_parity_interactive,
    plot_parity_detailed
)

from data_loading import(
    pdf_loading_01,
    MA_loading
)

# Data loading
main_folder_pdf = '/Users/sofiazinzani/Documents/Dottorato/Unimi/Daemon_Cost_Action/Structures/AuM_solo_pddf_01/AuM_Sub'
main_folder_ma  = '/Users/sofiazinzani/Documents/Dottorato/Unimi/Daemon_Cost_Action/Structures/Anisotropic/AuM_Sub/'

data_array_x, folders = pdf_loading_01(main_folder_pdf)
data_array_y = MA_loading(main_folder_ma, folders)
data_array_x = data_array_x[:, :198]

structure_per_sample = build_structure_per_sample(main_folder_ma, bin_index=0)
assert len(structure_per_sample) == data_array_x.shape[0]

scaler_x = StandardScaler()
data_array_x = scaler_x.fit_transform(data_array_x)

# Active learning setup
randoms = [2, 44, 55, 75]
fixed_random = 73
num_active_points = 10
num_iterations = 10
train_pool_number = 10
train_pool_size = 1 - train_pool_number / len(data_array_x)

all_indices = np.arange(data_array_x.shape[0])
idx_pool_train, idx_pool_other, data_array_x_pool_train, data_array_x_pool_other, data_array_y_pool_train, data_array_y_pool_other = train_test_split(
    all_indices, data_array_x, data_array_y, test_size=train_pool_size, random_state=fixed_random)

os.makedirs('data', exist_ok=True)
os.makedirs('plot', exist_ok=True)

learning_curve_path = 'data/learning_curves.csv'
with open(learning_curve_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Iteration', 'Num_Training_Samples', 'R2', 'R2_std', 'MSE', 'MSE_std', 'MAE', 'MAE_std'])

prediction_timing = []
deviating_points = {'True Values': [], 'Predicted Values': [], 'Pred Std Dev': []}

for it in range(num_iterations):
    print(f'Iteration: {it}\n{"_"*22}')
    all_model_preds = []
    all_model_stds = []
    r2_all, mse_all, mae_all = [], [], []
    models = {}

    for i, random in enumerate(randoms):
        X_train, X_val, y_train, y_val = train_test_split(data_array_x_pool_train, data_array_y_pool_train, test_size=0.2, random_state=random)
        kernel = C(0.1, (1e-3, 1e3)) * RBF(length_scale=2.0)
        model = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=5)
        model.fit(X_train, y_train)
        models[f'model_{i}'] = model

        start_time = time.time()
        predictions, std_devs = model.predict(data_array_x_pool_other, return_std=True)
        duration = time.time() - start_time

        prediction_timing.append({
            'Iteration': it,
            'Model': f'model_{i}',
            'Num_Samples': data_array_x_pool_other.shape[0],
            'Prediction_Time_Seconds': duration
        })

        mse = mean_squared_error(data_array_y_pool_other, predictions)
        r2 = r2_score(data_array_y_pool_other, predictions)
        mae = mean_absolute_error(data_array_y_pool_other, predictions)

        all_model_preds.append(predictions)
        all_model_stds.append(std_devs)
        r2_all.append(r2)
        mse_all.append(mse)
        mae_all.append(mae)

    all_model_preds = np.stack(all_model_preds)
    all_model_stds = np.stack(all_model_stds)
    mean_preds = np.mean(all_model_preds, axis=0).reshape(-1, 1)
    std_preds = np.mean(all_model_stds, axis=0).reshape(-1, 1)

    train_preds = np.mean([model.predict(data_array_x_pool_train) for model in models.values()], axis=0).reshape(-1, 1)
    train_stds = np.mean([model.predict(data_array_x_pool_train, return_std=True)[1] for model in models.values()], axis=0).reshape(-1, 1)

    most_uncertain_indices = np.argsort(std_preds[:, 0])[-num_active_points:][::-1]
    deviating_points['True Values'].extend(data_array_y_pool_other[most_uncertain_indices, 0])
    deviating_points['Predicted Values'].extend(mean_preds[most_uncertain_indices, 0])
    deviating_points['Pred Std Dev'].extend(std_preds[most_uncertain_indices, 0])

    # Save predictions and plots
    if it == 0 or it == num_iterations - 1:
        df_deviating = pd.DataFrame(deviating_points)
        df_deviating.to_csv(f'data/deviating_points_{it}.csv', index=False)

        df_plot = pd.DataFrame({
            'True Value': data_array_y_pool_other[:, 0],
            'Predicted Value': mean_preds[:, 0],
            'Uncertainty (±1σ)': std_preds[:, 0],
            'Structure': structure_per_sample[idx_pool_other]
        })
        plot_parity_interactive(df_plot, it, f'plot/interactive_parity_plot_iteration_{it}.html')

        idx_all = idx_pool_other
        idx_uncertain = most_uncertain_indices
        idx_deviating = np.arange(len(deviating_points['True Values']))

        structure_all = structure_per_sample[idx_all]
        structure_uncertain = structure_per_sample[idx_pool_other][idx_uncertain]
        structure_deviating = structure_per_sample[np.concatenate([idx_pool_train, idx_pool_other])][:len(idx_deviating)]

        plot_parity_detailed(data_array_y_pool_other, mean_preds, std_preds, deviating_points,
                             structure_all, structure_uncertain, structure_deviating, idx_uncertain,
                             it, f'plot/interactive_parity_plot_deviating_iteration_{it}.html')

    # Update training set with most uncertain samples
    new_x = data_array_x_pool_other[most_uncertain_indices]
    new_y = data_array_y_pool_other[most_uncertain_indices]
    data_array_x_pool_train = np.vstack([data_array_x_pool_train, new_x])
    data_array_y_pool_train = np.vstack([data_array_y_pool_train, new_y])
    data_array_x_pool_other = np.delete(data_array_x_pool_other, most_uncertain_indices, axis=0)
    data_array_y_pool_other = np.delete(data_array_y_pool_other, most_uncertain_indices, axis=0)
    idx_pool_train = np.concatenate([idx_pool_train, idx_pool_other[most_uncertain_indices]])
    idx_pool_other = np.delete(idx_pool_other, most_uncertain_indices)

    with open(learning_curve_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            it, len(idx_pool_train),
            np.mean(r2_all), np.std(r2_all),
            np.mean(mse_all), np.std(mse_all),
            np.mean(mae_all), np.std(mae_all)
        ])

# Save timing and learning curves
pd.DataFrame(prediction_timing).to_csv('data/prediction_timing.csv', index=False)
learning_df = pd.read_csv(learning_curve_path)
save_learning_curves_plot(learning_df)
