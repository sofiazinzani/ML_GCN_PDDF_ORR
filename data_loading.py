import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# Funzione per ordinamento alfanumerico
def alphanumeric_sort(element):
    return [int(c) if c.isdigit() else c.lower() for c in re.split('([0-9]+)', element)]

def gcn_loading(main_folder):
    feature_list = []

    folders = sorted(
        [name for name in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, name)) and '-' in name],
        key=alphanumeric_sort)

    for folder in folders:
        folder_path = os.path.join(main_folder, folder)
        subdirs = sorted([name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))], key=alphanumeric_sort)

        for subdir in subdirs:
            subdir_path = os.path.join(folder_path, subdir)
            sub_sub_dirs = sorted([name for name in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, name))], key=alphanumeric_sort)

            for sub_sub_dir in sub_sub_dirs:
                path = os.path.join(subdir_path, sub_sub_dir)
                gcn_path = os.path.join(path, "GCN/gcn_genome.dat")

                try:
                    feature = pd.read_csv(gcn_path, sep=r'\s+', comment='#', header=None)
                    if feature.shape[1] > 2:
                        feature_list.append(feature.iloc[:, 2].to_numpy())
                except Exception as e:
                    print(f"Error in file: {gcn_path} -> {e}")

    data_array_x = np.vstack(feature_list)
    return data_array_x, folders




def pdf_loading(main_folder):
    dataframes = []

    folders = sorted(
        [name for name in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, name)) and '-' in name],
        key=alphanumeric_sort
    )

    for folder in folders:
        folder_path = os.path.join(main_folder, folder)
        subdirs = sorted(
            [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))],
            key=alphanumeric_sort
        )

        for subdir in subdirs:
            subdir_path = os.path.join(folder_path, subdir)
            sub_sub_dirs = sorted(
                [name for name in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, name))],
                key=alphanumeric_sort
            )

            for sub_sub_dir in sub_sub_dirs:
                path = os.path.join(subdir_path, sub_sub_dir)
                pdf_path = os.path.join(path, "Creation_Structures_Prop_df/pdf.dat")

                data = []
                try:
                    with open(pdf_path, 'r') as file:
                        for _ in range(3): next(file)

                        for line in file:
                            if line.startswith('#') or not line.strip():
                                continue
                            tokens = line.strip().split()
                            try:
                                tokens = [float(x) if '.' in x else int(x) for x in tokens]
                                data.append(tokens)
                            except ValueError:
                                continue
                except FileNotFoundError:
                    continue

                if data:
                    df = pd.DataFrame(data, columns=["bin upper value", "occurrence"])
                    dataframes.append(df)

    # Uniform lenght
    max_length = max(len(df) for df in dataframes)
    for i, df in enumerate(dataframes):
        num_missing = max_length - len(df)
        if num_missing > 0:
            last_bin = df["bin upper value"].iloc[-1]
            extra_bins = last_bin + np.arange(1, num_missing + 1) * 0.01
            padding_df = pd.DataFrame({
                "bin upper value": extra_bins,
                "occurrence": np.zeros(num_missing)
            })
            dataframes[i] = pd.concat([df, padding_df], ignore_index=True)

    feature_list = [df["occurrence"].to_numpy() for df in dataframes]
    data_array_x = np.vstack(feature_list)

    return data_array_x, folders



def pdf_loading_05(main_folder):
    dataframes = []

    folders = sorted(
        [name for name in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, name)) and '-' in name],
        key=alphanumeric_sort
    )

    for folder in folders:
        folder_path = os.path.join(main_folder, folder)
        subdirs = sorted(
            [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))],
            key=alphanumeric_sort
        )

        for subdir in subdirs:
            subdir_path = os.path.join(folder_path, subdir)
            sub_sub_dirs = sorted(
                [name for name in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, name))],
                key=alphanumeric_sort
            )

            for sub_sub_dir in sub_sub_dirs:
                path = os.path.join(subdir_path, sub_sub_dir)
                pdf_path = os.path.join(path, "Creation_Structures_Prop_df_05/pdf.dat")

                data = []
                try:
                    with open(pdf_path, 'r') as file:
                        for _ in range(3): next(file)

                        for line in file:
                            if line.startswith('#') or not line.strip():
                                continue
                            tokens = line.strip().split()
                            try:
                                tokens = [float(x) if '.' in x else int(x) for x in tokens]
                                data.append(tokens)
                            except ValueError:
                                continue
                except FileNotFoundError:
                    continue

                if data:
                    df = pd.DataFrame(data, columns=["bin upper value", "occurrence"])
                    dataframes.append(df)

    # Uniform lenght
    max_length = max(len(df) for df in dataframes)
    for i, df in enumerate(dataframes):
        num_missing = max_length - len(df)
        if num_missing > 0:
            last_bin = df["bin upper value"].iloc[-1]
            extra_bins = last_bin + np.arange(1, num_missing + 1) * 0.01
            padding_df = pd.DataFrame({
                "bin upper value": extra_bins,
                "occurrence": np.zeros(num_missing)
            })
            dataframes[i] = pd.concat([df, padding_df], ignore_index=True)

    feature_list = [df["occurrence"].to_numpy() for df in dataframes]
    data_array_x = np.vstack(feature_list)

    return data_array_x, folders




def pdf_loading_025(main_folder):
    dataframes = []

    folders = sorted(
        [name for name in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, name)) and '-' in name],
        key=alphanumeric_sort
    )

    for folder in folders:
        folder_path = os.path.join(main_folder, folder)
        subdirs = sorted(
            [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))],
            key=alphanumeric_sort
        )

        for subdir in subdirs:
            subdir_path = os.path.join(folder_path, subdir)
            sub_sub_dirs = sorted(
                [name for name in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, name))],
                key=alphanumeric_sort
            )

            for sub_sub_dir in sub_sub_dirs:
                path = os.path.join(subdir_path, sub_sub_dir)
                pdf_path = os.path.join(path, "Creation_Structures_Prop_df_025/pdf.dat")

                data = []
                try:
                    with open(pdf_path, 'r') as file:
                        for _ in range(3): next(file)

                        for line in file:
                            if line.startswith('#') or not line.strip():
                                continue
                            tokens = line.strip().split()
                            try:
                                tokens = [float(x) if '.' in x else int(x) for x in tokens]
                                data.append(tokens)
                            except ValueError:
                                continue
                except FileNotFoundError:
                    continue

                if data:
                    df = pd.DataFrame(data, columns=["bin upper value", "occurrence"])
                    dataframes.append(df)

    # Uniform lenght
    max_length = max(len(df) for df in dataframes)
    for i, df in enumerate(dataframes):
        num_missing = max_length - len(df)
        if num_missing > 0:
            last_bin = df["bin upper value"].iloc[-1]
            extra_bins = last_bin + np.arange(1, num_missing + 1) * 0.01
            padding_df = pd.DataFrame({
                "bin upper value": extra_bins,
                "occurrence": np.zeros(num_missing)
            })
            dataframes[i] = pd.concat([df, padding_df], ignore_index=True)

    feature_list = [df["occurrence"].to_numpy() for df in dataframes]
    data_array_x = np.vstack(feature_list)

    return data_array_x, folders



def pdf_loading_01(main_folder):
    dataframes = []

    folders = sorted(
        [name for name in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, name)) and '-' in name],
        key=alphanumeric_sort
    )

    for folder in folders:
        folder_path = os.path.join(main_folder, folder)
        subdirs = sorted(
            [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))],
            key=alphanumeric_sort
        )

        for subdir in subdirs:
            subdir_path = os.path.join(folder_path, subdir)
            sub_sub_dirs = sorted(
                [name for name in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, name))],
                key=alphanumeric_sort
            )

            for sub_sub_dir in sub_sub_dirs:
                path = os.path.join(subdir_path, sub_sub_dir)
                pdf_path = os.path.join(path, "Creation_Structures_Prop_df_01/pdf.dat")

                data = []
                try:
                    with open(pdf_path, 'r') as file:
                        for _ in range(3): next(file)

                        for line in file:
                            if line.startswith('#') or not line.strip():
                                continue
                            tokens = line.strip().split()
                            try:
                                tokens = [float(x) if '.' in x else int(x) for x in tokens]
                                data.append(tokens)
                            except ValueError:
                                continue
                except FileNotFoundError:
                    continue

                if data:
                    df = pd.DataFrame(data, columns=["bin upper value", "occurrence"])
                    dataframes.append(df)

    # Uniform lenght
    max_length = max(len(df) for df in dataframes)
    for i, df in enumerate(dataframes):
        num_missing = max_length - len(df)
        if num_missing > 0:
            last_bin = df["bin upper value"].iloc[-1]
            extra_bins = last_bin + np.arange(1, num_missing + 1) * 0.01
            padding_df = pd.DataFrame({
                "bin upper value": extra_bins,
                "occurrence": np.zeros(num_missing)
            })
            dataframes[i] = pd.concat([df, padding_df], ignore_index=True)

    feature_list = [df["occurrence"].to_numpy() for df in dataframes]
    data_array_x = np.vstack(feature_list)

    return data_array_x, folders
    
    
    


def MA_loading(main_folder, folders):
    labels_list = []
    count = 0

    for folder in folders:
        folder_path = os.path.join(main_folder, folder)
        subdirs = sorted(
            [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))],
            key=alphanumeric_sort
        )

        for subdir in subdirs:
            subdir_path = os.path.join(folder_path, subdir)
            sub_sub_dirs = sorted(
                [name for name in os.listdir(subdir_path) if os.path.isdir(os.path.join(subdir_path, name))],
                key=alphanumeric_sort
            )

            for sub_sub_dir in sub_sub_dirs:
                count += 1
                gcn_path = os.path.join(subdir_path, sub_sub_dir, "GCN/eta.dat")
                try:
                    label_df = pd.read_csv(gcn_path, sep=r'\s+', comment='#', header=None)
                    if label_df.shape[1] > 6:
                        labels_list.append(label_df.iloc[:, 7].to_numpy())
                    else:
                        print(f"Columns not sufficient: {gcn_path}")
                except FileNotFoundError:
                    print(f"File not found: {gcn_path}")

    data_array_y = np.vstack(labels_list)
    return data_array_y

