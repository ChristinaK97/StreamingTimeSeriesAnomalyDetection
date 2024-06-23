import warnings

import numpy as np
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')

from pathlib import Path
from typing import List, Union, Tuple

import pandas as pd

base_path = "TSB-UAD-Public"

# ===========================================================================================================
# Load datasets
# ===========================================================================================================

def load_dataset(filenames: List[str], sample_size=100000, plot_series=True, vebrose=True) -> Tuple[
    pd.DataFrame, int, List[int]]:
    df, normality, shift_idxs = generate_dataset(filenames, sample_size=sample_size)

    if vebrose:
        print(f"Series len = {df.shape[0]}\nNormality = {normality}\nDistro shift indexes = {shift_idxs}")
    if plot_series:
       plot_generated_dataset(df, plot_around_distro_shifts=True, plot_range=5000, shift_idxs=shift_idxs)

    return df, normality, shift_idxs



def read_dataset_as_df(filepath):
    df = pd.read_csv(filepath, header=None).dropna()
    df[0] = df[0].astype(float) # assuming univariate
    df[1] = df[1].astype(int)   # label
    return df


def generate_dataset(filenames: List[str], sample_size: Union[int, float] = None):
    # Read
    filenames = [Path(base_path, fp) for fp in filenames]
    normality = len(filenames)
    datasets = [read_dataset_as_df(fp) for fp in filenames]

    # Sampling
    if sample_size is not None:
        if sample_size <= 1:  # sample size was defined as a percentage of the original series' size
            sample_size = [int(dataset.shape[0] * sample_size) for dataset in datasets]
        else:  # sample size was defined as maximun number of points
            sample_size = [min(dataset.shape[0], sample_size) for dataset in datasets]

        for i, (dataset_sample_size, dataset) in enumerate(zip(sample_size, datasets)):
            datasets[i] = dataset[:dataset_sample_size]

    # Concatenate
    shift_idxs = []
    if len(datasets) > 1:
        cumulative_length = 0
        for i, dataset in enumerate(datasets[:-1]):
            cumulative_length += len(dataset)
            shift_idxs.append(cumulative_length)

    # Concatenate the list of DataFrames along rows (axis=0)
    merged_dataset = pd.concat(datasets, ignore_index=True)
    return merged_dataset, normality, shift_idxs



# ===========================================================================================================
# Plot time series
# ===========================================================================================================

def plot_generated_dataset(df: pd.DataFrame,
                           plot_around_distro_shifts=False,
                           plot_range=1000,
                           shift_idxs: List[int] = None):
    # Extract values from DataFrame
    if plot_around_distro_shifts and shift_idxs is not None and len(shift_idxs) > 0:
        for shift_idx in shift_idxs:  # different plot for each distro shift
            x_values = np.arange(shift_idx - plot_range, shift_idx + plot_range + 1)
            df_partial = df.loc[x_values]
            _plot_generated_dataset(df_partial, x_values, [shift_idx])

    else:
        x_values = np.arange(len(df))  # Use numerical index as x-values
        _plot_generated_dataset(df, x_values, shift_idxs)


def _plot_generated_dataset(df: pd.DataFrame,
                            x_values,
                            shift_idxs=None):
    y_values = df[0].values  # First column ('0') values as y-values
    color_values = df[1].values  # Second column ('1') values for coloring (0 or 1)

    # Plotting the time series with conditional coloring
    plt.figure(figsize=(12, 4))  # Set the figure size (width, height)

    # Plot the entire time series as a line plot
    plt.plot(x_values, y_values, color='black', linewidth=2)

    # Initialize segment start index and color
    segment_start = 0
    current_color = color_values[0]

    # Iterate over data points to plot line segments with different colors
    for i in range(1, len(df)):
        if color_values[i] != current_color:
            # Plot the previous segment with the current color
            plt.plot(x_values[segment_start:i], y_values[segment_start:i],
                     color='red' if current_color == 1 else 'black', linewidth=2)
            # Update segment start index and current color
            segment_start = i
            current_color = color_values[i]

    # Plot the last segment
    plt.plot(x_values[segment_start:], y_values[segment_start:], color='red' if current_color == 1 else 'black',
             linewidth=2)

    for shift_idx in shift_idxs:
        plt.axvline(shift_idx, color='blue', linestyle='--', linewidth=1.5)

    # Display the plot
    plt.show()




# ===========================================================================================================
# Get information/statistics about the datasets
# ===========================================================================================================

def get_1_normality_info(ALL_SETS, sample_size):
    data_info = []
    for s in ALL_SETS:
        df, _, _ = load_dataset([s], sample_size=sample_size, plot_series=False, vebrose=False)
        length = df.shape[0]
        n_labels = df.groupby(by=[1]).count().reset_index()
        try:
            n_anomalies = n_labels[n_labels[1] == 1][0].values[0]
        except IndexError:
            n_anomalies = 0
        n_normalpoi = n_labels[n_labels[1] == 0][0].values[0]
        anomaly_rate = round((n_anomalies / float(length)) * 100, 2)
        positive_rate = round((n_normalpoi / float(length)) * 100, 2)
        data_info.append({
            "datasets": s[0:s.find('/')],
            "length": length,
            "# normal": n_normalpoi,
            "positive rate": positive_rate,
            "# anomalies": n_anomalies,
            "anomaly rate": anomaly_rate,
        })
    data_info = pd.DataFrame(data_info).sort_values(by=["datasets"])
    return data_info



def get_k_normality_info(ALL_SETS, k_normality_filenames: List[List[str]], sample_size, aggregate=False, split_columns = False):
    data_info = get_1_normality_info(ALL_SETS, sample_size)

    if not isinstance(k_normality_filenames[0], list):
        k_normality_filenames = [s.split("_") for s in k_normality_filenames]
    if "/" in k_normality_filenames[0][0]:
        k_normality_filenames = [[ si[:si.find("/")] for si in s] for s in k_normality_filenames]

    s_infos = []
    for s in k_normality_filenames:
        s_info = {f"si {col}" if col!='datasets' else col : [] for col in data_info.columns}
        for si in s:
            si_info = data_info[data_info['datasets'] == si]
            s_info["datasets"].append(si)
            for col in data_info.columns:
                if col != "datasets":
                    s_info[f"si {col}"].append(si_info[col].values[0])
        s_info['datasets'] = '_'.join(s_info['datasets'])
        s_infos.append(s_info)
    data_info = pd.DataFrame(s_infos).sort_values(by=["datasets"])

    if aggregate:
        for col in ['si length', 'si # normal', 'si # anomalies']:
            data_info[col[3:]] = data_info[col].apply(lambda x: sum(x))
        data_info['anomaly rate'] = round((data_info['# anomalies'] / data_info['length']) * 100, 2)
        data_info['rate diff'] = data_info['si anomaly rate'].apply(lambda rates: round(rates[-1] - rates[0], 2))
    if split_columns:
        data_info = split_list_columns(data_info)

    return data_info



def split_list_columns(df):
    # Identify columns that contain lists
    list_cols = [col for col in df.columns if col.startswith('si')]
    new_column_names = []

    for col in list_cols:
        # Get the maximum length of the lists in the column
        max_length = df[col].apply(len).max()

        # Create new columns for each element in the lists
        for i in range(max_length):
            new_col_name = f's{i + 1} {col[2:]}'  # Name new columns as s1, s2, etc.
            df[new_col_name] = df[col].apply(lambda x: x[i] if i < len(x) else None)
            new_column_names += new_col_name

        # Drop the original list column
        df.drop(columns=[col], inplace=True)
    return df

