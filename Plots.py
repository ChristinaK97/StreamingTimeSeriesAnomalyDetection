import warnings

import numpy as np
from matplotlib import pyplot as plt

from ResultsAnalysis import prepare_results_df, get_datasets_x_model_pivot_table, make_boxplot

warnings.filterwarnings('ignore')

import os, sys
from os import path
from pathlib import Path
from typing import List, Union, Tuple

import pandas as pd
import seaborn as sns

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 5000)


base_path = "TSB-UAD-Public"
DODGERS = 'Dodgers/101-freeway-traffic.test.out'
ECG = 'ECG/MBA_ECG803_data.out'
IOPS = 'IOPS/KPI-1c6d7a26-1f1a-3321-bb4d-7a9d969ec8f0.test.out'
KDD21 = 'KDD21/001_UCR_Anomaly_DISTORTED1sddb40_35000_52000_52620.out'
MGAB = 'MGAB/1.test.out'
NAB = 'NAB/NAB_data_Traffic_1.out'
SENSORSCOPE = 'SensorScope/stb-2.test.out'
YAHOO = 'YAHOO/Yahoo_A1real_1_data.out'
DAPHNET = 'Daphnet/S01R02E0.test.csv@1.out'
GHL = 'GHL/01_Lev_fault_Temp_corr_seed_11_vars_23.test.csv@4.out'
GENESIS = 'Genesis/genesis-anomalies.test.csv@1.out'
MITDB = 'MITDB/100.test.csv@1.out'
OPPORTUNITY = 'OPPORTUNITY/S1-ADL1.test.csv@16.out'
OCCUPANCY = 'Occupancy/room-occupancy.train.csv@3.out'
SMD = 'SMD/machine-1-1.test.csv@1.out'
SVDB = 'SVDB/801.test.csv@1.out'

ALL_SETS = [DODGERS, ECG, IOPS, KDD21, MGAB, NAB, SENSORSCOPE, YAHOO, DAPHNET, GHL, GENESIS, MITDB, OPPORTUNITY,
            OCCUPANCY, SMD, SVDB]

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


def load_dataset(filenames: List[str], sample_size=100000, plot_series=True, vebrose=True) -> Tuple[
    pd.DataFrame, int, List[int]]:
    df, normality, shift_idxs = generate_dataset(filenames, sample_size=sample_size)

    if vebrose:
        print(f"Series len = {df.shape[0]}\nNormality = {normality}\nDistro shift indexes = {shift_idxs}")
    #if plot_series:
       # plot_generated_dataset(df, plot_around_distro_shifts=True, plot_range=5000, shift_idxs=shift_idxs)

    return df, normality, shift_idxs



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



def get_k_normality_info(ALL_SETS, k_normality_filenames: List[List[str]], sample_size, aggregate=False, split_columns = False):
    data_info = get_1_normality_info(ALL_SETS, sample_size)

    if not isinstance(k_normality_filenames[0], list):
        k_normality_filenames = [s.split("_") for s in k_normality_filenames]
    if "/" in k_normality_filenames[0]:
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


def correlations(pivot_mat, n_characterists):

    corr_mat = pivot_mat.corr()
    corr_mat = corr_mat[corr_mat.columns[0:corr_mat.shape[0] - n_characterists]].iloc[corr_mat.shape[0] - n_characterists:]

    plt.figure(figsize=(12,10))  # Adjust the figure size as needed
    min_value = corr_mat.min().min()
    max_value = corr_mat.max().max()
    corr_mat.columns = corr_mat.columns.str.replace('EncDec-AD', '')
    corr_mat.index = corr_mat.index.str.replace('EncDec-AD', '')
    sns.heatmap(corr_mat, annot=True, cmap='coolwarm', vmin=min_value, vmax=max_value, fmt='.2f', linewidths=.5, square=n_characterists<=5)
    plt.tick_params(axis='x', rotation=20)
    plt.title('Correlation Heatmap')
    plt.ion()
    plt.show(block=False)

def confusion_mat(conf_mat: List[List[int]]):
    df_cm = pd.DataFrame(conf_mat, index=['True Negative', 'True Positive'],
                         columns=['Predicted Negative', 'Predicted Positive'])

    plt.figure(figsize=(7, 7))
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g', cmap='Blues')

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Gold Labels')
    plt.ion()
    plt.show(block=False)



normality = 3
results_df = prepare_results_df(normality, 5, False)
avg_values = results_df[results_df['datasets']=='Average']
print(avg_values)
results_df = results_df[results_df['datasets']!='Average']

pivot_map = get_datasets_x_model_pivot_table(results_df, save=False, print_=False).reset_index(drop=True)

data_info = get_1_normality_info(ALL_SETS, 100000) if normality == 1 else \
            get_k_normality_info(ALL_SETS, pivot_map[['datasets']].values.flatten(), 50000, True, False)
n_characteristics = len([col for col in data_info.columns if not col.startswith('si')]) - 1
print(data_info, "\n\n")

merged_results = pd.merge(results_df, data_info, on='datasets')
merged_pivot = pd.merge(pivot_map, data_info, on='datasets')

make_boxplot(results_df, "Performance across all datasets")

if normality == 1:
    make_boxplot(merged_results[merged_results['anomaly rate'] < 3],  "Series with low anomaly rate (<3)")
    make_boxplot(merged_results[merged_results['anomaly rate'] >= 3], "Series with high anomaly rate (>=3)")
    pass
else:
    make_boxplot(merged_results[merged_results['rate diff'] < 0], "Series where the anomaly rate decreases")
    make_boxplot(merged_results[merged_results['rate diff'] > 0], "Series where the anomaly rate increases")

    sorted_merged_pivot = pd.concat([
        merged_pivot[merged_pivot['rate diff'] < 0].sort_values(by=['rate diff'], ascending=False),
        merged_pivot[merged_pivot['rate diff'] > 0].sort_values(by=['rate diff'])
    ])



correlations(merged_pivot, n_characteristics)

# conf_mat = merged_results[(merged_results['datasets']=='Occupancy_SVDB') & (merged_results['model']=='Online-EncDec-AD')]['confusion_matrix'].values[0]


def make_runtime_boxplot(df):
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 3, figsize=(10, 6))  # Create subplots for normality 1, 2, 3
    normalities = [1.0, 2.0, 3.0]
    titles = ['Normality 1', 'Normality 2', 'Normality 3']
    model_order = ['SAND', 'Offline-EncDec-AD', 'EncDec-AD-Batch','Online-EncDec-AD', 'AE-LSTM']
    model_short_names = ['SAND', 'Offline', 'Batch', 'Online', 'AE-LSTM']
    palette = sns.color_palette('deep', 5)

    for i, (ax, normality, title) in enumerate(zip(axes, normalities, titles)):
        # Filter the dataframe for the current normality
        df_normality = df[df['normality'] == normality]

        # Create the boxplot
        sns.boxplot(x='model', y='time', data=df_normality, ax=ax, order=model_order,
                    width=0.8, palette=palette, showfliers=False)

        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel('Model')
        ax.set_ylabel('Time (s)' if i == 0 else None)
        ax.tick_params(axis='x', rotation=20)
        ax.set_xticklabels(model_short_names)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.ion()
    plt.show(block=False)

def make_avg_runtime_heatmap(avg_runtimes):
    results = {f"normality {k}": [] for k in range(1, 4)}
    model_order = ['SAND', 'Offline-EncDec-AD', 'EncDec-AD-Batch', 'Online-EncDec-AD', 'AE-LSTM']
    model_short_names = ['SAND', 'Offline', 'Batch', 'Online', 'AE-LSTM']

    for normality in results.keys():
        for model in model_order:
            df = avg_runtimes[avg_runtimes['model']== model]
            value = round(float(df[df['normality'] == float(normality[-1])]['time'].values[0]), 2)
            results[normality].append(value)

    results = pd.DataFrame(results.values(), columns=model_order, index=results.keys()).transpose()

    annot = results.copy()
    for normality in results.columns:
        values = results[normality].copy()
        for i in range(len(model_order)):
            min_model = np.argmin(values)
            results[normality].iloc[min_model] = i
            values[min_model] = float('inf')

    plt.figure()
    sns.heatmap(results, cmap=sns.dark_palette("#69d", reverse=True, as_cmap=True),
                annot=annot, fmt='.1f', square=True,
                yticklabels=model_short_names, xticklabels=[1, 2, 3])
    print(annot)
    plt.title('Average Runtime Heatmap')
    plt.ylabel('Model')
    plt.xlabel('Normality')
    plt.ion()
    plt.show(block=False)
    plt.savefig('time heatmap.png', dpi=300)


def runtime_analysis():
    runtimes, avg_runtimes = [], []
    for normality in range(1, 4):
        norm_results_df = prepare_results_df(normality, 5, True)
        norm_runtime = norm_results_df[['datasets', 'normality', 'model', 'time']]

        norm_avg_runtime = norm_runtime[norm_runtime['datasets'] == 'Average']
        norm_runtime = norm_runtime[norm_runtime['datasets'] != 'Average']

        runtimes.append(norm_runtime)
        avg_runtimes.append(norm_avg_runtime)

    runtimes = pd.concat(runtimes)
    avg_runtimes = pd.concat(avg_runtimes)

    print(runtimes)
    print(avg_runtimes)
    make_runtime_boxplot(runtimes)
    make_avg_runtime_heatmap(avg_runtimes)



runtime_analysis()
plt.pause(1200)