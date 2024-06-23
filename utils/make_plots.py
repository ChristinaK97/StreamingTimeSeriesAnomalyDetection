import re
import warnings

import numpy as np
from matplotlib import pyplot as plt

from utils.handle_results_files import prepare_results_df, print_sep_df_per_dataset

warnings.filterwarnings('ignore')

from typing import List

import pandas as pd
import seaborn as sns

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 5000)



# ===========================================================================================================
# Model comparison (evaluation metrics)
# ===========================================================================================================


def boxplots_comparing_models(df, figure_name ="", metrics = None, ablation_results=False):
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.5))
    fig.suptitle(figure_name)
    # Plot each metric
    if metrics is None:
        metrics = ['AUC', 'F', 'Recall', 'Precision']
    titles = [metric if metric != 'F' else 'F1 Score' for metric in metrics]
    if ablation_results:
        model_order = ['Default', 'w/o DAT', 'w/o IL', 'w/o CDD']
        model_short_names = model_order
    else:
        model_order = ['SAND', 'Offline-EncDec-AD', 'EncDec-AD-Batch', 'Online-EncDec-AD', 'AE-LSTM']
        model_short_names = ['SAND', 'Offline', 'Batch', 'Online', 'AE-LSTM']
    palette = sns.color_palette('deep',5)

    for ax, metric, title in zip(axes, metrics, titles):

        # TODO: should drop 0 values for plot? (SAND results are the most problematic -> completely different plot)
        df_without_zeros = df[(df[metric] > 0.) | (df['model'] != 'SAND')]

        sns.boxplot(x='model', y=metric, data=df_without_zeros, ax=ax, order=model_order, width=0.8,
                    palette=palette, showfliers=not ablation_results)
        ax.set_title(title)
        ax.set_xlabel('Model')
        ax.set_ylabel(None)
        ax.tick_params(axis='x', rotation=20)
        ax.set_xticklabels(model_short_names)

    # Adjust layout
    plt.tight_layout()
    plt.ion()
    plt.show(block=False)
    plt.savefig(f"OUTPUTS/figures/boxplot {re.sub(r'[^A-Za-z0-9_ ]', '', figure_name)} abl {str(ablation_results)}.png", dpi=300)


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
    plt.savefig(f"OUTPUTS/figures/conf mat {conf_mat[0][0]} {conf_mat[0][0]} {conf_mat[1][0]} {conf_mat[1][1]}.png", dpi=300)


# ===========================================================================================================
# Model comparison (Runtime)
# ===========================================================================================================

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
    plt.savefig(f"OUTPUTS/figures/boxplot time.png", dpi=300)


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
                annot=annot, fmt='.1f', square=True, cbar=False,
                yticklabels=model_short_names, xticklabels=[1, 2, 3])
    print(annot)
    plt.title('Average Runtime Heatmap')
    plt.ylabel('Model')
    plt.xlabel('Normality')
    plt.ion()
    plt.show(block=False)
    plt.savefig(f"OUTPUTS/figures/heatmap time.png", dpi=300)



# ===========================================================================================================
# Correlations between dataset characteristics and model performance
# ===========================================================================================================

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
    plt.savefig(f"OUTPUTS/figures/correlations n_charact {n_characterists}.png", dpi=300)



# ===========================================================================================================
# OnlineEncDec-AD Ablation Study
# ===========================================================================================================

def get_ablation_study_results():
    ablation_results = prepare_results_df(normality=None, load_ablation_study_results=True)
    ablation_results = ablation_results[ablation_results['datasets']!='Average'].drop(columns=['model'])
    ablation_results['model'] = ablation_results[['use_increm_learn', 'use_drift_detect', 'use_dynamic_thrs']].apply(
                    lambda row: 'w/o ' + row.index[~row.astype(bool)].tolist()[0], axis=1)
    ablation_results = ablation_results[['datasets','model'] + [col for col in ablation_results.columns if col not in {'datasets','model'}]]
    ablation_results['model'].replace({'w/o use_increm_learn':'w/o IL', 'w/o use_drift_detect':'w/o CDD', 'w/o use_dynamic_thrs':'w/o DAT'}, inplace=True)


    default_results = prepare_results_df(normality=None)
    default_results = default_results[(default_results['model']=='Online-EncDec-AD') & (default_results['datasets']!='Average')]
    default_results[['ablation', 'use_dynamic_thrs', 'model']] = False, True, 'Default'
    default_results = default_results[['datasets','model'] + [col for col in ablation_results.columns if col not in {'datasets','model'}]]
    default_results = default_results[default_results['datasets'].isin(ablation_results['datasets'].unique())]

    results = pd.concat([ablation_results, default_results])\
                .sort_values(by=['normality','datasets','model'])
    avg = results.groupby(by=['normality','model']).mean().reset_index()
    print("Average:\n", avg, sep="")
    return results, avg




def show_ablation_study_results_for_normality(abl_study_res, normality):
    norm_results = abl_study_res[abl_study_res['normality'] == normality]
    print_sep_df_per_dataset(norm_results)
    boxplots_comparing_models(norm_results, "Ablation Study", ablation_results=True)
    ablation_study_heatmap(norm_results, 'AUC')
    ablation_study_heatmap(norm_results, 'F')




def ablation_study_heatmap(abl_study_res, metric):
    results = {f: [] for f in ['Default', 'w/o DAT', 'w/o IL', 'w/o CDD']}
    datasets = sorted(abl_study_res['datasets'].unique())
    for feature in results.keys():
        for dataset in datasets:
            df = abl_study_res[abl_study_res['datasets'] == dataset]
            try:
                value = float(df[df['model'] == feature][metric]) * 100
            except:
                value = 0.0
            results[feature].append(value)

    results = pd.DataFrame(results.values(), columns=datasets, index=results.keys())
    annot = results.copy()
    for dataset in results.columns:
        values = results[dataset].copy()
        n_remain = len(abl_study_res['model'].unique())
        rank = n_remain
        while n_remain > 0:
            max_value = np.max(values)
            max_value_count = 0
            for model_idx, value in enumerate(values):
                if value == max_value:
                    results[dataset].iloc[model_idx] = rank
                    values[model_idx] = float('-inf')
                    n_remain -= 1
                    max_value_count += 1
            rank -= max_value_count

    plt.figure()

    def abbreviate_datasets(datasets):
        return '_'.join([ si[ : (min(6,len(si)) if len(datasets)>12 else len(si)) ] for si in datasets.split('_')])
    results.columns = results.columns.map(abbreviate_datasets)

    sns.heatmap(results, cmap=sns.dark_palette("#69d", reverse=True, as_cmap=True),
                annot=annot, fmt='.2f', square=True, cbar=False)
    plt.xticks(rotation=60)
    plt.title(f'{metric} scores per dataset')
    plt.tight_layout()
    plt.ion()
    plt.show(block=False)
    plt.savefig(f"OUTPUTS/figures/heatmap ablation {metric}.png", dpi=300)


# ===========================================================================================================
# ===========================================================================================================

