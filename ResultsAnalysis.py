import glob
import json
import os
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 5000)


def read_files():
    results_path = "OUTPUTS/results"
    results_path = Path(results_path)
    pattern = os.path.join(results_path, '**', '*.json')
    json_files = glob.glob(pattern, recursive=True)
    jsons_contents = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            try:
                json_data = json.load(f)
                json_data['file'] = os.path.basename(json_file)
                jsons_contents.append(json_data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {json_file}: {e}")
    return pd.DataFrame(jsons_contents)

def sort_df(df):
    # Define a custom order for the "model" column
    model_order = ['SAND', 'AE-LSTM', 'EncDec-AD', 'EncDec-AD-Batch', 'Online_EncDec-AD']
    model_order_dict = {model: idx for idx, model in enumerate(model_order)}
    # Extract the first element of the list in the "datasets" column for sorting
    df['datasets_sort_key'] = df['datasets'].apply(lambda x: x[0] if x else '')
    # Map the "model" column to the custom order and add it as a new column
    df['model_order'] = df['model'].map(model_order_dict)
    # Sort by the extracted sort key and the custom order
    df = df.sort_values(by=['datasets_sort_key', 'model_order']).drop(['datasets_sort_key', 'model_order'], axis=1)
    return df

def clean_up_df(df: pd.DataFrame):
    # append simplified dataset name column
    def extract_datasets_names(datasets_full):
        datasets_full = [dataset[0:dataset.find('/')] for dataset in datasets_full]
        return "_".join(datasets_full)
    datasets_col = df['datasets'].copy()
    simplified_datasets_col = df['datasets'].apply(lambda x: extract_datasets_names(x))
    df['datasets'] = simplified_datasets_col
    df['datasets_full'] = datasets_col
    df.drop(columns=['online','window'], inplace=True)

    # reorder columns
    columns = ['datasets'] + [col for col in df.columns if col not in ['datasets','normality']] + ['normality']
    df = df[columns]
    # rename offline enc dec
    df['model'] = df['model'].replace('EncDec-AD', 'Offline-EncDec-AD') \
                             .replace('OnlineEncDecAD', 'Online-EncDec-AD')
    return df


def add_average_values_per_model(df):
    avg = df.groupby(by=['model']).mean().reset_index()
    avg['datasets'] = ["Average"] * avg.shape[0]
    avg = avg[['datasets'] + [col for col in avg.columns if col != 'datasets']]
    avg = sort_df(avg)
    df = pd.concat([df, avg], ignore_index=True)
    return df


def print_sep_df_per_dataset(df: pd.DataFrame):
    datasets = df['datasets'].unique()
    for d in datasets:
        ddf = df[df['datasets'] == d]
        print(ddf)
        print("\n", "-"*200)
    print()


def get_results_for_datasets_with_all_models (df, num_of_models = 4):
    datasets_with_all_results = df[['datasets', 'model']].groupby(by=['datasets']).count().reset_index()
    datasets_with_all_results = datasets_with_all_results[datasets_with_all_results['model'] >= num_of_models].reset_index(drop=True)
    datasets_with_all_results = set(datasets_with_all_results['datasets'].values)
    print("selected datasets # ", len(datasets_with_all_results))
    selected_results_df = df[df['datasets'].isin(datasets_with_all_results)]
    return selected_results_df


def get_datasets_x_model_pivot_table(df, metrics=None, print_=True, save=True):

    if metrics is None:
        metrics = ['AUC', 'F']

    model_order = ['SAND', 'Offline-EncDec-AD', 'EncDec-AD-Batch', 'Online-EncDec-AD', 'AE-LSTM']
    model_order_dict = {model: idx for idx, model in enumerate(model_order)}
    df_copy = df.copy()
    df_copy['model'] = df_copy['model'].apply(lambda x: f'{model_order_dict[x]}. {x}')
    for metric in ['AUC', 'F', 'Precision', 'Recall']:
        df_copy[metric] = df_copy[metric] * 100

    pivot_df = df_copy[['datasets', 'model'] + metrics].pivot(index="datasets", columns="model")
    pivot_df.columns = pd.MultiIndex.from_tuples([(metric, model) for model, metric in pivot_df.columns])
    pivot_df = pivot_df.sort_index(axis=1)
    pivot_df.reset_index(inplace=True)

    # Custom sort: keep 'Average' row last
    pivot_df = pivot_df.sort_values(by='datasets')
    average_row = pivot_df[pivot_df['datasets'] == 'Average']
    pivot_df = pivot_df[pivot_df['datasets'] != 'Average']
    pivot_df = pd.concat([pivot_df, average_row])

    pivot_df = pivot_df.applymap(lambda x: float('{:.2f}'.format(x)) if isinstance(x, float) else x)
    if print_:
        print(pivot_df)
    # print(pivot_df.to_string(index=False), "\n")

    # Save to Excel
    pivot_df.columns = ['_'.join(col) if isinstance(col, tuple) and 'datasets' not in col
                        else ''.join(col)  for col in pivot_df.columns]
    if save:
        output_file = Path(f"OUTPUTS/Norm_{int(df['normality'].values[0])}_pivot.xlsx")
        pivot_df.to_excel(output_file, index=False, engine="openpyxl")
    return pivot_df




def make_boxplot(df, figure_name = "", metrics = None):
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 4, figsize=(13, 4.5))
    fig.suptitle(figure_name)
    # Plot each metric
    if metrics is None:
        metrics = ['AUC', 'F', 'Recall', 'Precision']
    titles = [metric if metric != 'F' else 'F1 Score' for metric in metrics]
    model_order = ['SAND', 'Offline-EncDec-AD', 'EncDec-AD-Batch', 'Online-EncDec-AD', 'AE-LSTM']
    model_short_names = ['SAND', 'Offline', 'Batch', 'Online', 'AE-LSTM']
    palette = sns.color_palette('deep',5)

    for ax, metric, title in zip(axes, metrics, titles):

        # TODO: should drop 0 values for plot? (SAND results are the most problematic -> completely different plot)
        df_without_zeros = df[(df[metric] > 0.) | (df['model'] != 'SAND')]

        sns.boxplot(x='model', y=metric, data=df_without_zeros, ax=ax, order=model_order, width=0.8, palette=palette)
        ax.set_title(title)
        ax.set_xlabel('Model')
        ax.set_ylabel(None)
        ax.tick_params(axis='x', rotation=20)
        ax.set_xticklabels(model_short_names)

    # Adjust layout
    plt.tight_layout()
    plt.ion()
    plt.show(block=False)


# ============================================================================================================

def prepare_results_df(normality, num_of_models = 4, filter_results = False):
    df = read_files()
    df = df[df['normality'] == normality]
    df = sort_df(df)
    df = clean_up_df(df)
    if filter_results:
        df = get_results_for_datasets_with_all_models(df, num_of_models)
    df = add_average_values_per_model(df)

    return df


if __name__ == "__main__":
    df = prepare_results_df(3, 5, False)
    print_sep_df_per_dataset(df)
    pivot_map = get_datasets_x_model_pivot_table(df)
    make_boxplot(df)
    # correlations(pivot_map)



