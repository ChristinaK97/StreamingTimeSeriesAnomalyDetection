import glob
import json
import os
import re
import time
from pathlib import Path
from typing import List

import pandas as pd
from TSB_UAD.utils.slidingWindows import printResult

# ===========================================================================================================
# Save results
# ===========================================================================================================

output_path_base = os.path.join("OUTPUTS", "results")


def save_dict_to_json_file(dictionary, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(dictionary, json_file, indent=4, ensure_ascii=False)


def normalize_file_names(filenames: List[str]):
    filenames_norm = [filename[:filename.find('/') + 1] for filename in filenames]
    filenames_norm = "_".join(filenames_norm)
    filenames_norm = re.sub(r'[^A-Za-z0-9_]', '.', filenames_norm)
    return filenames_norm


def get_json_file_path(filenames: List[str], normality: int, model_name: str, online: bool):
    filenames_norm = normalize_file_names(filenames)
    output_json = os.path.join(output_path_base, f"normality_{normality}",
                    f"{filenames_norm}_{model_name}_{'online' if online else 'offline'}_{str(time.time())[-4:]}.json")
    return output_json


def results_file_exists(filenames: List[str], normality: int, model_name: str, online: bool):
    filenames_norm = normalize_file_names(filenames)
    output_json = pattern = os.path.join(output_path_base, f"normality_{normality}",
                                      f"{filenames_norm}_{model_name}_{'online' if online else 'offline'}_*.json")
    # print(output_json)
    return glob.glob(output_json)




# save TSB report to a json file
def gather_TSB_results(model_name, online, label, score, slidingWindow, filenames, normality, elapsed_time,
                       save_to_json=True, additional_stats=None):
    output_json = get_json_file_path(filenames, normality, model_name, online)
    results = printResult(None, label, score, slidingWindow, None, None)
    results = [float(value) for value in results]
    results = {
        "model": model_name,
        "online": online,
        "normality": normality,
        "datasets": filenames,
        "series_length": len(label),
        "window": int(slidingWindow),
        "AUC": results[0],
        "Precision": results[1],
        "Recall": results[2],
        "F": results[3],
        "Precision@k": results[9],
        "Rprecision": results[7],
        "Rrecall": results[4],
        "Rf": results[8],
        "ExistenceReward": results[5],
        "OverlapReward": results[6],
        "RAUC": results[10],
        "time": elapsed_time
    }
    if additional_stats is not None:
        results = {**results, **additional_stats}

    save_dict_to_json_file(results, output_json)
    return results


def gather_AELSTM_results(
        model_name: str, time_series_len: int, filenames: List[str], normality: int, AELSTMresults,
        label, scores, slidingWindow, elapsed_time, additional_stats=None, ablation_study=False):
    output_json = get_json_file_path(filenames, normality, model_name, True)
    if ablation_study:
        output_json = output_json.replace('results', 'ablation_study')
    TSBresults = printResult(None, label, scores, slidingWindow, None, None)
    TSBresults = [float(value) for value in TSBresults]

    results = {
        "model": model_name,
        "online": True,
        "normality": normality,
        "datasets": filenames,
        "series_length": time_series_len,
        "window": int(slidingWindow),
        "AUC": float(AELSTMresults['AUC']),
        "Precision": float(AELSTMresults['report']['anamoly']['precision']),
        "Recall": float(AELSTMresults['report']['anamoly']['recall']),
        "F": float(AELSTMresults['report']['anamoly']['f1-score']),
        "Precision@k": TSBresults[9],
        "Rprecision": TSBresults[7],
        "Rrecall": TSBresults[4],
        "Rf": TSBresults[8],
        "ExistenceReward": TSBresults[5],
        "OverlapReward": TSBresults[6],
        "RAUC": TSBresults[10],
        "Accuracy": float(AELSTMresults['Accuarcy']),
        "confusion_matrix": AELSTMresults['confusion matrix'].astype(int).tolist(),
        "time": elapsed_time
    }
    if additional_stats is not None:
        results = {**results, **additional_stats}

    save_dict_to_json_file(results, output_json)
    return results



# ===========================================================================================================
# Load results files and clean dataframe
# ===========================================================================================================

def prepare_results_df(normality, num_of_models = 4, filter_results = False, load_ablation_study_results=False):
    df = _read_files(load_ablation_study_results)
    if normality is not None:
        df = df[df['normality'] == normality]
    df = _sort_df(df)
    df = _clean_up_df(df)
    if load_ablation_study_results:
        df = df[df['model'] == 'Online-EncDec-AD']
    elif filter_results:
        df = _get_results_for_datasets_with_all_models(df, num_of_models)
    df = _add_average_values_per_model(df)

    return df



def _read_files(load_ablation_study_results=False):
    results_path = "ablation_study" if load_ablation_study_results else "results"
    results_path = f"OUTPUTS/{results_path}"
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

def _sort_df(df):
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

def _clean_up_df(df: pd.DataFrame):
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


def _add_average_values_per_model(df):
    avg = df.groupby(by=['model']).mean().reset_index()
    avg['datasets'] = ["Average"] * avg.shape[0]
    avg = avg[['datasets'] + [col for col in avg.columns if col != 'datasets']]
    avg = _sort_df(avg)
    df = pd.concat([df, avg], ignore_index=True)
    return df


def print_sep_df_per_dataset(df: pd.DataFrame):
    datasets = df['datasets'].unique()
    for d in datasets:
        ddf = df[df['datasets'] == d]
        print(ddf)
        print("\n", "-"*200)
    print()


def _get_results_for_datasets_with_all_models (df, num_of_models = 4):
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





