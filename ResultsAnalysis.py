import glob
import json
import os
from pathlib import Path

import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 5000)


def read_files():
    results_path = "C:/Users/karal/progr/Python/Mining_Massive_Datasets/StreamingTimeSeriesAnomalyDetection/OUTPUTS/results"
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
    model_order = ['SAND', 'AE-LSTM', 'EncDec-AD', 'EncDec-AD-Batch', 'OnlineEncDecAD', ]
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
    df['model'] = df['model'].replace('EncDec-AD', 'Offline_EncDec-AD')
    return df

def print_sep_df_per_dataset(df: pd.DataFrame):
    datasets = df['datasets'].unique()
    for d in datasets:
        ddf = df[df['datasets'] == d]
        print(ddf)
        print("\n", "-"*100)


def get_results_for_datasets_with_all_models (df, num_of_models = 4):
    datasets_with_all_results = df[['datasets', 'model']].groupby(by=['datasets']).count().reset_index()
    datasets_with_all_results = datasets_with_all_results[datasets_with_all_results['model'] >= num_of_models].reset_index(drop=True)
    datasets_with_all_results = set(datasets_with_all_results['datasets'].values)
    print("selected datasets # ", len(datasets_with_all_results))
    selected_results_df = df[df['datasets'].isin(datasets_with_all_results)]
    return selected_results_df


normality = 2
num_of_models = 4
filter_results = True

df = read_files()
df = sort_df(df)
df = clean_up_df(df)
df = df[df['normality'] == normality]

if filter_results:
    df = get_results_for_datasets_with_all_models(df, num_of_models)

print_sep_df_per_dataset(df)

