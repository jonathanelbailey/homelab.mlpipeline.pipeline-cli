import datetime
import os
from multiprocessing import Pool

import nfl_data_py as nfl
import pandas as pd
from clearml import Dataset


def load_pbp_data(seasons):
    pbp_data = nfl.import_pbp_data(seasons, thread_requests=True)
    return pbp_data


def clean_nfl_data(df):
    cleaned_data = nfl.clean_nfl_data(df)
    return cleaned_data


def save_to_csv(df_tuple):
    season, file_prefix, df = df_tuple
    filename = f"{file_prefix}_{season}.csv"
    file_folder_path = os.path.join(file_prefix, filename)
    df.to_csv(file_folder_path)
    print(f"Saved {file_folder_path}")


def process_dataframe(df, file_prefix):
    grouped = df.groupby("season")
    df_tuples = [(season, file_prefix, group) for season, group in grouped]
    with Pool() as pool:
        pool.map(save_to_csv, df_tuples)


def read_from_csv(file_path):
    print(f"Reading {file_path}")
    return pd.read_csv(file_path, low_memory=False, index_col=0)


def read_csvs_in_parallel(file_list):
    with Pool() as pool:
        dataframes = pool.map(read_from_csv, file_list)
    return pd.concat(dataframes, ignore_index=True)


def is_current(dataset, tags):
    files = dataset.list_files()
    dataset_start = files[0].split(".", 1)[0].split("_")[-1]
    dataset_end = files[-1].split(".", 1)[0].split("_")[-1]
    dataset_years = f"{dataset_start}-{dataset_end}"
    dataset_state = tags[1]
    if dataset_state == "COMPLETE" and dataset_years == tags[0]:
        return True
    return False


def get_dataset(dataset_name, dataset_project, tags=None):
    try:
        dataset = Dataset.get(dataset_name=dataset_name, dataset_project=dataset_project, writable_copy=True)

    except ValueError:
        print(f"Dataset {dataset_name} not found in project {dataset_project}, creating new dataset.")
        dataset = Dataset.get(dataset_name=dataset_name, dataset_project=dataset_project, auto_create=True, writable_copy=True, tags=tags)

    return dataset


def update_dataset(dataset, file_prefix):

    wildcard = f"{file_prefix}_*.csv"
    dataset.add_files(path=file_prefix, wildcard=wildcard)
    dataset.upload()
    dataset.finalize()


def generate_tags(args):
    return [
        f"{args.start_year}-{args.end_year[0]}",
        f"{args.end_year[1]}",
    ]


def generate_seasons(args):
    return list(range(args.start_year, args.end_year[0] + 1, 1))


def generate_files_diff(dataset, seasons):
    return len(dataset.get_files()) - len(seasons)


def generate_required_seasons(diff, seasons):
    return seasons[-diff:]


def get_seasons_to_import(args):
    seasons = generate_seasons(args)
    diff = generate_files_diff(args.dataset, seasons)

    return generate_required_seasons(diff, seasons)


def get_current_season_year():
    now = datetime.datetime.now()
    if 2 > now.month < 9:
        return (now.year - 1, "COMPLETE")
    return (now.year, "IN PROGRESS")
