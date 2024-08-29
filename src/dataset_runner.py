import os
from types import SimpleNamespace
from src.calibration import generate_vegas_wp_calibration_data
from src.data_utils import (
    clean_nfl_data,
    generate_tags,
    get_dataset,
    get_seasons_to_import,
    is_current,
    load_pbp_data,
    process_dataframe,
    read_csvs_in_parallel,
    update_dataset,
    generate_seasons,
)
from src.constants import (
    VEGAS_WP_CALIBRATION_DATASET_TAGS,
    VEGAS_WP_CALIBRATION_DATASET,
    VEGAS_WP_CALIBRATION_DATASET_PROJECT
)


def pbp_dataset(args):
    args = SimpleNamespace(**args)

    if args.update:
        remote_dataset = get_dataset(args.dataset, args.project, writable_copy=True, tags=generate_tags(args))
        if remote_dataset.is_final():
            print(f"Dataset is current. No new data to import.")
            return

        print(f"Importing Play-by-Play Data for the following years:\n{generate_seasons(args)}")
        pbp_data = _process_pbp_data(generate_seasons(args))

        pbp_dataset_file_prefix = "pbp_data"
        _save_pbp_data(pbp_data, pbp_dataset_file_prefix)

        update_dataset(remote_dataset, pbp_dataset_file_prefix)

    if args.calibrate and args.vegas:
        _calibrate_data(args)


def _create_directory(directory):

    os.makedirs(directory, exist_ok=True)


def _get_seasons_if_needed(dataset, args):

    if is_current(dataset, generate_tags(args)):

        return []

    return get_seasons_to_import(dataset, args)


def _process_pbp_data(seasons):

    pbp_df = load_pbp_data(seasons)

    pbp_data = clean_nfl_data(pbp_df)

    return pbp_data


def _save_pbp_data(pbp_data, file_prefix):

    _create_directory(file_prefix)

    process_dataframe(pbp_data, file_prefix)


def _generate_calibration_tags(args):

    return VEGAS_WP_CALIBRATION_DATASET_TAGS + generate_tags(args)


def _process_calibration_data(pbp_dataset):

    dataset_files = pbp_dataset.list_files()
    dataset_files_path = pbp_dataset.get_local_copy()
    dataset_files_list = [os.path.join(dataset_files_path, file) for file in dataset_files]

    pbp_data = read_csvs_in_parallel(dataset_files_list)

    return generate_vegas_wp_calibration_data(pbp_data)


def _save_calibration_data(calibration_data, cal_prefix):

    _create_directory(cal_prefix)

    process_dataframe(calibration_data, cal_prefix)


def _calibrate_data(args):
    pbp_data = get_dataset(args.dataset, args.project, writable_copy=False, tags=generate_tags(args))

    cal_data = get_dataset(
        VEGAS_WP_CALIBRATION_DATASET,
        VEGAS_WP_CALIBRATION_DATASET_PROJECT,
        tags=_generate_calibration_tags(args))

    if cal_data.is_final():
        print("Calibration data is current. No new data to import")
        return

    cal_data_prefix = "cal_data"
    calibration_data = _process_calibration_data(pbp_data)
    calibration_data.to_csv("cal_data.csv")
    # _save_calibration_data(calibration_data, cal_data_prefix)
    #
    # update_dataset(cal_data, cal_data_prefix)
