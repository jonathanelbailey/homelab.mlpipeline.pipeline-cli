import os
from types import SimpleNamespace
from src.calibration import generate_vegas_wp_calibration_data
from src.data_utils import (
    clean_nfl_data,
    generate_files_diff,
    generate_required_seasons,
    generate_tags,
    get_dataset,
    get_seasons_to_import,
    is_current,
    load_pbp_data,
    process_dataframe,
    read_csvs_in_parallel,
    update_dataset,
)

CAL_DATA_VEGAS_WP = "Vegas WP Calibration Data"


def pbp_dataset(args):
    args = SimpleNamespace(**args)

    if args.update:
        remote_dataset = get_dataset(args.dataset, args.project, generate_tags(args))

        seasons_to_be_imported = _get_seasons_if_needed(remote_dataset, args)

        if not seasons_to_be_imported:
            print(f"Dataset is current. No new data to import.")
            return

        print(f"Importing Play-by-Play Data for the following years:\n{seasons_to_be_imported}")
        pbp_data = _process_pbp_data(seasons_to_be_imported)

        pbp_dataset_file_prefix = "pbp_data"
        _save_pbp_data(pbp_data, pbp_dataset_file_prefix)

        update_dataset(remote_dataset, pbp_dataset_file_prefix)

        if args.calibrate and args.vegas:
            _calibrate_data(args, remote_dataset)


def _create_directory(directory):

    os.makedirs(directory, exist_ok=True)


def _get_seasons_if_needed(dataset, args):

    if is_current(dataset, generate_tags(args)):

        return []

    return get_seasons_to_import(args)


def _process_pbp_data(seasons):

    pbp_df = load_pbp_data(seasons)

    pbp_data = clean_nfl_data(pbp_df)

    return pbp_data


def _save_pbp_data(pbp_data, file_prefix):

    _create_directory(file_prefix)

    process_dataframe(pbp_data, file_prefix)


def _generate_calibration_tags(args):

    return ["CALIBRATION DATA", f"{args.dataset}"] + generate_tags(args)


def _process_calibration_data(dataset, seasons):

    file_list = dataset.get_files()

    pbp_data = read_csvs_in_parallel(generate_required_seasons(generate_files_diff(file_list, seasons), file_list))

    return generate_vegas_wp_calibration_data(pbp_data)


def _save_calibration_data(calibration_data, cal_prefix):

    _create_directory(cal_prefix)

    process_dataframe(calibration_data, cal_prefix)


def _calibrate_data(args, dataset):

    cal_data = get_dataset(CAL_DATA_VEGAS_WP, args.project, _generate_calibration_tags(args))

    seasons_to_calibrate = _get_seasons_if_needed(cal_data, args)

    if not seasons_to_calibrate:
        print("Calibration data is current. No new data to import")
        return

    print(f"Importing Calibration Data for the following years:\n{seasons_to_calibrate}")
    calibration_data = _process_calibration_data(dataset, seasons_to_calibrate)

    _save_calibration_data(calibration_data, args.cal_prefix)
