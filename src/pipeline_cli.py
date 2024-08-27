import argparse
from textwrap import dedent
from src.data_utils import get_current_season_year
from src.dataset_runner import pbp_dataset

META_VAR = "YYYY"


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Import Data from nfl-data-py to a ClearML Dataset",
        epilog=dedent(
            """\
            FURTHER READING
            ---------------

            Subject                             Link
            -------                             ----
            Field Descriptions:                 https://www.nflfastr.com/articles/field_descriptions.html
            WP Calibration Data:                https://github.com/guga31bb/metrics/tree/master/wp_tuning
        """
        ),
    )

    subparsers = parser.add_subparsers(help="subcommand help", required=True)

    _add_dataset_subparser(subparsers)

    kwargs = vars(parser.parse_args())

    subcommand = kwargs.pop("backend")
    del kwargs["/subparser1"]

    subcommand(kwargs)


def _add_dataset_subparser(subparsers):

    subparser = subparsers.add_parser("datasets", description="Import data to a ClearML Dataset")

    subparser.add_argument(
        "-s", "--start-year", type=int, metavar=META_VAR, required=False, default=1999, help="Starting Season.  Default: 1999"
    )

    subparser.add_argument(
        "-e",
        "--end-year",
        type=int,
        metavar="YYYY",
        required=False,
        default=get_current_season_year(),
        help=f"Ending Season.  Default: {get_current_season_year()}",
    )

    subparser.add_argument(
        "-d",
        "--dataset",
        type=str,
        metavar="DATASET_NAME",
        required=False,
        default="NFL Play-by-Play Data",
        help="Dataset Name.  Default: NFL Play-by-Play Data",
    )

    subparser.add_argument("-u", "--update", action="store_true", help="Update the dataset with the latest data")

    subparser.add_argument(
        "-p",
        "--project",
        type=str,
        metavar="PROJECT_NAME",
        required=False,
        default="NFL Models",
        help="Project Name.  Default: NFL Models",
    )

    subparser.add_argument("-c", "--calibrate", action="store_true", help="Generate calibration data")

    subparser.add_argument("--vegas", action="store_true", help="Generate Vegas WP Calibration Data")

    subparser.set_defaults(backend=pbp_dataset)
