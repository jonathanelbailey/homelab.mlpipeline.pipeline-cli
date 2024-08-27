import unittest
from unittest.mock import patch

from src.pipeline_cli import main


class TestPipelineCli(unittest.TestCase):

    @patch("src.pipeline_cli.pbp_dataset")
    def test_entry_point_dataset(self, mock_pbp_dataset):
        start_year = "2018"
        end_year = "2020"
        dataset_name = "NFL Play-by-Play Data"
        project_name = "NFL WP Model"

        patch_args = [
            "pipeline_cli.py",
            "datasets",
            "-s",
            start_year,
            "-e",
            end_year,
            "-d",
            dataset_name,
            "-u",
            "-p",
            project_name,
            "-c",
            "--vegas",
        ]

        with patch("sys.argv", patch_args):
            main()

        mock_pbp_dataset.assert_called_once_with(
            start_year=int(start_year),
            end_year=int(end_year),
            dataset=dataset_name,
            update=True,
            project=project_name,
            calibrate=True,
            vegas=True,
        )


if __name__ == "__main__":
    unittest.main()
