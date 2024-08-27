import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pandas as pd

from src.dataset_runner import pbp_dataset


class DatasetRunner(unittest.TestCase):

    def setUp(self):
        self.seasons = [2018, 2019, 2020]
        self.test_data = pd.DataFrame({"column1": [1, 2, 3], "column2": ["a", "b", "c"], "season": self.seasons})
        self.grouped_test_data = self.test_data.groupby("season")
        self.args = {
            "start_year": 2018,
            "end_year": 2020,
            "dataset": "NFL Play-by-Play Data",
            "project": "NFL WP Model",
            "update": True,
            "calibrate": True,
            "vegas": True,
        }

    @patch("src.dataset_runner._calibrate_data")
    @patch("src.dataset_runner.update_dataset")
    @patch("src.dataset_runner.process_dataframe")
    @patch("src.dataset_runner.clean_nfl_data")
    @patch("src.dataset_runner.load_pbp_data")
    @patch("src.dataset_runner.print")
    @patch("src.dataset_runner.os.makedirs")
    @patch("src.dataset_runner.get_seasons_to_import")
    @patch("src.dataset_runner.generate_tags")
    @patch("src.dataset_runner.get_dataset")
    def test_pbp_dataset(
        self,
        mock_get_dataset,
        mock_generate_tags,
        mock_get_seasons_to_import,
        mock_makedirs,
        mock_print,
        mock_load_pbp_data,
        mock_clean_nfl_data,
        mock_process_dataframe,
        mock_update_dataset,
        mock_calibrate_data,
    ):
        mock_get_dataset.return_value = MagicMock()
        mock_generate_tags.return_value = ["2018-2020", "COMPLETE"]
        mock_get_seasons_to_import.return_value = self.seasons
        mock_load_pbp_data.return_value = self.test_data
        mock_calibrate_data.return_value = self.test_data
        pbp_dataset(self.args)

        mock_makedirs.assert_called_with("pbp_data", exist_ok=True)
        mock_print.assert_has_calls(
            [
                call("Importing Play-by-Play Data for the following years:\n[2018, 2019, 2020]"),
            ]
        )
        mock_load_pbp_data.assert_called_with(self.seasons)
        mock_clean_nfl_data.assert_called_with(self.test_data)
        mock_process_dataframe.assert_called()
        mock_update_dataset.assert_called_with(mock_get_dataset.return_value, "pbp_data")


if __name__ == "__main__":
    unittest.main()
