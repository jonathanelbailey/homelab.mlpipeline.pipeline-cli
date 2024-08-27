import os
import random
import unittest
from unittest.mock import MagicMock, call, patch

import pandas as pd

from src.data_utils import (
    clean_nfl_data,
    generate_files_diff,
    generate_required_seasons,
    generate_seasons,
    generate_tags,
    get_dataset,
    get_seasons_to_import,
    load_pbp_data,
    process_dataframe,
    read_csvs_in_parallel,
    read_from_csv,
    save_to_csv,
    update_dataset,
)


class TestDataUtils(unittest.TestCase):

    def setUp(self):
        self.seasons = [2018, 2019, 2020]
        self.test_data = pd.DataFrame({"column1": [1, 2, 3], "column2": ["a", "b", "c"], "season": self.seasons})
        self.grouped_test_data = self.test_data.groupby("season")

    @patch("src.data_utils.nfl.import_pbp_data")
    def test_load_pbp_data(self, mock_import_pbp_data):
        mock_import_pbp_data.return_value = self.test_data
        result = load_pbp_data(self.seasons)
        mock_import_pbp_data.assert_called_with(self.seasons, thread_requests=True)
        pd.testing.assert_frame_equal(result, self.test_data)

    @patch("src.data_utils.nfl.clean_nfl_data")
    def test_clean_nfl_data(self, mock_clean_nfl_data):
        mock_clean_nfl_data.return_value = self.test_data
        result = clean_nfl_data(self.test_data)
        mock_clean_nfl_data.assert_called_with(self.test_data)
        pd.testing.assert_frame_equal(result, self.test_data)

    @patch("src.data_utils.print")
    @patch("src.data_utils.pd.DataFrame.to_csv")
    def test_save_to_csv(self, mock_to_csv, mock_print):
        file_prefix = "dummy_prefix"
        season = self.seasons[0]
        df_tuple = (self.seasons[0], file_prefix, self.test_data)
        expected_filename = os.path.join(file_prefix, f"{file_prefix}_{season}.csv")
        save_to_csv(df_tuple)
        mock_to_csv.assert_called_with(expected_filename)
        mock_print.assert_called_with(f"Saved {expected_filename}")

    @patch("src.data_utils.Pool")
    @patch("src.data_utils.pd.DataFrame.groupby")
    def test_process_dataframe(self, mock_groupby, MockPool):
        mock_groupby.return_value = self.grouped_test_data
        file_prefix = "dummy_prefix"
        mock_pool_instance = MockPool.return_value.__enter__.return_value
        mock_pool_instance.map.return_value = None

        process_dataframe(self.test_data, file_prefix)

        mock_groupby.assert_called_with("season")
        MockPool.assert_called_with()
        mock_pool_instance.map.assert_called()

    @patch("src.data_utils.print")
    @patch("src.data_utils.pd.read_csv")
    def test_read_from_csv(self, mock_read_csv, mock_print):
        file_path = "dummy_file_path"

        read_from_csv(file_path)

        mock_read_csv.assert_called_with(file_path, low_memory=False, index_col=0)
        mock_print.assert_called_with(f"Reading {file_path}")

    @patch("src.data_utils.pd.concat")
    @patch("src.data_utils.Pool")
    def test_read_csvs_in_parallel(self, MockPool, mock_concat):
        file_list = ["file1", "file2", "file3"]

        mock_pool_instance = MockPool.return_value.__enter__.return_value
        mock_pool_instance.map.return_value = [self.test_data, self.test_data, self.test_data]
        mock_concat.return_value = self.test_data

        actual = read_csvs_in_parallel(file_list)

        MockPool.assert_called_with()
        mock_pool_instance.map.assert_called_with(read_from_csv, file_list)
        mock_concat.assert_called_with([self.test_data, self.test_data, self.test_data], ignore_index=True)
        assert isinstance(actual, pd.DataFrame)

    def run_test_get_dataset(self, MockDataset, mock_print, side_effect, expected_calls, tags, expected):
        MockDataset.get.side_effect = side_effect
        dataset_name = expected_calls[0]["dataset_name"]
        dataset_project = expected_calls[0]["dataset_project"]
        print_msg = f"Dataset {dataset_name} not found in project {dataset_project}, creating new dataset."

        actual = get_dataset(dataset_name, dataset_project, tags)
        if side_effect[0] == ValueError:
            mock_print.assert_called_once_with(print_msg)
        MockDataset.get.assert_has_calls([call(**expected_call) for expected_call in expected_calls])

        assert isinstance(actual, expected)

    @patch("src.data_utils.print")
    @patch("src.data_utils.Dataset")
    def test_get_dataset(self, MockDataset, mock_print):
        test_cases = [
            # Test Case 1: Dataset Exists
            (
                [MagicMock()],  # side_effect when no ValueError is raised
                [{"dataset_name": "dummy_dataset", "dataset_project": "dummy_project", "writable_copy": True}],
                None,  # No tags needed
                MagicMock,  # Expected result
            ),
            # Test Case 2: Dataset Does Not Exist, create new dataset
            (
                [ValueError(), MagicMock()],  # side_effect when ValueError is raised
                [
                    {"dataset_name": "dummy_dataset", "dataset_project": "dummy_project", "writable_copy": True},
                    {
                        "dataset_name": "dummy_dataset",
                        "dataset_project": "dummy_project",
                        "auto_create": True,
                        "writable_copy": True,
                        "dataset_tags": ["2018-2020"],
                    },
                ],
                ["2018-2020"],  # tags needed
                MagicMock,  # Expected result
            ),
        ]

        for side_effect, expected_calls, tags, expected in test_cases:
            with self.subTest(side_effect=side_effect, expected_calls=expected_calls, tags=tags, expected=expected):
                self.run_test_get_dataset(MockDataset, mock_print, side_effect, expected_calls, tags, expected)

    @patch("src.data_utils.Dataset")
    def test_update_dataset(self, MockDataset):
        file_prefix = "dummy_prefix"
        dataset = MockDataset.get.return_value
        dataset.add_files.return_value = None
        dataset.upload.return_value = None
        dataset.finalize.return_value = None

        update_dataset(dataset, file_prefix)

        dataset.add_files.assert_called_with(path=file_prefix, wildcard=f"{file_prefix}_*.csv")
        dataset.upload.assert_called()
        dataset.finalize.assert_called()

    def test_generate_tags(self):
        args = MagicMock()
        args.start_year = 2018
        args.end_year = (2020, "COMPLETE")
        expected = ["2018-2020", "COMPLETE"]
        actual = generate_tags(args)
        assert actual == expected

    def test_generate_seasons(self):
        args = MagicMock()
        args.start_year = 2018
        args.end_year = (2020, "COMPLETE")
        expected = [2018, 2019, 2020]
        actual = generate_seasons(args)
        assert actual == expected

    def test_generate_files_diff(self):
        dataset = MagicMock()
        dataset.get_files.return_value = ["file1", "file2", "file3"]
        seasons = [2018, 2019, 2020]
        expected = 0
        actual = generate_files_diff(dataset, seasons)
        assert actual == expected

    def test_generate_required_seasons(self):
        diff = random.randint(0, 3)
        seasons = [2018, 2019, 2020]
        expected = seasons[-diff:]
        actual = generate_required_seasons(diff, expected)
        assert actual == expected

    @patch("src.data_utils.generate_required_seasons")
    @patch("src.data_utils.generate_files_diff")
    @patch("src.data_utils.generate_seasons")
    def test_get_seasons_to_import(self, mock_generate_seasons, mock_generate_files_diff, mock_generate_required_seasons):
        args = MagicMock()
        args.start_year = 2018
        args.end_year = (2020, "COMPLETE")
        mock_generate_seasons.return_value = [2018, 2019, 2020]
        mock_generate_files_diff.return_value = 0
        expected = []
        mock_generate_required_seasons.return_value = expected

        actual = get_seasons_to_import(args)

        mock_generate_seasons.assert_called_with(args)
        mock_generate_files_diff.assert_called_with(args.dataset, mock_generate_seasons.return_value)
        mock_generate_required_seasons.assert_called_with(mock_generate_files_diff.return_value, mock_generate_seasons.return_value)

        assert actual == expected


if __name__ == "__main__":
    unittest.main()
