from typing import Any, Callable, Optional
from  concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import xgboost as xgb


# Notice Preprocess class Must be named "Preprocess"
class Preprocess(object):
    def __init__(self):
        # set internal state, this will be called only once. (i.e. not per request)
        self._model = xgb.XGBClassifier()
        self.executor = ThreadPoolExecutor(max_workers=32)

    def load(self, local_file_name: str) -> Optional[Any]:  # noqa
        """
        Optional: provide loading method for the model
        useful if we need to load a model in a specific way for the prediction engine to work
        :param local_file_name: file name / path to read load the model from
        :return: Object that will be called with .predict() method for inference
        """

        # Example now lets load the actual model
        self._model.load_model(local_file_name)

    def preprocess(self, body: dict, state: dict, collect_custom_statistics_fn=None) -> Any:
        WP_SELECTED_COLUMNS = [
            "season",
            "half_seconds_remaining",
            "game_seconds_remaining",
            "score_differential",
            "down",
            "ydstogo",
            "yardline_100",
            "posteam_timeouts_remaining",
            "defteam_timeouts_remaining",
            "home",
            "receive_2h_ko",
            "spread_time",
            "diff_time_ratio",
        ]

        df = pd.DataFrame.from_dict(body)

        df["home"] = df.apply(lambda row: 1 if row["posteam"] == row["home_team"] else 0, axis=1)

        ko_df = df.groupby("game_id", group_keys=False).apply(
            lambda x: x.assign(receive_2h_ko=np.where((x["qtr"] <= 2) & (x["posteam"] == x["defteam"].dropna().iloc[0]), 1, 0))
        )

        spread_share_df = ko_df.assign(
            posteam_spread=np.where(ko_df["home"] == 1, ko_df["spread_line"], -1 * ko_df["spread_line"]),
            elapsed_share=(3600 - ko_df["game_seconds_remaining"]) / 3600,
        )

        final_df = spread_share_df.assign(
            spread_time=spread_share_df["posteam_spread"] * np.exp(-4 * spread_share_df["elapsed_share"]),
            diff_time_ratio=spread_share_df["score_differential"] / (np.exp(-4 * spread_share_df["elapsed_share"])),
        )

        final_df = final_df.filter(items=WP_SELECTED_COLUMNS)

        return final_df.to_numpy()

    def process(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Optional[Callable[[dict], None]],
    ) -> Any:  # noqa
        # """
        # Optional: do something with the actual data, return any type of object.
        # The returned object will be passed as is to the postprocess function engine
        #
        # :param data: object as recieved from the preprocessing function
        # :param state: Use state dict to store data passed to the post-processing function call.
        #     This is a per-request state dict (meaning a dict instance per request)
        #     Usage example:
        #     >>> def preprocess(..., state):
        #             state['preprocess_aux_data'] = [1,2,3]
        #     >>> def postprocess(..., state):
        #             print(state['preprocess_aux_data'])
        # :param collect_custom_statistics_fn: Optional, if provided allows to send a custom set of key/values
        #     to the statictics collector servicd.
        #     None is passed if statiscs collector is not configured, or if the current request should not be collected
        #
        #     Usage example:
        #     >>> if collect_custom_statistics_fn:
        #     >>>   collect_custom_statistics_fn({"type": "classification"})
        #
        # :return: Object to be passed tp the post-processing function
        # """

        # this is where we do the heavy lifting, i.e. run our model.
        # notice we know data is a numpy array of type float, because this is what we prepared in preprocessing function
        # data is also a numpy array, as returned from our fit function
        return self._model.predict_proba(data)

    def postprocess(self, data: Any, state: dict, collect_custom_statistics_fn=None) -> dict:
        # post process the data returned from the model inference engine
        # data is the return value from model.predict we will put is inside a return value as Y
        return dict(y=data.tolist() if isinstance(data, np.ndarray) else data)
