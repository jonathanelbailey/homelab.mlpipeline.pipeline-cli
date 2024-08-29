from clearml import PipelineController, TaskTypes
from src.constants import (
    VEGAS_WP_MODEL_PIPELINE_PROJECT,
    VEGAS_WP_MODEL_PIPELINE,
    VEGAS_WP_CALIBRATION_DATASET,
)


def get_data(dataset: str):
    from multiprocessing import Pool
    import os
    import pandas as pd
    from src.data_utils import get_dataset, read_from_csv
    from src.constants import VEGAS_WP_CALIBRATION_DATASET_PROJECT

    print("Getting calibration dataset...")
    cal_dataset = get_dataset(dataset, VEGAS_WP_CALIBRATION_DATASET_PROJECT, writable_copy=False)
    cal_dataset_files = cal_dataset.list_files()
    cal_data_csv_path = cal_dataset.get_local_copy()
    cal_data_csv_paths = [os.path.join(cal_data_csv_path, file) for file in cal_dataset_files]
    with Pool() as pool:
        dataframes = pool.map(read_from_csv, cal_data_csv_paths)

    cal_data = pd.concat(dataframes, ignore_index=True)
    print(f"Calibration Data:\n{cal_data.head()}")

    return cal_data


def preprocessing(**kwargs):
    import pandas as pd
    from sklearn.model_selection import GroupKFold

    cal_data = kwargs["cal_data"]

    print("Preprocessing Data...")
    X = cal_data.loc[:, ~cal_data.columns.isin(["season", "game_id", "label", "home_team", "away_team"])]
    y = cal_data["label"]
    groups = cal_data["game_id"]

    print("Creating Folds...")
    group_fold = GroupKFold(n_splits=5)
    for train_index, test_index in group_fold.split(X, y, groups):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    return X_train, y_train, X_test, y_test


def generate_classifier(**kwargs):
    import json

    import xgboost as xgb

    kwargs["monotone_constraints"] = json.loads(kwargs["monotone_constraints"])
    kwargs["eval_metric"] = kwargs["eval_metric"].split(",")
    kwargs["n_estimators"] = int(kwargs["n_estimators"])
    kwargs["early_stopping_rounds"] = int(kwargs["early_stopping_rounds"])
    kwargs["max_depth"] = int(kwargs["max_depth"])
    kwargs["min_child_weight"] = int(kwargs["min_child_weight"])
    kwargs["learning_rate"] = float(kwargs["learning_rate"])
    kwargs["gamma"] = float(kwargs["gamma"])
    kwargs["subsample"] = float(kwargs["subsample"])
    kwargs["colsample_bytree"] = float(kwargs["colsample_bytree"])
    print("Creating Classifier...")
    model = xgb.XGBClassifier(**kwargs)

    return model


def train(**kwargs):
    import matplotlib.pyplot as plt
    import xgboost as xgb

    print("Training Model...")

    model = kwargs["model"]
    X_train = kwargs["X_train"]
    y_train = kwargs["y_train"]
    X_test = kwargs["X_test"]
    y_test = kwargs["y_test"]

    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=50)
    model.score(X_test, y_test)
    xgb.plot_importance(model)
    plt.show()

    model.save_model("wp_vegas_model.json")

    return model


if __name__ == "__main__":

    # create the pipeline controller
    pipe = PipelineController(
        name=VEGAS_WP_MODEL_PIPELINE,
        project=VEGAS_WP_MODEL_PIPELINE_PROJECT,
        packages=[
            "xgboost",
            "pandas",
            "scikit-learn",
            "matplotlib",
            "numpy<2.0.0",
            "nfl-data-py",
            "clearml",
            "pipeline-cli @ git+https://github.com/jonathanelbailey/homelab.mlpipeline.pipeline-cli.git"
        ],
        repo="https://github.com/jonathanelbailey/homelab.mlpipeline.pipeline-cli.git",
        repo_branch="feature/initial_project",
    )

    # set the default execution queue to be used (per step we can override the execution)
    pipe.set_default_execution_queue("default")

    # add pipeline components
    pipe.add_parameter(name="dataset", description="Dataset", default=VEGAS_WP_CALIBRATION_DATASET, param_type="str")

    pipe.add_parameter(name="n_estimators", description="Number of estimators", default=15000, param_type="int")

    pipe.add_parameter(name="booster", description="Booster type", default="gbtree", param_type="str")

    pipe.add_parameter(name="device", description="Device type", default="cuda", param_type="str")

    pipe.add_parameter(name="sampling_method", description="Sampling method", default="gradient_based", param_type="str")

    pipe.add_parameter(name="objective", description="Objective function", default="binary:logistic", param_type="str")

    pipe.add_parameter(
        name="eval_metric",
        description="Evaluation metrics",
        default="auc,error,logloss",
        param_type="list(str)",
    )

    pipe.add_parameter(name="early_stopping_rounds", description="Early stopping rounds", default=200, param_type="int")

    pipe.add_parameter(name="tree_method", description="Tree method", default="approx", param_type="str")

    pipe.add_parameter(name="grow_policy", description="Grow policy", default="lossguide", param_type="str")

    pipe.add_parameter(name="learning_rate", description="Learning rate", default=0.05, param_type="float")

    pipe.add_parameter(name="gamma", description="Gamma", default=0.79012017, param_type="float")

    pipe.add_parameter(name="subsample", description="Subsample ratio", default=0.9224245, param_type="float")

    pipe.add_parameter(name="colsample_bytree", description="Colsample bytree ratio", default=0.4166666666666667, param_type="float")

    pipe.add_parameter(name="max_depth", description="Max depth", default=5, param_type="int")

    pipe.add_parameter(name="min_child_weight", description="Minimum child weight", default=7, param_type="int")

    pipe.add_parameter(
        name="monotone_constraints",
        description="Monotone constraints",
        default={
            "receive_2h_ko": 0,
            "spread_time": 1,
            "home": 0,
            "half_seconds_remaining": 0,
            "game_seconds_remaining": 0,
            "diff_time_ratio": 1,
            "score_differential": 1,
            "down": -1,
            "ydstogo": -1,
            "yardline_100": -1,
            "posteam_timeouts_remaining": 1,
            "defteam_timeouts_remaining": -1,
        },
        param_type="dict(str:int)",
    )

    pipe.add_function_step(
        name="get_data",
        function=get_data,
        function_kwargs=dict(dataset="${pipeline.dataset}"),
        function_return=["cal_data"],
        cache_executed_step=True,
        task_type=TaskTypes.data_processing,
    )
    pipe.add_function_step(
        name="preprocessing",
        function=preprocessing,
        parents=["get_data"],
        function_kwargs=dict(cal_data="${get_data.cal_data}"),
        function_return=["X_train", "y_train", "X_test", "y_test"],
        cache_executed_step=True,
        task_type=TaskTypes.data_processing,
    )
    pipe.add_function_step(
        name="generate_classifier",
        function=generate_classifier,
        function_kwargs=dict(
            n_estimators="${pipeline.n_estimators}",
            booster="${pipeline.booster}",
            device="${pipeline.device}",
            sampling_method="${pipeline.sampling_method}",
            objective="${pipeline.objective}",
            eval_metric="${pipeline.eval_metric}",
            early_stopping_rounds="${pipeline.early_stopping_rounds}",
            tree_method="${pipeline.tree_method}",
            grow_policy="${pipeline.grow_policy}",
            learning_rate="${pipeline.learning_rate}",
            gamma="${pipeline.gamma}",
            subsample="${pipeline.subsample}",
            colsample_bytree="${pipeline.colsample_bytree}",
            max_depth="${pipeline.max_depth}",
            min_child_weight="${pipeline.min_child_weight}",
            monotone_constraints="${pipeline.monotone_constraints}",
        ),
        function_return=["model"],
        monitor_models=["model"],
        cache_executed_step=True,
        task_type=TaskTypes.training,
    )
    pipe.add_function_step(
        name="train",
        function=train,
        parents=["generate_classifier", "preprocessing"],
        function_kwargs=dict(
            model="${generate_classifier.model}",
            X_train="${preprocessing.X_train}",
            y_train="${preprocessing.y_train}",
            X_test="${preprocessing.X_test}",
            y_test="${preprocessing.y_test}",
        ),
        function_return=["model"],
        monitor_models=["model"],
        cache_executed_step=False,
        task_type=TaskTypes.training,
    )

    # For debugging purposes run on the pipeline on current machine
    # Use run_pipeline_steps_locally=True to further execute the pipeline component Tasks as subprocesses.
    # pipe.start_locally(run_pipeline_steps_locally=True)
    pipe.start(queue="default")
    # Start the pipeline on the services queue (remote machine, default on the clearml-server)
    # pipe.start()

    print("pipeline completed")
