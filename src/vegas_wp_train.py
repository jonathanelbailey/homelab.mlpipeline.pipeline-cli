from clearml import Task
from multiprocessing import Pool
import os
import pandas as pd
from sklearn.model_selection import GroupKFold
import json
import xgboost as xgb
import matplotlib.pyplot as plt
from src.data_utils import get_dataset, read_from_csv
from src.constants import (
    VEGAS_WP_MODEL_PIPELINE_PROJECT,
    VEGAS_WP_MODEL_PIPELINE,
    VEGAS_WP_CALIBRATION_DATASET,
    VEGAS_WP_CALIBRATION_DATASET_PROJECT
)

task = Task.init(project_name=VEGAS_WP_MODEL_PIPELINE_PROJECT, task_name=VEGAS_WP_MODEL_PIPELINE, output_uri=True)

seasons = list(range(1999, 2024, 1))

print("Getting calibration dataset...")
cal_data = read_from_csv("cal_data.csv")

print(f"Calibration Data:\n{cal_data.head()}")

print("Preprocessing Data...")
X = cal_data.loc[:, ~cal_data.columns.isin(["season", "game_id", "label", "home_team", "away_team"])]
y = cal_data["label"]
groups = cal_data["game_id"]

print("Creating Folds...")
group_fold = GroupKFold(n_splits=5)
for train_index, test_index in group_fold.split(X, y, groups):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

params = dict(
n_estimators=15000,
booster="gbtree",
device="cuda",
sampling_method="gradient_based",
objective="binary:logistic",
eval_metric=["logloss"],
early_stopping_rounds=200,
tree_method="approx",
grow_policy="lossguide",
learning_rate=0.05,
gamma=0.79012017,
subsample=0.9224245,
colsample_bytree=0.4166666666666667,
max_depth=5,
min_child_weight=7,
monotone_constraints={
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
})

print("Creating Classifier...")
model = xgb.XGBClassifier(**params)


print("Training Model...")

model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=50)
model.score(X_test, y_test)
xgb.plot_importance(model)
plt.show()

model.save_model("wp_vegas_model.json")