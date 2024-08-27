import numpy as np
import pandas as pd

WP_SELECTED_COLUMNS = [
    "label",
    "game_id",
    "home_team",
    "away_team",
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
WP_DROP_COLUMNS = ["season", "game_id", "label", "home_team", "away_team"]

NEXT_SCORE_HALF_SELECTED_COLUMNS = [
    "game_id",
    "Next_Score_Half",
    "Drive_Score_Half",
    "play_type",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "yardline_100",
    "roof",
    "posteam",
    "defteam",
    "home_team",
    "ydstogo",
    "season",
    "qtr",
    "down",
    "week",
    "drive",
    "ep",
    "score_differential",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "desc",
    "receiver_player_name",
    "pass_location",
    "air_yards",
    "yards_after_catch",
    "complete_pass",
    "incomplete_pass",
    "interception",
    "qb_hit",
    "extra_point_result",
    "field_goal_result",
    "sp",
    "Winner",
    "spread_line",
    "total_line",
]

EPA_MODEL_SELECTED_COLUMNS = [
    "label",
    "season",
    "half_seconds_remaining",
    "yardline_100",
    "home",
    "retractable",
    "dome",
    "outdoors",
    "ydstogo",
    "era0",
    "era1",
    "era2",
    "era3",
    "era4",
    "down1",
    "down2",
    "down3",
    "down4",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "Total_W_Scaled",
]


def drop_rows(df):
    print("Dropping Rows")
    new_df = df.dropna(subset=["down", "game_seconds_remaining", "score_differential", "yardline_100", "result", "posteam"])
    new_df = new_df.loc[new_df["qtr"] <= 4]
    new_df = new_df.loc[new_df["result"] != 0]
    return new_df


def add_home_column(df):
    print("Adding Home Column")
    df["home"] = df.apply(lambda row: 1 if row["posteam"] == row["home_team"] else 0, axis=1)
    return df


def add_label_column(df):
    print("Adding Label Column")
    df["label"] = df.apply(
        lambda row: (
            1
            if (row["result"] > 0 and row["posteam"] == row["home_team"]) or (row["result"] < 0 and row["posteam"] == row["away_team"])
            else 0
        ),
        axis=1,
    )
    return df


def add_receive_2h_ko_column(df):
    print("Adding Receive 2H KO Column")
    new_df = df.groupby("game_id", group_keys=False).apply(
        lambda x: x.assign(receive_2h_ko=np.where((x["qtr"] <= 2) & (x["posteam"] == x["defteam"].dropna().iloc[0]), 1, 0))
    )
    return new_df


def add_posteam_spread_elasped_share_columns(df):
    print("Adding Posessing Team Spread and Elapsed Share Columns")
    new_df = df.assign(
        posteam_spread=np.where(df["home"] == 1, df["spread_line"], -1 * df["spread_line"]),
        elapsed_share=(3600 - df["game_seconds_remaining"]) / 3600,
    )
    return new_df


def add_spread_time_diff_time_ration_columns(df):
    print("Adding Spread Time and Diff Time Ratio Columns")
    new_df = df.assign(
        spread_time=df["posteam_spread"] * np.exp(-4 * df["elapsed_share"]),
        diff_time_ratio=df["score_differential"] / (np.exp(-4 * df["elapsed_share"])),
    )
    return new_df


def select_relevant_columns(df):
    print("Selecting Relevant Columns")
    return df.filter(items=WP_SELECTED_COLUMNS)


def drop_irrelevant_columns(df):
    print("Dropping Irrelevant Columns")
    return df.drop(columns=WP_DROP_COLUMNS)


def add_features(df):
    new_df = add_home_column(df)
    new_df = add_label_column(new_df)
    new_df = add_receive_2h_ko_column(new_df)
    new_df = add_posteam_spread_elasped_share_columns(new_df)
    new_df = add_spread_time_diff_time_ration_columns(new_df)

    return new_df


def generate_vegas_wp_calibration_data(df):
    print("Preprocessing Data")
    new_df = drop_rows(df)
    new_df = add_features(new_df)
    new_df = select_relevant_columns(new_df)
    return new_df


def make_model_mutations(pbp):
    pbp["era0"] = np.where(pbp["season"] <= 2001, 1, 0)
    pbp["era1"] = np.where((pbp["season"] > 2001) & (pbp["season"] <= 2005), 1, 0)
    pbp["era2"] = np.where((pbp["season"] > 2005) & (pbp["season"] <= 2013), 1, 0)
    pbp["era3"] = np.where((pbp["season"] > 2013) & (pbp["season"] <= 2017), 1, 0)
    pbp["era4"] = np.where(pbp["season"] > 2017, 1, 0)

    pbp["era"] = np.select([pbp["era0"] == 1, pbp["era1"] == 1, pbp["era2"] == 1, (pbp["era3"] == 1) | (pbp["era4"] == 1)], [0, 1, 2, 3])

    pbp["era"] = pd.Categorical(pbp["era"])

    pbp["down1"] = np.where(pbp["down"] == 1, 1, 0)
    pbp["down2"] = np.where(pbp["down"] == 2, 1, 0)
    pbp["down3"] = np.where(pbp["down"] == 3, 1, 0)
    pbp["down4"] = np.where(pbp["down"] == 4, 1, 0)

    pbp["home"] = np.where(pbp["posteam"] == pbp["home_team"], 1, 0)

    pbp["model_roof"] = np.where((pbp["roof"].isna()) | (pbp["roof"] == "open") | (pbp["roof"] == "closed"), "retractable", pbp["roof"])
    pbp["model_roof"] = pd.Categorical(pbp["model_roof"])

    pbp["retractable"] = np.where(pbp["model_roof"] == "retractable", 1, 0)
    pbp["dome"] = np.where(pbp["model_roof"] == "dome", 1, 0)
    pbp["outdoors"] = np.where(pbp["model_roof"] == "outdoors", 1, 0)

    return pbp


def find_game_next_score_half(pbp_dataset):
    # Which rows are the scoring plays
    score_plays = pbp_dataset.index[(pbp_dataset["sp"] == 1) & (pbp_dataset["play_type"] != "no_play")].tolist()

    def find_next_score(play_i, score_plays_i, pbp_df):
        # game_id = pbp_df.iloc[0]['game_id']
        # df_len = len(pbp_df)
        # print(f"{game_id}: Play {play_i} of {len(pbp_df)} Status: Beginning")
        # Find the next score index for the current play
        next_scores = [i for i in score_plays_i if i >= play_i]
        # print(f"{game_id}: Play {play_i} of {df_len}: Next Scores: {next_scores} Status: Phase 1")
        next_score_i = next_scores[0] if next_scores else None
        # print(f"{game_id}: Play {play_i} of {df_len}: Next Score Index: {next_score_i} Status: Phase 2")
        # next_score_i_qtr = pbp_df.loc[next_score_i, 'qtr'] if next_score_i else None
        # print(f"{game_id}: Play {play_i} of {df_len}: Next Score Qtr: {next_score_i_qtr} Status: Phase 3")

        if (
            next_score_i is None
            or (pbp_df.loc[play_i, "qtr"] in [1, 2] and pbp_df.loc[next_score_i, "qtr"] in [3, 4, 5])
            or (pbp_df.loc[play_i, "qtr"] in [3, 4] and pbp_df.loc[next_score_i, "qtr"] == 5)
        ):
            score_type = "No_Score"
            # print(f"{game_id}: Play {play_i} of {df_len}: Score Type: {score_type}, Status: Phase 4")
            score_drive = pbp_df.loc[play_i, "drive"]
            # print(f"{game_id}: Play {play_i} of {df_len}: Score Drive: {score_drive} Status: Phase 5")
        else:
            score_drive = pbp_df.loc[next_score_i, "drive"]
            # print(f"{game_id}: Play {play_i} of {df_len}: Score Drive: {score_drive} Status: Phase 6")

            if pbp_df.loc[next_score_i, "touchdown"] == 1 and (pbp_df.loc[next_score_i, "td_team"] != pbp_df.loc[next_score_i, "posteam"]):
                if pbp_df.loc[play_i, "posteam"] == pbp_df.loc[next_score_i, "posteam"]:
                    score_type = "Opp_Touchdown"
                else:
                    score_type = "Touchdown"
            elif pbp_df.loc[next_score_i, "field_goal_result"] == "made":
                if pbp_df.loc[play_i, "posteam"] == pbp_df.loc[next_score_i, "posteam"]:
                    score_type = "Field_Goal"
                else:
                    score_type = "Opp_Field_Goal"
            elif pbp_df.loc[next_score_i, "touchdown"] == 1:
                if pbp_df.loc[play_i, "posteam"] == pbp_df.loc[next_score_i, "posteam"]:
                    score_type = "Touchdown"
                else:
                    score_type = "Opp_Touchdown"
            elif pbp_df.loc[next_score_i, "safety"] == 1:
                if pbp_df.loc[play_i, "posteam"] == pbp_df.loc[next_score_i, "posteam"]:
                    score_type = "Opp_Safety"
                else:
                    score_type = "Safety"
            elif pbp_df.loc[next_score_i, "extra_point_result"] == "good":
                if pbp_df.loc[play_i, "posteam"] == pbp_df.loc[next_score_i, "posteam"]:
                    score_type = "Extra_Point"
                else:
                    score_type = "Opp_Extra_Point"
            elif pbp_df.loc[next_score_i, "two_point_conv_result"] == "success":
                if pbp_df.loc[play_i, "posteam"] == pbp_df.loc[next_score_i, "posteam"]:
                    score_type = "Two_Point_Conversion"
                else:
                    score_type = "Opp_Two_Point_Conversion"
            elif pbp_df.loc[next_score_i, "defensive_two_point_conv"] == 1:
                if pbp_df.loc[play_i, "posteam"] == pbp_df.loc[next_score_i, "posteam"]:
                    score_type = "Opp_Defensive_Two_Point"
                else:
                    score_type = "Defensive_Two_Point"
            else:
                score_type = None
        # print(f"{pbp_df.iloc[0]['game_id']}: Play {play_i} of {len(pbp_df)}: Score Type: {score_type} Status: Completed")
        return pd.DataFrame({"Next_Score_Half": [score_type], "Drive_Score_Half": [score_drive]}, index=[play_i])

    # Applying the helper function to each row of the dataset
    next_score_data = [find_next_score(i, score_plays, pbp_dataset) for i in pbp_dataset.index]
    return pd.concat(next_score_data)


def get_regular_season_games(df):
    return df.loc[df["season_type"] == "REG"]


def get_unique_game_ids(df):
    return df["game_id"].unique()


def determine_winner(df):
    df["Winner"] = np.where(
        df["home_score"] > df["away_score"], df["home_team"], np.where(df["home_score"] < df["away_score"], df["away_team"], "TIE")
    )

    return df


def generate_game_dfs_by_game_id(df):
    grouped = df.groupby("game_id")
    games = [game for game_id, game in grouped]

    return games


def update_label_mapping(df):
    label_mapping = {"Touchdown": 0, "Opp_Touchdown": 1, "Field_Goal": 2, "Opp_Field_Goal": 3, "Safety": 4, "Opp_Safety": 5, "No_Score": 6}
    df["label"] = df["Next_Score_Half"].map(label_mapping)
    df["label"] = pd.factorize(df["label"])[0]
    return df


def add_nflscrapr_weights(df):
    df["Drive_Score_Dist"] = df["Drive_Score_Half"] - df["drive"]
    df["Drive_Score_Dist_W"] = (df["Drive_Score_Dist"].max() - df["Drive_Score_Dist"]) / (
        df["Drive_Score_Dist"].max() - df["Drive_Score_Dist"].min()
    )
    df["ScoreDiff_W"] = (df["score_differential"].dropna().abs().max() - df["score_differential"].dropna().abs()) / (
        df["score_differential"].dropna().abs().max() - df["score_differential"].dropna().abs().min()
    )
    df["Total_W"] = df["Drive_Score_Dist_W"] + df["ScoreDiff_W"]
    df["Total_W_Scaled"] = (df["Total_W"] - df["Total_W"].dropna().min()) / (df["Total_W"].dropna().max() - df["Total_W"].dropna().min())
    df["Total_W_Scaled"] = df["Total_W_Scaled"].fillna(0)
    df["label"] = df["label"].astype("int64") + 1

    return df


def filter_epa_model_selected_columns(df):
    cal_data = df.dropna(subset=["defteam_timeouts_remaining", "posteam_timeouts_remaining", "yardline_100"])
    cal_data = cal_data[EPA_MODEL_SELECTED_COLUMNS]
    return cal_data


def process_game_epa(df):
    reg_pbp_data = df.copy()
    reg_pbp_data = reg_pbp_data.loc[reg_pbp_data["season_type"] == "REG"]
    reg_pbp_data["Winner"] = np.where(
        reg_pbp_data["home_score"] > reg_pbp_data["away_score"],
        reg_pbp_data["home_team"],
        np.where(reg_pbp_data["home_score"] < reg_pbp_data["away_score"], reg_pbp_data["away_team"], "TIE"),
    )
    unique_game_ids = reg_pbp_data["game_id"].unique()
    pbp_next_score_half = pd.concat(
        [find_game_next_score_half(reg_pbp_data[reg_pbp_data["game_id"] == game_id]) for game_id in unique_game_ids]
    )
    # Bind to original DataFrame
    reg_pbp_data = pd.concat([reg_pbp_data, pbp_next_score_half], axis=1)

    # Apply filters for estimating the models
    filtered_pbp_data = reg_pbp_data[
        (
            reg_pbp_data["Next_Score_Half"].isin(
                ["Opp_Field_Goal", "Opp_Safety", "Opp_Touchdown", "Field_Goal", "No_Score", "Safety", "Touchdown"]
            )
        )
        & (reg_pbp_data["play_type"].isin(["field_goal", "no_play", "pass", "punt", "run", "qb_spike"]))
        & (reg_pbp_data["two_point_conv_result"].isna())
        & (reg_pbp_data["extra_point_result"].isna())
        & (reg_pbp_data["down"].notna())
        & (reg_pbp_data["game_seconds_remaining"].notna())
    ]

    # Selecting specific columns to keep file size manageable
    selected_columns = [
        "game_id",
        "Next_Score_Half",
        "Drive_Score_Half",
        "play_type",
        "game_seconds_remaining",
        "half_seconds_remaining",
        "yardline_100",
        "roof",
        "posteam",
        "defteam",
        "home_team",
        "ydstogo",
        "season",
        "qtr",
        "down",
        "week",
        "drive",
        "ep",
        "score_differential",
        "posteam_timeouts_remaining",
        "defteam_timeouts_remaining",
        "desc",
        "receiver_player_name",
        "pass_location",
        "air_yards",
        "yards_after_catch",
        "complete_pass",
        "incomplete_pass",
        "interception",
        "qb_hit",
        "extra_point_result",
        "field_goal_result",
        "sp",
        "Winner",
        "spread_line",
        "total_line",
    ]
    filtered_pbp_data = filtered_pbp_data[selected_columns]
    cal_data_df = make_model_mutations(filtered_pbp_data)

    # Define label mapping
    label_mapping = {"Touchdown": 0, "Opp_Touchdown": 1, "Field_Goal": 2, "Opp_Field_Goal": 3, "Safety": 4, "Opp_Safety": 5, "No_Score": 6}

    # Apply label mapping
    cal_data_df["label"] = cal_data_df["Next_Score_Half"].map(label_mapping)
    cal_data_df["label"] = pd.factorize(cal_data_df["label"])[0]

    # Use nflscrapR weights
    cal_data_df["Drive_Score_Dist"] = cal_data_df["Drive_Score_Half"] - cal_data_df["drive"]
    cal_data_df["Drive_Score_Dist_W"] = (cal_data_df["Drive_Score_Dist"].max() - cal_data_df["Drive_Score_Dist"]) / (
        cal_data_df["Drive_Score_Dist"].max() - cal_data_df["Drive_Score_Dist"].min()
    )
    cal_data_df["ScoreDiff_W"] = (
        cal_data_df["score_differential"].dropna().abs().max() - cal_data_df["score_differential"].dropna().abs()
    ) / (cal_data_df["score_differential"].dropna().abs().max() - cal_data_df["score_differential"].dropna().abs().min())
    cal_data_df["Total_W"] = cal_data_df["Drive_Score_Dist_W"] + cal_data_df["ScoreDiff_W"]
    cal_data_df["Total_W_Scaled"] = (cal_data_df["Total_W"] - cal_data_df["Total_W"].dropna().min()) / (
        cal_data_df["Total_W"].dropna().max() - cal_data_df["Total_W"].dropna().min()
    )
    cal_data_df["Total_W_Scaled"] = cal_data_df["Total_W_Scaled"].fillna(0)

    return cal_data_df
