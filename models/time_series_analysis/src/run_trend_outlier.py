import os
from datetime import datetime

import pandas as pd
from tsfresh.feature_extraction import EfficientFCParameters

from time_series_analysis import TrendAnalyzer, clean_data, get_entity_column
from utils import Config

conf = Config("configs.json")

# Prepare data
grouped = pd.read_csv(
    os.path.join(conf.paths.path_to_generated, "subset_data/grouped.csv"),
    dtype={"TIN": str, "OIN": str, "ADG_CODE": str, "CRN": str},
)
grouped["TIN"] = grouped["TIN"].astype(str).str.zfill(8)
grouped = clean_data(grouped, conf)


outlier_results = pd.DataFrame()
outlier_df = pd.DataFrame()

entity_column = get_entity_column(conf)

trend_analyzer = TrendAnalyzer(entity_column)

default_fc_parameters = EfficientFCParameters()
features, trends = trend_analyzer.run_feature_extraction(
    data=grouped,
    save_path="outputs",
    time_column=conf.time_col,
    target_column=conf.target_col,
    default_fc_parameters=default_fc_parameters,
    mode="write",  # use mode write to compute and save the results, use mode read to load the results alreadt computed
)
df = features["NORMALIZED_trend"].copy()
outliers_result = trend_analyzer.run_outlier_detection(df, entity_column=entity_column)

trend_analyzer.plot_entity_outliers(
    trends=trends,
    outliers_result=outliers_result,
    save_path="",
    target_column=conf.target_col,
    sub_entity_id_col=entity_column,
    time_col="ds",
    together=True,
)
