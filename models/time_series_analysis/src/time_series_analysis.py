import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Union
import sys
import cmdstanpy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_percentage_error
from tqdm import tqdm
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

from utils import Config



def clean_data(grouped, conf):
    # Subset to ADG_CODE
    # TODO: support other product category input / filters
    if conf.adg_code:
        grouped = grouped[grouped.ADG_CODE == conf.adg_code]
    # Subset to the right product
    if conf.sub_category_ARM_1:
        grouped = grouped[grouped["sub_category_ARM.1"] == conf.sub_category_ARM_1]

    if conf.category_score:
        grouped = grouped[grouped.category_score >= conf.category_score]

    if conf.brand_name:
        grouped = grouped[grouped.brand_name == conf.brand_name]

    if conf.TINs:
        # Convert SimpleNamespace to dictionary to get keys
        if hasattr(conf.TINs, '__dict__'):
            tin_keys = list(conf.TINs.__dict__.keys())
        else:
            tin_keys = list(conf.TINs.keys())
        grouped = grouped[grouped.TIN.isin(tin_keys)]


    # Subset to within the days_lookback
    grouped["RECEIPT_TIME_HOUR"] = pd.to_datetime(grouped["RECEIPT_TIME_HOUR"])
    print(f"Subsetting by time: {conf.start_date} to {conf.end_date}")
    print(grouped.shape)
    grouped = grouped[
        (grouped.RECEIPT_TIME_HOUR >= datetime.fromisoformat(conf.start_date))
        & (grouped.RECEIPT_TIME_HOUR <= datetime.fromisoformat(conf.end_date))
    ]
    print(grouped.shape)
    # Aggregate to daily
    if conf.time_col == "RECEIPT_TIME_DAY":
        conf.aggregation_level = "daily"
        grouped["RECEIPT_TIME_DAY"] = grouped["RECEIPT_TIME_HOUR"].dt.date
        grouped = (
            grouped.groupby(["TIN", "OIN", "CRN", "RECEIPT_TIME_DAY", "ADG_CODE"])
            .agg(
                QANTITY_sum=("QANTITY_sum", "sum"),
                TOTAL_WITHOUT_TAXES_sum=("TOTAL_WITHOUT_TAXES_sum", "sum"),
                TOTAL_WITH_TAXES_sum=("TOTAL_WITH_TAXES_sum", "sum"),
                count=("count", "sum"),
            )
            .reset_index()
        )
    elif conf.time_col == "RECEIPT_TIME_HOUR":
        conf.aggregation_level = "hourly"
    else:
        raise ValueError(
            f"Unsupported `time_col`: {conf.time_col}. Only `RECEIPT_TIME_DAY` or `RECEIPT_TIME_HOUR`"
        )

    if grouped.shape[0] == 0:
         raise ValueError("No observation left after subsetting to brand/daterange")
    return grouped

def get_entity_column(conf):
    # Check if we have a nested structure in TINs
    if conf.TINs:
        # Convert SimpleNamespace to dictionary to inspect structure
        if hasattr(conf.TINs, '__dict__'):
            tins_dict = conf.TINs.__dict__
        else:
            tins_dict = conf.TINs
        
        has_oins = False
        has_crns = False
        
        for tin_key, tin_value in tins_dict.items():
            if hasattr(tin_value, '__dict__'):
                oin_dict = tin_value.__dict__
            else:
                oin_dict = tin_value
            
            if oin_dict: 
                has_oins = True
                for oin_key, crn_list in oin_dict.items():
                    if crn_list:
                        has_crns = True
                        break
                break
        
        if has_crns:
            return "CRN"
        elif has_oins:
            return "OIN"
        else:
            return "TIN"
    else:
        if hasattr(conf, 'OINs') and conf.OINs is not None:
            if hasattr(conf, 'CRNs') and conf.CRNs is not None:
                return "CRN"
            else:
                return "OIN"
        else:
            return "TIN"


class BaseAnalyzer:
    def __init__(self, entity_column):
        self.entity_column = entity_column

    def create_full_time_range(
        self, df: pd.DataFrame, time_column: str
    ) -> pd.DataFrame:
        start_time, end_time = df[time_column].min(), df[time_column].max()
        return pd.DataFrame(
            pd.date_range(
                start=start_time,
                end=end_time,
                freq="D" if "DAY" in time_column else "h",
            ),
            columns=[time_column],
        )

    def decompose_time_series(
        self,
        df: pd.DataFrame,
        time_column: str,
        target_column: str = "TOTAL_WITH_TAXES_sum",
        seasonality_mode: str = "multiplicative",
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        monthly_seasonality: bool = True,
        country_holidays: bool = True,
        anomaly_factor: float = 1.5,
    ) -> pd.DataFrame:
        assert len(df) == len(
            df[time_column].unique()
        ), "The index contains duplicate values"

        df = df.copy()
        df["ds"] = df[time_column]
        df["y"] = df[target_column]
        df = df[["ds", "y"]].copy()

        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=False,
            seasonality_mode=seasonality_mode,
            changepoint_range=1,
            changepoint_prior_scale=1,
        )
        if daily_seasonality and "DAY" not in time_column:
            model.add_seasonality(name="daily", period=1, fourier_order=7)
        if weekly_seasonality and "DAY" not in time_column:
            model.add_seasonality(name="weekly", period=7, fourier_order=7)
        if monthly_seasonality and df.shape[0] > 30.5:
            model.add_seasonality(name="monthly", period=30.5, fourier_order=7)
        if country_holidays:
            model.add_country_holidays(country_name="AM")

        model.fit(df)

        future = model.make_future_dataframe(periods=0)
        forecast = model.predict(future)
        forecast = pd.merge(forecast, df, on="ds", how="inner")
        forecast["error"] = forecast["y"] - forecast["yhat"]
        forecast["yhat_lower"] = forecast["yhat_lower"].apply(lambda x: max(x, 1))
        forecast["yhat_upper"] = forecast["yhat_upper"].apply(lambda x: max(x, 1))
        forecast["yhat"] = forecast["yhat"].apply(lambda x: max(x, 0))
        forecast["uncertainty"] = forecast["yhat_upper"] - forecast["yhat_lower"]
        forecast["ratio"] = forecast.apply(
            lambda x: np.abs(
                x["error"] / x["uncertainty"] if x["uncertainty"] != 0 else 0
            ),
            axis=1,
        )
        forecast["anomaly"] = forecast["ratio"].apply(lambda x: x > anomaly_factor)

        return forecast

    def _prepare_data_for_analysis(
        self,
        data: pd.DataFrame,
        time_column: str,
        target_column: str = "TOTAL_WITH_TAXES_sum",
    ) -> pd.DataFrame:
        full_time_range = self.create_full_time_range(data, time_column)
        prepared_data = []
        idx_entity_map = {}
        for idx, entity_id in enumerate(
            tqdm(
                data[self.entity_column].unique(),
                desc=f"Processing {self.entity_column}",
            )
        ):
            entity_data = data[data[self.entity_column] == entity_id]
            aggregated_data = self._aggregate_entity_data(
                entity_data=entity_data,
                full_time_range=full_time_range,
                time_column=time_column,
            )
            forecast = self.decompose_time_series(
                df=aggregated_data,
                time_column=time_column,
                target_column=target_column,
                seasonality_mode="multiplicative",
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=True,
                monthly_seasonality=True,
                country_holidays=True,
            )
            trend = self.prepare_trend_for_feature_extraction(
                df=forecast, trend_column="trend", time_column="ds"
            )
            trend[self.entity_column] = entity_id
            prepared_data.append(trend)
            idx_entity_map[idx] = entity_id

        return pd.concat(prepared_data), idx_entity_map

    def _aggregate_entity_data(
        self,
        entity_data: pd.DataFrame,
        full_time_range: pd.DataFrame,
        time_column: str,
    ) -> pd.DataFrame:
        aggregated = entity_data.groupby(time_column).agg(
            {
                "TOTAL_WITH_TAXES_sum": "sum",
                "QANTITY_sum": "sum",
                "count": "sum",
                self.entity_column: "first",
            }
        )
        aggregated["TOTAL_WITH_TAXES_mean"] = (
            aggregated["TOTAL_WITH_TAXES_sum"] / aggregated["QANTITY_sum"]
        )
        aggregated["QANTITY_mean"] = aggregated["QANTITY_sum"] / aggregated["count"]
        aggregated["UNIT_PRICE_mean"] = (
            aggregated["TOTAL_WITH_TAXES_mean"] / aggregated["QANTITY_mean"]
        )
        aggregated = (
            aggregated.replace([np.inf, -np.inf], np.nan).fillna(0).reset_index()
        )
        aggregated[time_column] = pd.to_datetime(aggregated[time_column])
        aggregated = pd.merge(full_time_range, aggregated, on=time_column, how="left")
        aggregated[self.entity_column] = aggregated[self.entity_column].fillna(
            aggregated[self.entity_column].mode()[0]
        )
        return aggregated.fillna(0)

    def prepare_trend_for_feature_extraction(
        self, df: pd.DataFrame, trend_column: str, time_column: str = "ds"
    ) -> pd.DataFrame:
        # trend = df[[time_column, trend_column]].copy()
        trend = df
        trend["date_idx"] = trend[time_column].rank(method="dense").sub(1).astype(int)
        trend[f"MEAN_{trend_column}"] = trend[trend_column].mean()
        trend[f"STD_{trend_column}"] = trend[trend_column].std()
        trend[f"STANDARDIZED_{trend_column}"] = (
            trend[trend_column] - trend[f"MEAN_{trend_column}"]
        ) / trend[f"STD_{trend_column}"]
        trend[f"NORMALIZED_{trend_column}"] = (
            trend[trend_column] / trend[f"MEAN_{trend_column}"]
        )

        return trend

    def analyze_zero_intervals(
        self, df: pd.DataFrame, time_column: str = "ds", target_column: str = "y"
    ) -> Dict[int, int]:
        assert len(df) == len(
            df[time_column].unique()
        ), "The index contains duplicate values"
        df = df.copy()
        df[time_column] = pd.to_datetime(df[time_column])
        df_sorted = df.sort_values(time_column).reset_index(drop=True)

        zero_intervals = defaultdict(int)
        current_interval = 0
        in_zero_sequence = False

        for _, row in df_sorted.iterrows():
            if row[target_column] == 0:
                if not in_zero_sequence:
                    in_zero_sequence = True
                    current_interval = 1
                else:
                    current_interval += 1
            else:
                if in_zero_sequence:
                    zero_intervals[current_interval] += 1
                    in_zero_sequence = False
                    current_interval = 0

        if in_zero_sequence:
            zero_intervals[current_interval] += 1

        return dict(zero_intervals)

    def plot_entity_outliers(
        self,
        trends: pd.DataFrame,
        outliers_result: pd.DataFrame,
        save_path: str,
        target_column: str,
        sub_entity_id_col: str = "OIN",
        time_col: str = "ds",
        together: bool = True,
    ) -> None:
        trends = trends.merge(
            outliers_result[[sub_entity_id_col, "OUTLIER", "OUTLIER_SCORE"]],
            on=sub_entity_id_col,
            how="left",
        )
        outliers_df = trends[trends["OUTLIER"] == 1]
        non_outliers_df = trends[trends["OUTLIER"] == 0]
        trend_types = ["y", "STANDARDIZED_trend"]

        fig = make_subplots(
            rows=1,
            cols=1 if together else 2,
            subplot_titles=None if together else ("Non-Outliers", "Outliers"),
            horizontal_spacing=0.05,
            shared_yaxes=True,
        )

        for trend_type in trend_types:
            self._add_traces(
                fig,
                non_outliers_df,
                trend_type,
                sub_entity_id_col,
                time_col,
                "Inlier",
                row=1,
                col=1,
                together=together,
            )
            self._add_traces(
                fig,
                outliers_df,
                trend_type,
                sub_entity_id_col,
                time_col,
                "Outlier",
                row=1,
                col=1 if together else 2,
                together=together,
            )

        outlier_selection_buttons = self._create_outlier_selection_buttons(
            fig, outliers_df, together, sub_entity_id_col
        )
        trend_selection_buttons = self._create_trend_selection_buttons(fig, trend_types)
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=outlier_selection_buttons,
                    direction="down",
                    showactive=True,
                    x=1.01,
                    y=1,
                    xanchor="left",
                    yanchor="top",
                    pad={"r": 10, "t": 10},
                    active=0,
                    type="dropdown",
                ),
                dict(
                    buttons=trend_selection_buttons,
                    direction="down",
                    showactive=True,
                    x=1.01,
                    y=0.95,
                    xanchor="left",
                    yanchor="top",
                    pad={"r": 10, "t": 10},
                    active=0,
                    type="dropdown",
                ),
            ],
            yaxis_title=f"{target_column} (raw or standardized)",
            showlegend=False,
            height=800,
            width=2000,
            title_text=f"Outlier detection results",
            margin=dict(t=100, r=200),
            yaxis2=dict(
                showticklabels=True,
            ),
        )
        if save_path:
            output_file = f"{save_path}/anomalies_plot_multi.html"
            fig.write_html(output_file)

    def _add_traces(
        self,
        fig: go.Figure,
        df: pd.DataFrame,
        trend_type: str,
        sub_entity_id_col: str,
        time_col: str,
        name_suffix: str,
        row: int,
        col: int,
        together: bool = False,
    ) -> None:
        blue = "lightskyblue" if together else "blue"
        color = "red" if name_suffix == "Outlier" else blue

        for sub_entity in df[sub_entity_id_col].unique():
            value = df.loc[df[sub_entity_id_col] == sub_entity, [time_col, trend_type]]
            # If we didn't find this entity
            if value.shape[0] == 1 and np.isnan(value.iloc[0][trend_type]):
                continue
            fig.add_trace(
                go.Scatter(
                    x=value[time_col],
                    y=value[trend_type],
                    mode="lines",
                    line=dict(color=color),
                    name=f'{sub_entity} ({name_suffix}) - {trend_type.split("_")[0].lower()}',
                    opacity=0.5,
                    hoverinfo="skip",
                    visible=(trend_type == "y"),
                ),
                row=row,
                col=col,
            )

    def _create_outlier_selection_buttons(
        self,
        fig: go.Figure,
        outliers_df: pd.DataFrame,
        together: bool = False,
        sub_entity_id_col: str = "OIN",
    ) -> List[Dict[str, str]]:
        outlier_selection_buttons = []
        line_color_all = (
            [
                "lightskyblue"
                if "Inlier" in trace.name
                else "red"
                if "Outlier" in trace.name
                else "grey"
                for trace in fig.data
            ]
            if together
            else [
                "blue"
                if "Inlier" in trace.name
                else "red"
                if "Outlier" in trace.name
                else "grey"
                for trace in fig.data
            ]
        )
        outlier_selection_buttons.append(
            dict(
                label="select",
                method="update",
                args=[
                    {
                        "line.color": line_color_all,
                        "line.width": [2 for trace in fig.data],
                    },
                    {"title": f"Outlier detection results"},
                ],
            )
        )
        buttons = []
        for sub_entity in outliers_df[sub_entity_id_col].unique():
            sub_entity = sub_entity
            line_color_selection = (
                [
                    "red"
                    if f"{sub_entity}" in trace.name
                    else "grey"
                    if "Outlier" in trace.name
                    else "grey"
                    for trace in fig.data
                ]
                if together
                else [
                    "red"
                    if f"{sub_entity}" in trace.name
                    else "blue"
                    if "Inlier" in trace.name
                    else "grey"
                    for trace in fig.data
                ]
            )
            outlier_score = outliers_df.loc[
                outliers_df[sub_entity_id_col] == sub_entity, "OUTLIER_SCORE"
            ].iloc[0]
            buttons.append(
                dict(
                    label=f"{sub_entity} ({outlier_score:.2f})",
                    method="update",
                    args=[
                        {
                            "line.color": line_color_selection,
                            "line.width": [
                                4 if f"{sub_entity}" in trace.name else 2
                                for trace in fig.data
                            ],
                        },
                        {
                            "title": f"Outlier detection results -- Highlighting {sub_entity}"
                        },
                    ],
                )
            )
        buttons = sorted(
            buttons,
            key=lambda x: float(
                x["label"].split(" ")[1].replace("(", "").replace(")", "").strip()
            ),
            reverse=True,
        )
        outlier_selection_buttons.extend(buttons)

        return outlier_selection_buttons

    def _create_trend_selection_buttons(
        self, fig: go.Figure, trend_types: List[str]
    ) -> List[Dict[str, str]]:
        trend_buttons = []
        for trend_type in trend_types:
            trend_type = trend_type.split("_")[0].lower()
            visibility = [(trace.name).endswith(trend_type) for trace in fig.data]
            trend_buttons.append(
                dict(
                    label=trend_type,
                    method="update",
                    args=[
                        {"visible": visibility},
                        {
                            "title": f"Outlier detection results -- visualizing {trend_type} trend"
                        },
                    ],
                )
            )

        return trend_buttons


# Identifies entities that have different trend behaviors by:
# 1. decompose the time series into trend, seasonality, noise
# 2. normalize or standardize trend
# 3. generate tsfresh features on the trend
# 4. run isolation-forest to identify trends that have very different features
class TrendAnalyzer(BaseAnalyzer):
    def extract_tsfresh_features(
        self,
        df: pd.DataFrame,
        column_id: str,
        save_path: str,
        default_fc_parameters: Dict[
            str, Dict[str, Union[str, int]]
        ] = MinimalFCParameters(),
        mode: str = "write",
    ) -> pd.DataFrame:
        features = {
            feature_type: None
            for feature_type in ["STANDARDIZED_trend", "NORMALIZED_trend"]
        }
        for column_name in features.keys():
            col_name_mapped = column_name.lower().split("_")[0]
            path = f"{save_path}/{column_id}_level/extracted_{col_name_mapped}.csv"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if mode == "write":
                os.makedirs(save_path, exist_ok=True)
                extracted = extract_features(
                    df,
                    column_id=column_id,
                    column_sort="date_idx",
                    column_value=column_name,
                    default_fc_parameters=default_fc_parameters,
                    n_jobs=1,
                )
                extracted.to_csv(path, index=True)
            else:
                extracted = pd.read_csv(path, index_col=0)
            features[column_name] = extracted

        return features

    def run_feature_extraction(
        self,
        data: pd.DataFrame,
        save_path: str,
        time_column: str,
        target_column: str = "TOTAL_WITH_TAXES_sum",
        default_fc_parameters: Dict[
            str, Dict[str, Union[str, int]]
        ] = MinimalFCParameters(),
        mode: str = "read",
    ) -> Dict[str, pd.DataFrame]:
        original_level = cmdstanpy.utils.get_logger().level
        cmdstanpy.utils.get_logger().setLevel(logging.CRITICAL)
        try:
            prepared_data, idx_entity_map = self._prepare_data_for_analysis(
                data, time_column, target_column
            )
            features = self.extract_tsfresh_features(
                df=prepared_data,
                column_id=self.entity_column,
                save_path=save_path,
                default_fc_parameters=default_fc_parameters,
                mode=mode,
            )
        finally:
            cmdstanpy.utils.get_logger().setLevel(original_level)

        return features, prepared_data

    def run_outlier_detection(
        self,
        df: pd.DataFrame,
        entity_column: str = "OIN",
        contamination: Union[float, str] = "auto",
    ) -> pd.DataFrame:
        df = df.copy()
        df = df.loc[:, ~(df.isna().sum() > 0)]
        isolation_forest = IsolationForest(contamination=contamination, random_state=42)
        isolation_forest = isolation_forest.fit(df)
        df["OUTLIER_SCORE"] = isolation_forest.decision_function(df)
        df["OUTLIER"] = (df["OUTLIER_SCORE"] < 0).astype(int)

        df[entity_column] = df.index.astype(str)
        return df[[entity_column, "OUTLIER", "OUTLIER_SCORE"]]


# Outlier detection that finds specific times within an entity
# 1. decompose the time series into trend, seasonality, noise
# 2. denote times where the noise is very different from the predictions
# We then identify the entities with the highest outlier score
# 1. for an entity, aggregate outlier score by summing the deviation
# 2. rank them and denote those very high scores outlier entities
class GroupsAnalyzer(BaseAnalyzer):
    def get_closing_times(self, data: pd.DataFrame, full_time_range: pd.DataFrame):
        data = data[["RECEIPT_TIME_HOUR", "TOTAL_WITH_TAXES_sum"]].copy()
        data["RECEIPT_TIME_HOUR"] = pd.to_datetime(data["RECEIPT_TIME_HOUR"])
        data = data.sort_values("RECEIPT_TIME_HOUR")
        data = pd.merge(full_time_range, data, on="RECEIPT_TIME_HOUR", how="left")

        return data[data["TOTAL_WITH_TAXES_sum"].isnull()].RECEIPT_TIME_HOUR

    def find_anomalies(
        self,
        data: pd.DataFrame,
        entity_ID: str,
        time_column: str,
        target_column: str = "TOTAL_WITH_TAXES_sum",
        save_path: str = "",
        anomaly_factor: float = 1.5,
    ):
        self.entity_ID = entity_ID
        full_time_range = self.create_full_time_range(data, time_column=time_column)
        entity_data = data[data[self.entity_column] == entity_ID]

        if entity_data.shape[0] == 0:
            print(f"Skipping {entity_ID}, no data")
            return pd.DataFrame()
        # self.closing_times = self.get_closing_times(data, full_time_range)
        entity_data = self._aggregate_entity_data(
            entity_data=entity_data,
            full_time_range=full_time_range,
            time_column=time_column,
        )

        if entity_data.shape[0] < 2:
            print(f"Skipping {entity_ID}, < 2 obs after aggregation")
            return pd.DataFrame()
        forecast = self.decompose_time_series(
            df=entity_data,
            time_column=time_column,
            target_column=target_column,
            seasonality_mode="multiplicative",
            yearly_seasonality=False,
            weekly_seasonality=True,
            daily_seasonality=True,
            monthly_seasonality=True,
            country_holidays=True,
            anomaly_factor=anomaly_factor,
        )
        forecast = self.prepare_trend_for_feature_extraction(
            df=forecast, trend_column="trend", time_column="ds"
        )

        df = forecast
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            # TODO: figure out what to do with closing times
            # df.loc[~df["ds"].isin(self.closing_times), "anomaly"] = False
            df[
                ["ds", "yhat", "y", "error", "error", "uncertainty", "ratio", "anomaly"]
            ].rename(
                columns={
                    "ds": time_column,
                    "yhat": "prediction",
                    "y": "actual",
                }
            ).to_csv(
                os.path.join(
                    save_path, f"time_series_anomalies_results_{entity_ID}.csv"
                ),
                index=False,
            )
        return df

    def get_outlier_score(self, df):
        print(f"Number of outliers: {df.anomaly.sum()}")
        score = (df["ratio"] * df["anomaly"]).sum()
        print(f"Score: {score}")
        return score


    def plot_temporal_anomalies_all(self, forecast_dict, save_path):
        fig = go.Figure()
        buttons = []
        first_entity_id = list(forecast_dict.keys())[0]
        
        for entity_id, forecast in forecast_dict.items():
            anomalies = forecast[forecast["anomaly"] == 1]
            forecast["ds"] = pd.to_datetime(forecast["ds"])
            forecast["month_year"] = forecast["ds"].dt.to_period("2W")
            # month_map = {
            #     1: "Հունվար", 2: "Փետրվար", 3: "Մարտ", 4: "Ապրիլ", 5: "Մայիս", 6: "Հունիս",
            #     7: "Հուլիս", 8: "Օգոստոս", 9: "Սեպտեմբեր", 10: "Հոկտեմբեր", 11: "Նոյեմբեր", 12: "Դեկտեմբեր"
            # }
            # forecast["month_name"] = forecast["ds"].dt.month.map(month_map)
            # forecast["year"] = forecast["ds"].dt.year
            # forecast["day"] = forecast["ds"].dt.day
            # forecast["month_year_label"] = (
            #     forecast["month_name"] + " " + forecast["day"].astype(str) + " " + forecast["year"].astype(str)
            # )

            unique_months = forecast.drop_duplicates(subset="month_year")
            tickvals = unique_months["ds"]
            # ticktext = unique_months["month_year_label"]
            ticktext = unique_months["ds"].apply(lambda x: x.strftime("%Y-%m-%d"))
            
            fig.add_trace(
                go.Scatter(
                    x=forecast["ds"], y=forecast["y"], mode="lines", name=f"Փաստացի - {entity_id}",
                    line=dict(color="blue"), visible=(entity_id == first_entity_id)
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast["ds"], y=forecast["yhat"], mode="lines", name=f"Կանխատեսված - {entity_id}",
                    line=dict(color="red"), visible=(entity_id == first_entity_id)
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast["ds"].tolist() + forecast["ds"].tolist()[::-1],
                    y=forecast["yhat_upper"].tolist() + forecast["yhat_lower"].tolist()[::-1],
                    fill="toself", fillcolor="rgba(128,128,128,0.2)",
                    line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", showlegend=False,
                    visible=(entity_id == first_entity_id)
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=anomalies["ds"], y=anomalies["y"], mode="markers",
                    name=f"Անոմալիաներ - {entity_id}", marker=dict(color="green", size=8),
                    hovertemplate="Ամսաթիվ: %{x|%Y-%m-%d %H:%M:%S}<br>Արժեք: %{y:.2f}<extra></extra>",
                    visible=(entity_id == first_entity_id)
                )
            )

            buttons.append(
                dict(
                    label=str(entity_id),
                    method="update",
                    args=[
                        {"visible": [eid == entity_id for eid in forecast_dict.keys() for _ in range(4)]},
                        {
                            "xaxis": dict(tickvals=tickvals, ticktext=ticktext),
                            "yaxis": dict(range=[0, anomalies["y"].max() * 1.1]),
                        },
                    ]
                )
            )

        fig.update_layout(
            title="Փաստացի և Կանխատեսված արժեքներ՝ տարբեր Entity_ID-ների համար",
            xaxis_title="Ամսաթիվ",
            yaxis_title="Արժեք",
            updatemenus=[
                dict(
                    buttons=buttons,
                    direction="down",
                    showactive=True,
                    x=1,
                    xanchor="right",
                    y=1.15,
                    yanchor="top"
                )
            ],
            height=600,
            hovermode="x",
        )

        fig.update_yaxes(range=[0, max(forecast["y"].max() for forecast in forecast_dict.values()) * 1.1])
        
        if save_path:
            output_file = os.path.join(save_path, "anomalies_plot_all.html")
            fig.write_html(output_file)
    

    def plot_temporal_anomalies(self, forecast, save_path):
        anomalies = forecast[forecast["anomaly"] == 1]
        # anomalies = anomalies[~anomalies["ds"].isin(self.closing_times)]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=forecast["ds"],
                y=forecast["y"],
                mode="lines",
                name="Փաստացի",
                line=dict(color="blue"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["ds"],
                y=forecast["yhat"],
                mode="lines",
                name="Կանխատեսված",
                line=dict(color="red"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["ds"].tolist() + forecast["ds"].tolist()[::-1],
                y=forecast["yhat_upper"].tolist()
                + forecast["yhat_lower"].tolist()[::-1],
                fill="toself",
                fillcolor="rgba(128,128,128,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                name="Անորոշություն",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=anomalies["ds"],
                y=anomalies["y"],
                mode="markers",
                name="Անոմալիաներ",
                marker=dict(color="green", size=8),
                hovertemplate="Ամսաթիվ: %{x|%Y-%m-%d %H:%M:%S}<br>Արժեք: %{y:.2f}<extra></extra>",
            )
        )
        forecast["ds"] = pd.to_datetime(forecast["ds"])
        forecast["month_year"] = forecast["ds"].dt.to_period("2W")

        # Create a dictionary mapping month-year to Armenian month names
        month_map = {
            1: "Հունվար",
            2: "Փետրվար",
            3: "Մարտ",
            4: "Ապրիլ",
            5: "Մայիս",
            6: "Հունիս",
            7: "Հուլիս",
            8: "Օգոստոս",
            9: "Սեպտեմբեր",
            10: "Հոկտեմբեր",
            11: "Նոյեմբեր",
            12: "Դեկտեմբեր",
        }
        forecast["month_name"] = forecast["ds"].dt.month.map(month_map)
        forecast["year"] = forecast["ds"].dt.year
        forecast["day"] = forecast["ds"].dt.day  # Extract day
        forecast["month_year_label"] = (
            forecast["month_name"]
            + " "
            + forecast["day"].astype(str)
            + " "
            + forecast["year"].astype(str)
        )

        # Get unique month-year labels and their corresponding first occurrence index
        unique_months = forecast.drop_duplicates(subset="month_year")
        tickvals = unique_months["ds"]
        ticktext = unique_months["month_year_label"]

        fig.update_layout(
            title=f"Փաստացի և Կանխատեսված (անորոշությամբ և հայտնաբերված անոմալիաներով) արժեքներ՝ {self.entity_column} {self.entity_ID}-ի համար",
            xaxis_title="Ամսաթիվ",
            yaxis_title="Արժեք",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            height=600,
            hovermode="x",
            xaxis=dict(
                showspikes=True,
                spikecolor="gray",
                spikesnap="cursor",
                spikemode="across",
                spikethickness=1,
                tickvals=tickvals,
                ticktext=ticktext,
            ),
        )
        fig.update_yaxes(range=[0, forecast["y"].max() * 1.1])
        if save_path:
            output_file = f"{save_path}/anomalies_plot_{self.entity_ID}.html"
            fig.write_html(output_file)
