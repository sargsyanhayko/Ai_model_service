import json
import os
import pandas as pd
import argparse

from time_series_analysis import GroupsAnalyzer, clean_data, get_entity_column
from models import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--adg", nargs="+", type=float, required=True)
    parser.add_argument("--path_to_plots", type=str, default="plots")
    parser.add_argument("--input_path", type=str, required=True)
    args = parser.parse_args()

    config_path = "configs.json"
    output_path = args.path_to_plots

    os.makedirs(output_path, exist_ok=True)


    with open(config_path, "r") as f:
        config = json.load(f)

    with open(args.input_path, "r") as f:
        input_data = json.load(f)


    config.update(input_data)
    config["year"] = args.year
    config["adg"] = args.adg
    config["path_to_plots"] = args.path_to_plots


    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    conf = Config.from_json_file(config_path)


    grouped = pd.read_csv(
        os.path.join(conf.paths.path_to_generated, "subset_data/grouped.csv"),
        dtype={"TIN": str, "OIN": str, "ADG_CODE": str, "CRN": str},
    )
    grouped["TIN"] = grouped["TIN"].astype(str).str.zfill(8)
    grouped = clean_data(grouped, conf)

    entity_column = get_entity_column(conf)
    print("ENTITY_COLUMN:", entity_column)

    groups_analyzer = GroupsAnalyzer(entity_column)

    outlier_results = pd.DataFrame()
    outlier_df = pd.DataFrame()

    def append_results(
        grouped_sub,
        entity_ID,
        entity_column,
        conf,
        outlier_results,
        outlier_df,
        results_dict=None,
    ):
        print(f"{entity_column} = {entity_ID}:")
        results = groups_analyzer.find_anomalies(
            data=grouped_sub,
            entity_ID=entity_ID,
            target_column=conf.target_col,
            time_column=conf.time_col,
            save_path=output_path,
            anomaly_factor=conf.anomaly_factor,
        )
        if results.empty:
            return outlier_results, outlier_df, results_dict

        results[entity_column] = entity_ID
        score = groups_analyzer.get_outlier_score(results)

        outlier_results = pd.concat([outlier_results, results], ignore_index=True)
        outlier_df = pd.concat(
            [
                outlier_df,
                pd.DataFrame({entity_column: [entity_ID], "OUTLIER_SCORE": [score]}),
            ],
            ignore_index=True,
        )
        results_dict[entity_ID] = results
        return outlier_results, outlier_df, results_dict

    results_dict = {}


    for TIN, oin_dict in conf.TINs.items():
        grouped_sub = grouped[grouped.TIN == TIN]
        if grouped_sub.shape[0] == 0:
            continue

        if entity_column == "TIN":
            outlier_results, outlier_df, results_dict = append_results(
                grouped_sub, TIN, entity_column, conf,
                outlier_results, outlier_df, results_dict
            )
        elif entity_column == "OIN":
            for OIN in oin_dict.keys():
                grouped_sub_oin = grouped_sub[grouped_sub.OIN == OIN]
                if grouped_sub_oin.shape[0] == 0:
                    continue
                outlier_results, outlier_df, results_dict = append_results(
                    grouped_sub_oin, OIN, entity_column, conf,
                    outlier_results, outlier_df, results_dict
                )
        elif entity_column == "CRN":
            for OIN, crn_list in oin_dict.items():
                grouped_sub_oin = grouped_sub[grouped_sub.OIN == OIN]
                if grouped_sub_oin.shape[0] == 0:
                    continue
                for CRN in crn_list:
                    grouped_sub_crn = grouped_sub_oin[grouped_sub_oin.CRN == CRN]
                    if grouped_sub_crn.shape[0] == 0:
                        continue
                    outlier_results, outlier_df, results_dict = append_results(
                        grouped_sub_crn, CRN, entity_column, conf,
                        outlier_results, outlier_df, results_dict
                    )
        else:
            raise ValueError(f"Unsupported entity_column: {entity_column}")

    if outlier_df.shape[0] == 0:
        print("No data for the selected TIN/OIN/CRNs. Please check data")
        return

    if conf.plot_temporal:
        groups_analyzer.plot_temporal_anomalies_all(results_dict, save_path=output_path)
        output_file = os.path.join(output_path, "anomalies_plot_all_data.csv")
        pd.concat(results_dict).reset_index(level=0).to_csv(output_file, index=False)

    outlier_df.sort_values("OUTLIER_SCORE", ascending=False, inplace=True)
    outlier_df["OUTLIER_RANKING"] = range(outlier_df.shape[0])
    outlier_df["OUTLIER"] = outlier_df.apply(
        lambda x: x.OUTLIER_SCORE > conf.entity_outlier_threshold
        and x.OUTLIER_RANKING <= 4,
        axis=1,
    )

    print(outlier_df)

    if conf.plot_entity:
        groups_analyzer.plot_entity_outliers(
            outlier_results,
            outlier_df,
            save_path=output_path,
            target_column=conf.target_col,
            sub_entity_id_col=entity_column,
        )
        output_file_results = os.path.join(output_path, "anomalies_plot_multi_results.csv")
        output_file_df = os.path.join(output_path, "anomalies_plot_multi_df.csv")
        outlier_results.to_csv(output_file_results, index=False)
        outlier_df.to_csv(output_file_df, index=False)


if __name__ == "__main__":
    main()
