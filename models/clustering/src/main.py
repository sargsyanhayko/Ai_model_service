import argparse
import os
import pandas as pd
import numpy as np

from classical_clustering import ClassicalClusteringPipeline
from column_groups_definition import get_columns_definition
from utils import Config

def get_clustering_columns(columns_level1, columns_level2, columns_definition):
    clustering_columns = []
    for key, def_list in columns_definition.items():
        if not def_list:
            continue
        val = columns_level1.get(key, None) if isinstance(columns_level1, dict) else getattr(columns_level1, key, None)
        if val is not None:
            clustering_columns.extend(def_list)
    clustering_columns.extend(columns_level2)
    return list(dict.fromkeys(clustering_columns))

def subset_TINs(clustering_data, target_TINs, conf, unhash=False):
    TIN_subset_df = pd.DataFrame({"TIN": target_TINs})

    if unhash:
        tin_map_path = os.path.join(conf["paths"]["path_to_provided"], "hash/tins and hash_tin.xls")
        all_sheets = pd.read_excel(tin_map_path, sheet_name=None)
        tin_map = pd.concat(all_sheets.values(), ignore_index=True)
        TIN_subset_df = pd.merge(TIN_subset_df, tin_map, on="TIN", how="left")
        TIN_subset_df.rename(columns={"TIN": "UNHASH_TIN", "HASH_TIN": "TIN"}, inplace=True)
        print(TIN_subset_df)

    clustering_data = pd.merge(clustering_data, TIN_subset_df, on="TIN", how="inner")
    clustering_data = clustering_data.reset_index(drop=True)

    if unhash:
        clustering_data["TIN"] = clustering_data["UNHASH_TIN"]
    print(f"After subsetting to specific TINs: {clustering_data.shape[0]}")
    return clustering_data

def subset_year(clustering_data, year):
    year = int(year)
    sub_data = clustering_data[clustering_data.TAX_YEAR == year]
    print(f"TINs after subsetting by year {year}: {sub_data.TIN.nunique()}")
    if sub_data.TIN.nunique() < 5:
        raise ValueError("Too few TINs after subsetting by year")
    return sub_data

def find_TINs_from_sector(clustering_data, sector=None, min_tins=5):
    s = None if sector is None else str(sector).strip().upper()
    ac = clustering_data["ACTIVITY_CODE"].astype(str).str.upper().replace("NAN", "")
    sub = ac.str.rsplit(pat=".", n=1).str[0]
    root = ac.str.split(pat=".", n=1).str[0]

    if s is None:
        mask = pd.Series(True, index=clustering_data.index)
    elif len(s) == 1 and s.isalpha():
        mask = root.str.startswith(s, na=False)
    elif len(s) == 3 and s[0].isalpha() and s[1:].isdigit():
        mask = root == s
    elif "." in s and s[0].isalpha() and s[1:3].isdigit():
        mask = (sub == s) | (ac == s)
    else:
        raise ValueError("Invalid sector format.")

    df_final = clustering_data.loc[mask].copy()
    df_final["sector"] = sub.loc[mask]
    print(df_final["sector"].value_counts())

    tins = df_final["TIN"].unique()
    if tins.size < min_tins:
        raise ValueError(f"Too few TINs after subsetting by sector '{s}'")
    return tins

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск кластеризации")
    parser.add_argument("--adg", nargs="+", required=True, help="ADG")
    parser.add_argument("--tin", nargs="+", required=False, help="Список TIN")
    parser.add_argument("--year", type=int, required=False, help="Год")
    parser.add_argument("--path_to_plots", type=str, default="../plots", help="Папка для графиков")
    parser.add_argument("--input_path", type=str, default="input/input.json", help="Путь к input.json")
    parser.add_argument("--sector", type=str, required=False, help="Сектор")
    args = parser.parse_args()
    


    columns_definition, conf = get_columns_definition(args.input_path, args.adg)


    if args.tin:
        conf["TINs"] = args.tin
        print(args.tin)
    if args.year:
        conf["year"] = args.year
    if args.sector:
        conf["sector"] = args.sector

    output_path = args.path_to_plots
    n_clust = int(conf["n_clusters"]) if isinstance(conf["n_clusters"], str) else None

    clustering_data = pd.read_pickle(os.path.join(conf["paths"]["path_to_generated"], "clustering_data.pkl"))
    clustering_data = clustering_data.reset_index(drop=True)
    clustering_columns = get_clustering_columns(conf["columns_level1"], conf["columns_level2"], columns_definition)


    if "v17a" in clustering_columns and "v7a" in clustering_columns:
        clustering_data["import_ratio"] = clustering_data.apply(
            lambda x: x["v17a"] / x["v7a"] if x["v7a"] != 0 else np.nan, axis=1
        )
        clustering_columns.append("import_ratio")

    clustering_data["productivity"] = clustering_data.apply(
        lambda x: x["A41"] / x["AVG_N_EMPLOYEES"] if x["AVG_N_EMPLOYEES"] != 0 else np.nan, axis=1
    )


    if "TINs" in conf and conf["TINs"] is not None:
        clustering_data = subset_TINs(clustering_data, conf["TINs"], conf)
    elif "sector" in conf and conf["sector"] is not None:
        TINs = find_TINs_from_sector(clustering_data, conf["sector"])
        clustering_data = subset_TINs(clustering_data, TINs, conf)
    if "year" in conf and conf["year"] is not None:
        clustering_data = subset_year(clustering_data, conf["year"])


    if conf.get("target_column") is None:
        print("Running classical clustering")
        pipeline = ClassicalClusteringPipeline(
            data=clustering_data,
            features=clustering_columns,
            random_seed=conf["random_seed"],
            histogram_column=conf["histogram_column"],
            save_path=output_path
        )
        results, cluster_stats = pipeline.run(conf["cluster_range"], n_clust, plot=conf["plot"])
    else:
        from guided_clustering import GuidedClusteringPipeline
        os.environ["JULIA_NUM_THREADS"] = "12"
        pipeline = GuidedClusteringPipeline(
            conf["target_column"],
            clustering_data,
            clustering_columns,
            conf["random_seed"],
            conf["max_depth"],
            output_path,
            path_to_generated=conf["paths"]["path_to_generated"],
            histogram_column=conf["histogram_column"],
        )
        results, cluster_stats = pipeline.run(
            conf["cluster_range"],
            n_clust,
            conf["minbucket"],
            plots_folder=conf["paths"]["plots_folder"],
            user=conf["user"],
            save_model=conf["save_json"],
            plot=False,
        )
    n_clust = int(conf.n_clusters) if isinstance(conf["n_clusters"], str) else None

    pipeline.save_results(results, cluster_stats, n=10)
    pipeline.cluster_plots_interactive(results, cluster_stats, n_clusters=5)
